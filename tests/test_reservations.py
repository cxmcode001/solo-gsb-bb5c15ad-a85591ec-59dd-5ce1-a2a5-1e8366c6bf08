"""预留模块端到端测试:整单事务、幂等重放、整单释放与余量查询。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db import connect, init_db
from app.main import app, get_conn


@pytest.fixture()
def client():
    mem = connect(":memory:")
    init_db(mem)

    def _override():
        yield mem

    app.dependency_overrides[get_conn] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    mem.close()


def _sample(client, op_no, sid, amount, unit="mL"):
    r = client.post("/api/v1/operations/sampling", json={
        "type": "sampling", "op_no": op_no, "sample_id": sid,
        "material": "M", "amount": amount, "unit": unit,
    })
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------- 创建与查询

def test_reserve_multiple_samples_atomic_views(client):
    _sample(client, "OP-1", "S1", 100)
    _sample(client, "OP-2", "S2", 50, unit="mg")

    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-1",
        "items": [
            {"sample_id": "S1", "amount": 30},
            {"sample_id": "S2", "amount": 20},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reservation_no"] == "R-1"
    assert body["status"] == "active"
    assert body["replayed"] is False
    by_id = {i["sample_id"]: i for i in body["items"]}
    assert by_id["S1"]["amount"] == 30.0
    assert by_id["S1"]["remaining"] == 100.0
    assert by_id["S1"]["reserved"] == 30.0
    assert by_id["S1"]["available"] == 70.0
    assert by_id["S2"]["unit"] == "mg"
    assert by_id["S2"]["reserved"] == 20.0
    assert by_id["S2"]["available"] == 30.0

    # 查询接口:当前余量 / 已预留 / 可用
    r = client.get("/api/v1/samples/S1/availability")
    assert r.json() == {
        "sample_id": "S1", "unit": "mL",
        "remaining": 100.0, "reserved": 30.0, "available": 70.0,
    }

    # 预留单可按号查询
    r = client.get("/api/v1/reservations/R-1")
    assert r.status_code == 200
    assert r.json()["status"] == "active"
    assert len(r.json()["items"]) == 2

    r = client.get("/api/v1/reservations/NOPE")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"

    r = client.get("/api/v1/samples/GHOST/availability")
    assert r.status_code == 409
    assert r.json()["conflicts"][0]["code"] == "sample_not_found"


def test_reserve_accumulates_and_blocks_when_insufficient(client):
    _sample(client, "OP-1", "S1", 100)

    # 首张预留 60,可用降到 40
    client.post("/api/v1/reservations", json={
        "reservation_no": "R-1",
        "items": [{"sample_id": "S1", "amount": 60}],
    })

    # 第二张预留 40,恰好足够,可用降为 0
    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-2",
        "items": [{"sample_id": "S1", "amount": 40}],
    })
    assert r.status_code == 200, r.text
    item = r.json()["items"][0]
    assert item["reserved"] == 100.0
    assert item["available"] == 0.0

    # 再预留 0.000000001 即不足:整单失败
    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-3",
        "items": [
            {"sample_id": "S1", "amount": 1},
        ],
    })
    assert r.status_code == 409
    c = r.json()["conflicts"][0]
    assert c["code"] == "insufficient_available"
    assert c["sample_id"] == "S1"
    assert c["requested"] == 1.0
    assert c["available"] == 0.0

    # 失败的预留没有落库,已预留量不变
    assert client.get("/api/v1/reservations/R-3").status_code == 404
    assert client.get("/api/v1/samples/S1/availability").json()["reserved"] == 100.0


def test_reserve_all_or_nothing_rollback(client):
    # 两个样品,S1 足够、S2 不足 -> 整单失败,S1 也不被锁定
    _sample(client, "OP-1", "S1", 100)
    _sample(client, "OP-2", "S2", 10)

    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-BAD",
        "items": [
            {"sample_id": "S1", "amount": 90},
            {"sample_id": "S2", "amount": 50},
            {"sample_id": "GHOST", "amount": 1},
        ],
    })
    assert r.status_code == 409
    codes = {c["code"] for c in r.json()["conflicts"]}
    assert codes == {"insufficient_available", "sample_not_found"}
    bad_sids = {c["sample_id"] for c in r.json()["conflicts"]}
    assert bad_sids == {"S2", "GHOST"}

    # 整单回滚:S1 未被预留
    avail = client.get("/api/v1/samples/S1/availability").json()
    assert avail["reserved"] == 0.0
    assert avail["available"] == 100.0
    assert client.get("/api/v1/reservations/R-BAD").status_code == 404


# ---------------------------------------------------------------- 幂等

def test_reserve_idempotent_replay(client):
    _sample(client, "OP-1", "S1", 100)
    body = {
        "reservation_no": "R-1",
        "items": [{"sample_id": "S1", "amount": 40}],
    }
    r1 = client.post("/api/v1/reservations", json=body)
    assert r1.status_code == 200
    assert r1.json()["replayed"] is False

    # 同号同体重放:不重复锁定
    r2 = client.post("/api/v1/reservations", json=body)
    assert r2.status_code == 200
    assert r2.json()["replayed"] is True
    assert r2.json()["items"][0]["amount"] == 40.0

    avail = client.get("/api/v1/samples/S1/availability").json()
    assert avail["reserved"] == 40.0  # 没有翻倍

    # 同号不同体:幂等键冲突
    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-1",
        "items": [{"sample_id": "S1", "amount": 41}],
    })
    assert r.status_code == 409
    assert r.json()["conflicts"][0]["code"] == "reservation_no_conflict"

    # 单内重复样品行:请求体校验失败
    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-DUP",
        "items": [
            {"sample_id": "S1", "amount": 1},
            {"sample_id": "S1", "amount": 2},
        ],
    })
    assert r.status_code == 422


def test_replay_after_release_reflects_released_status(client):
    _sample(client, "OP-1", "S1", 100)
    body = {"reservation_no": "R-1",
            "items": [{"sample_id": "S1", "amount": 40}]}
    client.post("/api/v1/reservations", json=body)
    client.post("/api/v1/reservations/R-1/release")

    # 释放后用原预留号重放:仍是首次预留内容,但状态已为 released
    r = client.post("/api/v1/reservations", json=body)
    assert r.status_code == 200
    assert r.json()["replayed"] is True
    assert r.json()["status"] == "released"
    assert r.json()["released_at"] is not None


# ---------------------------------------------------------------- 释放

def test_release_whole_reservation_restores_availability(client):
    _sample(client, "OP-1", "S1", 100)
    _sample(client, "OP-2", "S2", 100)

    client.post("/api/v1/reservations", json={
        "reservation_no": "R-1",
        "items": [
            {"sample_id": "S1", "amount": 60},
            {"sample_id": "S2", "amount": 30},
        ],
    })
    # 再来一张别的预留占住 S1 20,释放 R-1 后应只归还 60
    client.post("/api/v1/reservations", json={
        "reservation_no": "R-2",
        "items": [{"sample_id": "S1", "amount": 20}],
    })

    r = client.post("/api/v1/reservations/R-1/release")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "released"
    assert body["replayed"] is False
    by_id = {i["sample_id"]: i for i in body["released_items"]}
    assert by_id["S1"]["reserved"] == 20.0
    assert by_id["S1"]["available"] == 80.0
    assert by_id["S2"]["reserved"] == 0.0
    assert by_id["S2"]["available"] == 100.0
    assert body["released_at"] is not None

    # 再次释放:幂等重放,不报错也不改变数据
    r2 = client.post("/api/v1/reservations/R-1/release")
    assert r2.status_code == 200
    assert r2.json()["replayed"] is True
    assert client.get("/api/v1/samples/S1/availability").json()["reserved"] == 20.0

    # 查询已释放预留单
    r = client.get("/api/v1/reservations/R-1")
    assert r.json()["status"] == "released"
    assert r.json()["released_at"] is not None
    assert {i["sample_id"] for i in r.json()["items"]} == {"S1", "S2"}

    # 释放不存在的预留单
    r = client.post("/api/v1/reservations/NOPE/release")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_released_capacity_can_be_reserved_again(client):
    _sample(client, "OP-1", "S1", 10)
    client.post("/api/v1/reservations", json={
        "reservation_no": "R-1",
        "items": [{"sample_id": "S1", "amount": 10}],
    })
    # 全量锁定时再预留失败
    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-2",
        "items": [{"sample_id": "S1", "amount": 1}],
    })
    assert r.status_code == 409

    # 释放后容量恢复,新预留成功
    client.post("/api/v1/reservations/R-1/release")
    r = client.post("/api/v1/reservations", json={
        "reservation_no": "R-2",
        "items": [{"sample_id": "S1", "amount": 10}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["items"][0]["available"] == 0.0
