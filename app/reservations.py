"""预留模块:同一事务内为多个样品锁定数量,整单成功或整单失败。

规则:
- 一个预留号(reservation_no)对应一张预留单,是幂等键;
- 创建时在单个 ``BEGIN IMMEDIATE`` 事务中检查每个样品的
  可用量(当前余量 - 未释放预留),全部足够才一次写入,否则整单回滚;
- 预留号重放:同号同请求体返回首次结果(replayed=True),同号不同内容 409;
- 释放为整单释放:明细删除、预留单置 released,释放本身幂等;
- 预留单、明细、预留/释放事件均持久化,建表脚本随容器启动初始化。
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from . import schemas as s
from .service import ConflictError, _conflict, _r

_EPS = 1e-9  # 运算时浮点比较


# ---------------------------------------------------------------- 基础查询

def _get_reservation(
    conn: sqlite3.Connection, reservation_no: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM reservations WHERE reservation_no = ?",
        (reservation_no,),
    ).fetchone()


def _get_items(
    conn: sqlite3.Connection, reservation_no: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM reservation_items WHERE reservation_no = ?"
        " ORDER BY rowid",
        (reservation_no,),
    ).fetchall()


def _active_reserved(conn: sqlite3.Connection, sample_id: str) -> float:
    """该样品当前所有未释放(active)预留单锁定的总量。"""
    row = conn.execute(
        """SELECT COALESCE(SUM(i.amount), 0) AS reserved
             FROM reservation_items i
             JOIN reservations r ON r.reservation_no = i.reservation_no
            WHERE i.sample_id = ? AND r.status = 'active'""",
        (sample_id,),
    ).fetchone()
    return _r(row["reserved"])


def _availability_view(
    conn: sqlite3.Connection, sample_id: str
) -> tuple[float, str, float, float]:
    """返回 (当前余量, 单位, 已预留, 可用量)。调用方需保证样品存在。"""
    smp = conn.execute(
        "SELECT remaining, unit FROM samples WHERE sample_id = ?",
        (sample_id,),
    ).fetchone()
    remaining = _r(smp["remaining"])
    reserved = _active_reserved(conn, sample_id)
    return remaining, smp["unit"], reserved, _r(remaining - reserved)


# ---------------------------------------------------------------- 事件存取

def _latest_event(
    conn: sqlite3.Connection, reservation_no: str, event_type: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM reservation_events WHERE reservation_no = ?"
        " AND event_type = ? ORDER BY event_id DESC LIMIT 1",
        (reservation_no, event_type),
    ).fetchone()


def _reserve_result_from_effect(
    row: sqlite3.Row, effect: dict, *, replayed: bool
) -> s.ReserveResult:
    return s.ReserveResult(
        reservation_no=row["reservation_no"],
        status=row["status"],
        items=[s.ReservationItemOut(**i) for i in effect["items"]],
        replayed=replayed,
        created_at=row["created_at"],
        released_at=row["released_at"],
    )


# ---------------------------------------------------------------- 预留创建

def reserve(
    conn: sqlite3.Connection, req: s.ReservationRequest
) -> s.ReserveResult:
    """整单预留:任一样品不存在或可用量不足,整单失败且不改变任何预留。"""
    payload = req.model_dump_json()

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = _get_reservation(conn, req.reservation_no)
        if existing is not None:
            if existing["payload"] != payload:
                raise ConflictError([
                    _conflict(
                        "reservation_no_conflict",
                        f"预留号 {req.reservation_no} 已存在但请求内容不同,"
                        "同一预留号必须对应同一次预留(幂等键冲突)",
                    )
                ])
            # 幂等重放:返回首次预留结果,但状态取当前(可能已被释放)
            event = _latest_event(conn, req.reservation_no, "reserve")
            result = _reserve_result_from_effect(
                existing, json.loads(event["effect"]), replayed=True
            )
            conn.execute("COMMIT")
            return result

        # 1) 事务内逐个检查样品存在性与可用余量,收集全部冲突
        conflicts: list[s.Conflict] = []
        checked: list[tuple[str, float, str, float, float]] = []
        # (sample_id, amount, unit, remaining, available_before)
        for item in req.items:
            smp = conn.execute(
                "SELECT * FROM samples WHERE sample_id = ?",
                (item.sample_id,),
            ).fetchone()
            if smp is None:
                conflicts.append(_conflict(
                    "sample_not_found",
                    f"样品 {item.sample_id} 不存在,无法预留",
                    sample_id=item.sample_id,
                ))
                continue
            amount = _r(item.amount)
            remaining = _r(smp["remaining"])
            reserved = _active_reserved(conn, item.sample_id)
            available = _r(remaining - reserved)
            if amount > available + _EPS:
                conflicts.append(_conflict(
                    "insufficient_available",
                    f"样品 {item.sample_id} 可用量 {available} {smp['unit']} 不足,"
                    f"本次预留需要 {amount} {smp['unit']}"
                    f"(当前余量 {remaining},已预留 {reserved})",
                    sample_id=item.sample_id,
                    requested=amount, available=available,
                ))
                continue
            checked.append((item.sample_id, amount, smp["unit"],
                            remaining, available))

        if conflicts:
            raise ConflictError(conflicts)

        # 2) 全部足够:一次创建预留单与明细
        conn.execute(
            "INSERT INTO reservations (reservation_no, status, payload)"
            " VALUES (?, 'active', ?)",
            (req.reservation_no, payload),
        )
        item_out: list[s.ReservationItemOut] = []
        for sample_id, amount, unit, _remaining, _avail_before in checked:
            conn.execute(
                "INSERT INTO reservation_items (reservation_no, sample_id,"
                " amount, unit) VALUES (?, ?, ?, ?)",
                (req.reservation_no, sample_id, amount, unit),
            )
        # 落库后重算每个样品的余量/已预留/可用视图
        for sample_id, amount, unit, _r0, _a0 in checked:
            remaining, _u, reserved, available = _availability_view(
                conn, sample_id
            )
            item_out.append(s.ReservationItemOut(
                sample_id=sample_id, amount=amount, unit=unit,
                remaining=remaining, reserved=reserved, available=available,
            ))

        row = _get_reservation(conn, req.reservation_no)
        effect = {
            "reservation_no": req.reservation_no,
            "status": "active",
            "items": [i.model_dump() for i in item_out],
        }
        conn.execute(
            "INSERT INTO reservation_events (reservation_no, event_type,"
            " payload, effect) VALUES (?, 'reserve', ?, ?)",
            (req.reservation_no, payload,
             json.dumps(effect, ensure_ascii=False)),
        )
        result = _reserve_result_from_effect(row, effect, replayed=False)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return result


# ---------------------------------------------------------------- 整单释放

def release(
    conn: sqlite3.Connection, reservation_no: str
) -> Optional[s.ReleaseResult]:
    """主动释放整单预留。不存在返回 None(由路由转 404);已释放则幂等重放。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _get_reservation(conn, reservation_no)
        if row is None:
            conn.execute("ROLLBACK")
            return None

        if row["status"] == "released":
            event = _latest_event(conn, reservation_no, "release")
            effect = json.loads(event["effect"])
            result = s.ReleaseResult(
                reservation_no=reservation_no,
                status="released",
                released_items=[
                    s.ReservationItemOut(**i) for i in effect["released_items"]
                ],
                replayed=True,
                released_at=row["released_at"],
            )
            conn.execute("COMMIT")
            return result

        items = _get_items(conn, reservation_no)
        released_out: list[s.ReservationItemOut] = []
        for it in items:
            released_out.append(s.ReservationItemOut(
                sample_id=it["sample_id"],
                amount=_r(it["amount"]),
                unit=it["unit"],
            ))

        # 删除本单全部明细即解除锁定,再把预留单置为 released
        conn.execute(
            "DELETE FROM reservation_items WHERE reservation_no = ?",
            (reservation_no,),
        )
        conn.execute(
            "UPDATE reservations SET status = 'released',"
            " released_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
            " WHERE reservation_no = ?",
            (reservation_no,),
        )
        # 释放后重算涉及样品的余量/已预留/可用视图
        for out in released_out:
            remaining, _u, reserved, available = _availability_view(
                conn, out.sample_id
            )
            out.remaining, out.reserved, out.available = (
                remaining, reserved, available
            )

        row = _get_reservation(conn, reservation_no)
        effect = {
            "reservation_no": reservation_no,
            "status": "released",
            "released_items": [i.model_dump() for i in released_out],
        }
        conn.execute(
            "INSERT INTO reservation_events (reservation_no, event_type,"
            " payload, effect) VALUES (?, 'release', ?, ?)",
            (reservation_no,
             json.dumps({"reservation_no": reservation_no}),
             json.dumps(effect, ensure_ascii=False)),
        )
        result = s.ReleaseResult(
            reservation_no=reservation_no,
            status="released",
            released_items=released_out,
            replayed=False,
            released_at=row["released_at"],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return result


# ---------------------------------------------------------------- 查询

def get_reservation(
    conn: sqlite3.Connection, reservation_no: str
) -> Optional[s.ReserveResult]:
    """按预留号查询当前状态与明细(活跃预留读明细表,已释放读释放事件)。"""
    row = _get_reservation(conn, reservation_no)
    if row is None:
        return None
    if row["status"] == "active":
        items = [
            s.ReservationItemOut(
                sample_id=it["sample_id"],
                amount=_r(it["amount"]),
                unit=it["unit"],
            )
            for it in _get_items(conn, reservation_no)
        ]
        for out in items:
            remaining, _u, reserved, available = _availability_view(
                conn, out.sample_id
            )
            out.remaining, out.reserved, out.available = (
                remaining, reserved, available
            )
        return s.ReserveResult(
            reservation_no=reservation_no,
            status="active",
            items=items,
            replayed=False,
            created_at=row["created_at"],
            released_at=None,
        )
    event = _latest_event(conn, reservation_no, "release")
    effect = json.loads(event["effect"])
    return s.ReserveResult(
        reservation_no=reservation_no,
        status="released",
        items=[
            s.ReservationItemOut(**i) for i in effect["released_items"]
        ],
        replayed=False,
        created_at=row["created_at"],
        released_at=row["released_at"],
    )


def get_availability(
    conn: sqlite3.Connection, sample_id: str
) -> s.SampleAvailability:
    """样品当前余量 / 已预留量 / 可用量。"""
    smp = conn.execute(
        "SELECT unit FROM samples WHERE sample_id = ?", (sample_id,)
    ).fetchone()
    if smp is None:
        raise ConflictError([
            _conflict("sample_not_found", f"样品 {sample_id} 不存在",
                      sample_id=sample_id)
        ])
    remaining, unit, reserved, available = _availability_view(conn, sample_id)
    return s.SampleAvailability(
        sample_id=sample_id,
        unit=unit,
        remaining=remaining,
        reserved=reserved,
        available=available,
    )
