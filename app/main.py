"""FastAPI 入口:科研样品分装追溯 API。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse

from . import schemas as s
from . import reservations, service
from .config import BALANCE_TOLERANCE, DB_PATH
from .db import connect, get_conn, init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 确保默认库文件在容器启动时即建好表
    conn = connect(DB_PATH)
    try:
        init_db(conn)
        yield
    finally:
        conn.close()


app = FastAPI(
    title="科研样品分装追溯 API",
    description=(
        "取样 / 分装 / 合并操作的事务化追溯服务:\n"
        "- 每个操作带全局唯一操作号,重复提交幂等返回原结果\n"
        "- 事务内校验父样品余量、禁止谱系环路\n"
        "- 支持批量提交与 dry-run 预校验,冲突返回操作号与批内定位\n"
        "- 按样品回溯完整来源/去向,并提供余量(账本)校验\n"
        "- 预留模块:一个预留号为多个样品在同一事务中锁定数量,"
        "支持幂等重放、整单释放与余量/已预留/可用量查询"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(service.ConflictError)
async def conflict_handler(_request, exc: service.ConflictError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "conflict",
            "conflicts": [c.model_dump() for c in exc.conflicts],
        },
    )


@app.get("/health", tags=["系统"])
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------- 操作提交

def _submit_one(
    conn: sqlite3.Connection, op: s.AnyOp
) -> s.OpResult:
    results = service.submit_batch(conn, [op], index_conflicts=False)
    return results[0]


@app.post(
    "/api/v1/operations",
    response_model=s.OpResult,
    tags=["操作"],
    summary="提交单个取样/分装/合并操作(按 type 区分)",
)
def submit_operation(
    op: s.AnyOp, conn: sqlite3.Connection = Depends(get_conn)
) -> s.OpResult:
    return _submit_one(conn, op)


@app.post(
    "/api/v1/operations/sampling",
    response_model=s.OpResult, tags=["操作"], summary="取样",
)
def submit_sampling(
    op: s.SamplingOp, conn: sqlite3.Connection = Depends(get_conn)
) -> s.OpResult:
    return _submit_one(conn, op)


@app.post(
    "/api/v1/operations/aliquot",
    response_model=s.OpResult, tags=["操作"], summary="分装",
)
def submit_aliquot(
    op: s.AliquotOp, conn: sqlite3.Connection = Depends(get_conn)
) -> s.OpResult:
    return _submit_one(conn, op)


@app.post(
    "/api/v1/operations/merge",
    response_model=s.OpResult, tags=["操作"], summary="合并",
)
def submit_merge(
    op: s.MergeOp, conn: sqlite3.Connection = Depends(get_conn)
) -> s.OpResult:
    return _submit_one(conn, op)


@app.post(
    "/api/v1/batch",
    response_model=s.BatchResponse,
    tags=["操作"],
    summary="批量操作:整批在一个事务中提交,任一冲突全部回滚",
)
def submit_batch(
    body: s.BatchRequest, conn: sqlite3.Connection = Depends(get_conn)
) -> s.BatchResponse:
    results = service.submit_batch(
        conn, body.operations, batch_id=body.batch_id
    )
    return s.BatchResponse(
        batch_id=body.batch_id,
        status=service.batch_status(results),  # type: ignore[arg-type]
        results=results,
    )


# ---------------------------------------------------------------- 预校验

from pydantic import BaseModel  # noqa: E402


class DryRunResponse(BaseModel):
    valid: bool
    conflicts: list[s.Conflict]
    results: list[s.OpResult] = []
    would_commit_status: Optional[str] = None


@app.post(
    "/api/v1/validate",
    response_model=DryRunResponse,
    tags=["校验"],
    summary="预校验单个操作(dry-run,不写库)",
)
def validate_operation(
    op: s.AnyOp, conn: sqlite3.Connection = Depends(get_conn)
) -> DryRunResponse:
    results, conflicts = service.simulate_batch(
        conn, [op], index_conflicts=False
    )
    if conflicts:
        return DryRunResponse(valid=False, conflicts=conflicts)
    return DryRunResponse(
        valid=True,
        conflicts=[],
        results=results or [],
        would_commit_status=service.batch_status(results or []),
    )


@app.post(
    "/api/v1/batch/validate",
    response_model=DryRunResponse,
    tags=["校验"],
    summary="预校验批量操作(dry-run,不写库),返回所有可静态发现的冲突及批内下标",
)
def validate_batch(
    body: s.BatchRequest, conn: sqlite3.Connection = Depends(get_conn)
) -> DryRunResponse:
    results, conflicts = service.simulate_batch(
        conn, body.operations, batch_id=body.batch_id
    )
    if conflicts:
        return DryRunResponse(valid=False, conflicts=conflicts)
    return DryRunResponse(
        valid=True,
        conflicts=[],
        results=results or [],
        would_commit_status=service.batch_status(results or []),
    )


# ---------------------------------------------------------------- 查询

@app.get(
    "/api/v1/samples", tags=["查询"], summary="样品列表"
)
def list_samples(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    rows = conn.execute(
        "SELECT * FROM samples ORDER BY sample_id LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) AS c FROM samples").fetchone()["c"]
    return {
        "total": total,
        "items": [
            {
                "sample_id": r["sample_id"],
                "material": r["material"],
                "unit": r["unit"],
                "initial_amount": r["initial_amount"],
                "remaining": r["remaining"],
                "location": r["location"],
                "is_root": bool(r["is_root"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    }


@app.get(
    "/api/v1/samples/{sample_id}/trace",
    response_model=s.TraceResponse,
    tags=["查询"],
    summary="回溯完整来源(upstream)/去向(downstream)/双向(both)",
)
def trace_sample(
    sample_id: str,
    direction: str = Query("both", pattern="^(upstream|downstream|both)$"),
    conn: sqlite3.Connection = Depends(get_conn),
) -> s.TraceResponse:
    return service.trace_sample(conn, sample_id, direction)


@app.get(
    "/api/v1/samples/{sample_id}/balance",
    response_model=s.CheckResponse,
    tags=["校验"],
    summary="单个样品余量校验:冗余余量 vs 出入库流水重算",
)
def check_sample_balance(
    sample_id: str,
    tolerance: float = Query(BALANCE_TOLERANCE, gt=0),
    conn: sqlite3.Connection = Depends(get_conn),
) -> s.CheckResponse:
    b = service.get_balance(conn, sample_id, tolerance)
    conflicts = []
    if not b.consistent:
        conflicts.append(s.Conflict(
            code="balance_inconsistent",
            message=(
                f"样品 {b.sample_id} 冗余余量 {b.stored_remaining} 与账本重算 "
                f"{b.ledger_remaining} 不一致(容差 {tolerance})"
            ),
            sample_id=b.sample_id,
            available=b.stored_remaining,
            requested=b.ledger_remaining,
        ))
    return s.CheckResponse(
        sample_id=sample_id,
        consistent=b.consistent,
        tolerance=tolerance,
        samples=[b],
        conflicts=conflicts,
    )


@app.get(
    "/api/v1/checks/balances",
    response_model=s.CheckResponse,
    tags=["校验"],
    summary="全量余量一致性校验,返回所有冲突样品及定位",
)
def check_all(
    tolerance: float = Query(BALANCE_TOLERANCE, gt=0),
    conn: sqlite3.Connection = Depends(get_conn),
) -> s.CheckResponse:
    balances, conflicts = service.check_all_balances(conn, tolerance)
    return s.CheckResponse(
        sample_id=None,
        consistent=not conflicts,
        tolerance=tolerance,
        samples=balances,
        conflicts=conflicts,
    )


@app.get(
    "/api/v1/operations/{op_no}",
    tags=["查询"],
    summary="按操作号查询原始操作结果(幂等回放内容)",
)
def get_operation(
    op_no: str, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    row = conn.execute(
        "SELECT * FROM operations WHERE op_no = ?", (op_no,)
    ).fetchone()
    if row is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "message": f"操作号 {op_no} 不存在"},
        )
    return {
        "op_no": row["op_no"],
        "type": row["op_type"],
        "payload": json.loads(row["payload"]),
        "effect": json.loads(row["effect"]),
        "batch_id": row["batch_id"],
        "seq_index": row["seq_index"],
        "location": row["location"],
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------- 预留

@app.post(
    "/api/v1/reservations",
    response_model=s.ReserveResult,
    tags=["预留"],
    summary="用一个预留号为多个样品分别锁定数量,全部足够才一次创建",
)
def create_reservation(
    body: s.ReservationRequest,
    conn: sqlite3.Connection = Depends(get_conn),
) -> s.ReserveResult:
    return reservations.reserve(conn, body)


@app.get(
    "/api/v1/reservations/{reservation_no}",
    response_model=s.ReserveResult,
    tags=["预留"],
    summary="按预留号查询预留单当前状态与明细",
)
def get_reservation(
    reservation_no: str, conn: sqlite3.Connection = Depends(get_conn)
):
    result = reservations.get_reservation(conn, reservation_no)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": f"预留号 {reservation_no} 不存在",
            },
        )
    return result


@app.post(
    "/api/v1/reservations/{reservation_no}/release",
    response_model=s.ReleaseResult,
    tags=["预留"],
    summary="主动释放整单预留(已释放则幂等重放)",
)
def release_reservation(
    reservation_no: str, conn: sqlite3.Connection = Depends(get_conn)
):
    result = reservations.release(conn, reservation_no)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": f"预留号 {reservation_no} 不存在",
            },
        )
    return result


@app.get(
    "/api/v1/samples/{sample_id}/availability",
    response_model=s.SampleAvailability,
    tags=["预留"],
    summary="查询样品当前余量 / 已预留量 / 可用量",
)
def sample_availability(
    sample_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> s.SampleAvailability:
    return reservations.get_availability(conn, sample_id)
