from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging import get_logger, get_trace_id, new_trace_id, setup_logging, trace_id_var
from ..models.database import init_db
from ..settings import settings
from .middleware import RateLimitMiddleware
from .routers import chat, documents, knowledge_bases, retrieval

setup_logging()
log = get_logger("api")
_uvicorn_log = logging.getLogger("uvicorn")

app = FastAPI(title="pyrag", version="0.1.0")

app.add_middleware(RateLimitMiddleware)


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tid = request.headers.get("X-Trace-Id") or new_trace_id()
        trace_id_var.set(tid)
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Trace-Id"] = tid
        print(f"[REQUEST] {request.method} {request.url.path} -> {response.status_code} ({duration_ms:.0f}ms)", flush=True)
        return response


app.add_middleware(TraceMiddleware)

app.include_router(knowledge_bases.router, prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"code": "VALIDATION_ERROR", "message": str(exc.errors()), "trace_id": get_trace_id()},
    )


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error("未处理的异常", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "Internal server error", "trace_id": get_trace_id()},
    )


@app.on_event("startup")
def on_startup():
    try:
        from alembic import command
        from alembic.config import Config
        from pathlib import Path

        alembic_ini = Path(__file__).parent.parent.parent.parent / "alembic.ini"
        if alembic_ini.exists():
            cfg = Config(str(alembic_ini))
            cfg.set_main_option("sqlalchemy.url", settings.database_url)
            command.upgrade(cfg, "head")
            log.info("启动完成_数据库迁移成功")
        else:
            init_db()
            log.info("启动完成_创建所有表")
    except Exception as e:
        log.warning("数据库迁移失败_回退创建所有表", error=str(e))
        init_db()
    log.info("服务启动完成")


@app.get("/health")
async def health():
    from sqlalchemy import text

    from ..models.database import engine

    status: dict[str, str] = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["db"] = "ok"
    except Exception:
        status["db"] = "down"

    try:
        import redis as redis_lib

        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        status["redis"] = "ok"
    except Exception:
        status["redis"] = "down"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=3) as c:
            resp = await c.get(f"{settings.llm_base_url}/models")
            status["llm"] = "ok" if resp.status_code < 500 else "degraded"
    except Exception:
        status["llm"] = "down"

    all_ok = all(v == "ok" for v in status.values())
    status_code = 200 if all_ok else 503
    from fastapi.responses import JSONResponse

    return JSONResponse(content={"status": "ok" if all_ok else "degraded", **status}, status_code=status_code)


@app.get("/metrics")
async def metrics():
    from ..metrics import metrics_content_type, metrics_response

    return Response(content=metrics_response(), media_type=metrics_content_type())
