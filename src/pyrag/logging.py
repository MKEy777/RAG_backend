from __future__ import annotations

import contextvars
import logging
import sys
import time
import uuid

import structlog

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return trace_id_var.get()


def _add_trace_id(logger, method_name, event_dict):
    tid = trace_id_var.get()
    if tid:
        event_dict["trace_id"] = tid
    return event_dict


def bind_kb(kb_id: str, doc_id: str = "", kb_name: str = "", filename: str = "") -> None:
    structlog.contextvars.bind_contextvars(kb_id=kb_id)
    if doc_id:
        structlog.contextvars.bind_contextvars(doc_id=doc_id)
    if kb_name:
        structlog.contextvars.bind_contextvars(kb_name=kb_name)
    if filename:
        structlog.contextvars.bind_contextvars(filename=filename)


def unbind_kb() -> None:
    structlog.contextvars.unbind_contextvars("kb_id", "doc_id", "kb_name", "filename")


def setup_logging(level: str = "INFO") -> None:
    numeric_level = logging.getLevelName(level.upper())

    logging.basicConfig(
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=numeric_level,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_trace_id,
            structlog.dev.ConsoleRenderer(
                pad_event=0,
                level_styles=structlog.dev.ConsoleRenderer.get_default_level_styles(),
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class Timer:
    def __init__(self, logger: structlog.BoundLogger, event: str, **kw):
        self._logger = logger
        self._event = event
        self._kw = kw
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start) * 1000
        if exc_type:
            self._logger.error(self._event, duration_ms=round(duration_ms, 2), error=str(exc_val), **self._kw)
        else:
            self._logger.info(self._event, duration_ms=round(duration_ms, 2), **self._kw)
        return False
