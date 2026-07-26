from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..settings import settings

_connect_args = {}
_pool_kwargs: dict = {"pool_size": 10, "max_overflow": 20, "pool_recycle": 3600}

if settings.database_url.startswith("sqlite"):
    # 文件库禁用 StaticPool：单连接跨线程共享会让 API 请求的 ROLLBACK 吞掉后台线程未提交的写入
    _connect_args = {"check_same_thread": False}
    if ":memory:" in settings.database_url:
        _pool_kwargs = {"poolclass": StaticPool}
    else:
        _pool_kwargs = {"pool_size": 20, "max_overflow": 30, "pool_recycle": 3600}

engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args=_connect_args,
    **_pool_kwargs,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


def get_session() -> Session:
    return SessionLocal()


def init_db() -> None:
    from . import Base  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_columns()


def _migrate_sqlite_columns() -> None:
    """SQLite create_all 不会给已有表加新列，手动补齐。"""
    if not settings.database_url.startswith("sqlite"):
        return
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if not insp.has_table("knowledge"):
        return
    existing = {c["name"] for c in insp.get_columns("knowledge")}
    new_cols = {
        "graph_status": "VARCHAR(20)",
        "wiki_status": "VARCHAR(20)",
        "summary_status": "VARCHAR(20)",
    }
    with engine.begin() as conn:
        for col, col_type in new_cols.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE knowledge ADD COLUMN {col} {col_type}"))
