from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..settings import settings

_connect_args = {}
_pool_kwargs: dict = {"pool_size": 10, "max_overflow": 20, "pool_recycle": 3600}

if settings.database_url.startswith("sqlite"):
    _pool_kwargs = {"poolclass": StaticPool}
    if ":memory:" in settings.database_url:
        _connect_args = {"check_same_thread": False}

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
        cursor.close()


def get_session() -> Session:
    return SessionLocal()


def init_db() -> None:
    from . import Base  # noqa: F401

    Base.metadata.create_all(bind=engine)
