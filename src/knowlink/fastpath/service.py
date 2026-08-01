from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..logging import get_logger
from ..metrics import FASTPATH_DURATION, FASTPATH_HITS
from ..models.chunk import Chunk
from ..models.fastpath import FastPathEntry, FastPathHit
from ..settings import settings
from .cache import lookup_cache
from .filler import fill_answer
from .matcher import lookup_fastpath

log = get_logger("fastpath")


@dataclass
class FastPathResult:
    answer: str
    references: list[dict]
    via: str  # fastpath | cache
    layer: str  # exact | semantic | cache
    entry_id: str | None = None
    similarity: float | None = None


def _entry_references(session: Session, entry: FastPathEntry) -> list[dict]:
    if not entry.target_chunk_id:
        return []
    chunk = session.get(Chunk, entry.target_chunk_id)
    if chunk is None or chunk.deleted_at is not None:
        return []
    meta = chunk.metadata_ or {}
    return [{
        "id": "c1",
        "chunk_id": chunk.id,
        "document_id": chunk.knowledge_id,
        "title_path": " > ".join(meta.get("title_path", [])),
        "score": 1.0,
        "channel": "fastpath",
        "chunk_type": chunk.chunk_type,
    }]


async def try_fastpath(
    session: Session, kb_ids: list[str], query: str
) -> FastPathResult | None:
    """L0 → L1 两级快路径查询，返回首个命中；均未命中返回 None（走 L2）。"""
    start = time.perf_counter()

    lookup = await lookup_fastpath(session, kb_ids, query)
    query_vector = getattr(lookup, "query_vector", None) if lookup else None
    if lookup:
        answer = await fill_answer(session, lookup.entry)
        if answer is not None:
            lookup.entry.hit_count += 1
            from datetime import datetime, timezone
            lookup.entry.last_hit_at = datetime.now(timezone.utc)
            session.commit()
            duration_ms = round((time.perf_counter() - start) * 1000)
            record_hit(
                session, kb_ids, query,
                layer=lookup.layer,
                entry_id=lookup.entry.id,
                similarity=lookup.similarity,
                duration_ms=duration_ms,
            )
            FASTPATH_HITS.labels(layer=lookup.layer).inc()
            FASTPATH_DURATION.labels(layer=lookup.layer).observe(duration_ms / 1000)
            return FastPathResult(
                answer=answer,
                references=_entry_references(session, lookup.entry),
                via="fastpath",
                layer=lookup.layer,
                entry_id=lookup.entry.id,
                similarity=lookup.similarity,
            )

    cache_entry = await lookup_cache(session, kb_ids, query, query_vector=query_vector)
    if cache_entry is not None:
        duration_ms = round((time.perf_counter() - start) * 1000)
        record_hit(
            session, kb_ids, query,
            layer="cache",
            entry_id=cache_entry.id,
            duration_ms=duration_ms,
        )
        FASTPATH_HITS.labels(layer="cache").inc()
        FASTPATH_DURATION.labels(layer="cache").observe(duration_ms / 1000)
        return FastPathResult(
            answer=cache_entry.answer_text,
            references=cache_entry.references or [],
            via="cache",
            layer="cache",
            entry_id=cache_entry.id,
        )
    return None


def record_hit(
    session: Session,
    kb_ids: list[str],
    query: str,
    layer: str,
    entry_id: str | None = None,
    similarity: float | None = None,
    duration_ms: int | None = None,
) -> None:
    """命中日志：看板统计 + 未来挖掘数据源。写入失败不阻断查询。"""
    try:
        session.add(FastPathHit(
            kb_ids=kb_ids,
            layer=layer,
            entry_id=entry_id,
            query=query,
            similarity=similarity,
            duration_ms=duration_ms,
        ))
        session.commit()
        _trim_hits(session)
    except Exception as e:
        session.rollback()
        log.warning("快路径命中日志写入失败", error=str(e))


def _trim_hits(session: Session) -> None:
    limit = settings.fastpath_stats_history
    try:
        total = session.execute(select(func.count()).select_from(FastPathHit)).scalar_one()
        if total > limit:
            overflow = session.execute(
                select(FastPathHit.id).order_by(FastPathHit.created_at.asc()).limit(total - limit)
            ).scalars().all()
            if overflow:
                session.execute(delete(FastPathHit).where(FastPathHit.id.in_(overflow)))
                session.commit()
    except Exception:
        session.rollback()


def write_cache_background(
    session_factory,
    kb_ids: list[str],
    query: str,
    answer: str,
    references: list[dict] | None,
    source_chunk_ids: list[str],
) -> None:
    """L2 成功回答后的后台写穿（独立 DB session，不阻塞 SSE 关闭）。"""
    from .cache import write_cache

    async def _do() -> None:
        db = session_factory()
        try:
            await write_cache(db, kb_ids, query, answer, references, source_chunk_ids)
        except Exception as e:
            log.warning("语义缓存写穿失败", error=str(e))
        finally:
            db.close()

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_do())
    except RuntimeError:
        # 无运行循环（同步调用路径）时直接同步执行一次
        async def _sync() -> None:
            db = session_factory()
            try:
                await write_cache(db, kb_ids, query, answer, references, source_chunk_ids)
            except Exception as e:
                log.warning("语义缓存写穿失败", error=str(e))
            finally:
                db.close()

        asyncio.run(_sync())
