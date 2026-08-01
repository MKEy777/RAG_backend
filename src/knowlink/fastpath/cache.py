from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..indexing.embedder import embed_texts
from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.fastpath import FastPathEntry, SemanticCacheEntry
from ..settings import settings
from ..storage.factory import get_storage

log = get_logger("fastpath")

CACHE_NS_PREFIX = "sc_"


def kb_key(kb_ids: list[str]) -> str:
    """多 KB 组合的稳定 key（排序后 sha256 前 16 位）。"""
    return hashlib.sha256("|".join(sorted(kb_ids)).encode()).hexdigest()[:16]


def _ns(key: str) -> str:
    return f"{CACHE_NS_PREFIX}{key}"


def _as_utc(dt: datetime | None) -> datetime | None:
    """SQLite 读回的 datetime 无 tzinfo，统一补 UTC 以便比较。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _delete_entry(session: Session, entry: SemanticCacheEntry) -> None:
    session.delete(entry)
    session.commit()
    try:
        await get_storage().vector.delete_chunks(_ns(entry.kb_key), [entry.id])
    except Exception as e:
        log.warning("语义缓存向量删除失败", error=str(e))


async def _evict_if_needed(session: Session, key: str) -> None:
    """单 kb_key 超限时按 last_hit_at 淘汰最旧的（LRU）。"""
    limit = settings.cache_max_entries_per_kb
    count = session.execute(
        select(func.count()).select_from(SemanticCacheEntry).where(SemanticCacheEntry.kb_key == key)
    ).scalar_one()
    if count <= limit:
        return
    rows = (
        session.execute(
            select(SemanticCacheEntry)
            .where(SemanticCacheEntry.kb_key == key)
            .order_by(SemanticCacheEntry.last_hit_at.asc().nullsfirst(), SemanticCacheEntry.created_at.asc())
        )
        .scalars()
        .all()
    )
    doomed = rows[: count - limit]
    for entry in doomed:
        session.delete(entry)
    session.commit()
    try:
        ids = [e.id for e in doomed]
        if ids:
            await get_storage().vector.delete_chunks(_ns(key), ids)
    except Exception as e:
        log.warning("语义缓存淘汰向量删除失败", error=str(e))


async def write_cache(
    session: Session,
    kb_ids: list[str],
    query: str,
    answer: str,
    references: list[dict] | None,
    source_chunk_ids: list[str],
) -> None:
    """L2 成功回答后写穿语义缓存。仅缓存有引用且有源 chunk 的答案。"""
    if not settings.cache_enabled:
        return
    if not answer or not source_chunk_ids:
        return

    hashes: dict[str, str] = {}
    for cid in source_chunk_ids:
        chunk = session.get(Chunk, cid)
        if chunk is None or chunk.deleted_at is not None:
            return  # 任一源 chunk 已失效则不缓存
        hashes[cid] = chunk.content_hash or ""

    key = kb_key(kb_ids)
    entry = SemanticCacheEntry(
        kb_key=key,
        query_text=query,
        answer_text=answer,
        references=references or [],
        source_chunk_ids=list(hashes.keys()),
        source_chunk_hashes=[hashes[c] for c in hashes],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=settings.cache_ttl_s),
    )
    session.add(entry)
    session.commit()
    log.info("语义缓存写入", kb_key=key, query=query[:60], sources=len(hashes))

    try:
        vectors = await embed_texts([query])
        await get_storage().vector.upsert(
            _ns(key),
            [entry.id],
            np.asarray(vectors, dtype=np.float32),
            [{"entry_id": entry.id}],
        )
    except Exception as e:
        log.warning("语义缓存嵌入失败_条目保留无向量", error=str(e))

    await _evict_if_needed(session, key)


async def lookup_cache(
    session: Session,
    kb_ids: list[str],
    query: str,
    query_vector: np.ndarray | None = None,
) -> SemanticCacheEntry | None:
    """L1 语义缓存查询：相似度 ≥ 阈值 → 校验源 chunk 未变 → 命中。

    query_vector 可复用 L0 已嵌入的向量，避免二次嵌入。
    """
    if not settings.cache_enabled:
        return None

    key = kb_key(kb_ids)
    vec = query_vector
    if vec is None:
        try:
            vectors = await embed_texts([query])
            vec = vectors[0]
        except Exception as e:
            log.warning("语义缓存查询嵌入失败", error=str(e))
            return None

    try:
        results = await get_storage().vector.search(
            _ns(key), vec, top_k=3, threshold=settings.cache_similarity_threshold
        )
    except Exception as e:
        log.warning("语义缓存检索失败", error=str(e))
        return None

    now = datetime.now(timezone.utc)
    for r in results:
        entry_id = (r.get("metadata") or {}).get("entry_id")
        if not entry_id:
            continue
        entry = session.get(SemanticCacheEntry, entry_id)
        if entry is None:
            continue
        expires = _as_utc(entry.expires_at)
        if expires is not None and expires < now:
            await _delete_entry(session, entry)
            continue
        valid = True
        for cid, h in zip(entry.source_chunk_ids, entry.source_chunk_hashes):
            chunk = session.get(Chunk, cid)
            if chunk is None or chunk.deleted_at is not None or chunk.content_hash != h:
                valid = False
                break
        if not valid:
            log.info("语义缓存源已变更_失效", entry_id=entry.id)
            await _delete_entry(session, entry)
            continue
        entry.hit_count += 1
        entry.last_hit_at = now
        session.commit()
        return entry
    return None


def invalidate_cache_for_chunks(session: Session, chunk_ids: list[str], kb_ids: list[str] | None = None) -> None:
    """chunk 编辑/删除后失效引用它的缓存条目。"""
    if not chunk_ids:
        return
    id_set = set(chunk_ids)
    q = select(SemanticCacheEntry)
    if kb_ids:
        q = q.where(SemanticCacheEntry.kb_key == kb_key(kb_ids))
    for entry in session.execute(q).scalars().all():
        if id_set.intersection(entry.source_chunk_ids):
            session.delete(entry)
    session.commit()


def mark_entries_stale_for_chunks(session: Session, chunk_ids: list[str]) -> None:
    """快路径条目目标 chunk 被删除/编辑 → 置 stale，匹配时排除。"""
    if not chunk_ids:
        return
    rows = (
        session.execute(
            select(FastPathEntry).where(
                FastPathEntry.target_chunk_id.in_(chunk_ids),
                FastPathEntry.deleted_at.is_(None),
                FastPathEntry.status != "stale",
            )
        )
        .scalars()
        .all()
    )
    for entry in rows:
        entry.status = "stale"
    session.commit()


def mark_entries_stale_for_document(session: Session, doc_id: str) -> None:
    """文档删除/重解析 → 相关条目置 stale。"""
    rows = (
        session.execute(
            select(FastPathEntry).where(
                FastPathEntry.target_document_id == doc_id,
                FastPathEntry.deleted_at.is_(None),
                FastPathEntry.status != "stale",
            )
        )
        .scalars()
        .all()
    )
    for entry in rows:
        entry.status = "stale"
    session.commit()
