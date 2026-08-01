from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..indexing.embedder import embed_texts
from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.fastpath import FastPathEntry
from ..models.knowledge import Knowledge
from ..settings import settings
from ..storage.factory import get_storage

log = get_logger("fastpath")

# \w 在 Unicode 模式下包含 CJK，因此该正则只剥离空白与标点，保留中文词面
_NORMALIZE_RE = re.compile(r"[\s\u3000\uFEFF\W_]+", re.UNICODE)

# Windows 目录名不允许 ':'，故命名空间用前缀 + 下划线
ENTRY_NS_PREFIX = "fp_"


def normalize_query(q: str) -> str:
    """NFKC 归一化 + 去空白/标点 + 小写，用于精确问法匹配。"""
    q = unicodedata.normalize("NFKC", q or "").lower()
    return _NORMALIZE_RE.sub("", q)


@dataclass
class FastPathLookup:
    entry: FastPathEntry
    layer: str  # exact | semantic
    similarity: float
    query_vector: np.ndarray | None = None


def _load_active_entries(session: Session, kb_ids: list[str]) -> list[FastPathEntry]:
    if not kb_ids:
        return []
    rows = (
        session.execute(
            select(FastPathEntry).where(
                FastPathEntry.kb_id.in_(kb_ids),
                FastPathEntry.status == "active",
                FastPathEntry.deleted_at.is_(None),
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


def _entry_target_alive(session: Session, entry: FastPathEntry) -> bool:
    """懒校验：目标 chunk/文档仍存活，否则顺手置 stale。"""
    if entry.target_chunk_id:
        chunk = session.get(Chunk, entry.target_chunk_id)
        if chunk is None or chunk.deleted_at is not None or not chunk.enabled:
            entry.status = "stale"
            session.commit()
            return False
    if entry.target_document_id:
        knowledge = session.get(Knowledge, entry.target_document_id)
        if knowledge is not None and knowledge.deleted_at is not None:
            entry.status = "stale"
            session.commit()
            return False
    return True


async def _semantic_match(
    session: Session, kb_ids: list[str], query: str
) -> FastPathLookup | None:
    try:
        vectors = await embed_texts([query])
        query_vector = vectors[0]
    except Exception as e:
        log.warning("快路径语义匹配嵌入失败", error=str(e))
        return None

    storage = get_storage()
    best: tuple[float, str] | None = None
    for kb_id in kb_ids:
        try:
            results = await storage.vector.search(
                f"{ENTRY_NS_PREFIX}{kb_id}",
                query_vector,
                top_k=3,
                threshold=settings.fastpath_match_threshold,
            )
        except Exception as e:
            log.warning("快路径向量检索失败", kb_id=kb_id, error=str(e))
            continue
        for r in results:
            entry_id = (r.get("metadata") or {}).get("entry_id")
            if entry_id and (best is None or r["score"] > best[0]):
                best = (r["score"], entry_id)

    if best is None:
        return None
    entry = session.get(FastPathEntry, best[1])
    if entry is None or entry.status != "active" or entry.deleted_at is not None:
        return None
    if not _entry_target_alive(session, entry):
        return None
    return FastPathLookup(entry=entry, layer="semantic", similarity=best[0], query_vector=query_vector)


async def lookup_fastpath(
    session: Session, kb_ids: list[str], query: str
) -> FastPathLookup | None:
    """L0 快路径匹配：先归一化精确匹配（零嵌入开销），再向量语义匹配。"""
    if not settings.fastpath_enabled:
        return None

    entries = _load_active_entries(session, kb_ids)
    if not entries:
        return None

    nq = normalize_query(query)
    if nq:
        for entry in entries:
            candidates = [entry.canonical_query, *list(entry.example_queries or [])]
            for q in candidates:
                if q and nq == normalize_query(q):
                    if not _entry_target_alive(session, entry):
                        break
                    return FastPathLookup(entry=entry, layer="exact", similarity=1.0)

    return await _semantic_match(session, kb_ids, query)
