from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...fastpath.cache import (
    mark_entries_stale_for_chunks,
    mark_entries_stale_for_document,
)
from ...fastpath.matcher import ENTRY_NS_PREFIX
from ...fastpath.mining import mine_candidates
from ...logging import get_logger
from ...models.chunk import Chunk
from ...models.database import get_session
from ...models.fastpath import FastPathEntry, FastPathHit, SemanticCacheEntry
from ...storage.factory import get_storage

log = get_logger("fastpath_api")
router = APIRouter()


class EntryCreate(BaseModel):
    kb_id: str
    canonical_query: str = Field(min_length=1, max_length=512)
    example_queries: list[str] = []
    answer_template: str = ""
    variables: list[dict] = []
    target_chunk_id: str | None = None
    target_document_id: str | None = None


class EntryUpdate(BaseModel):
    canonical_query: str | None = None
    example_queries: list[str] | None = None
    answer_template: str | None = None
    variables: list[dict] | None = None
    target_chunk_id: str | None = None
    target_document_id: str | None = None
    status: str | None = Field(None, pattern="^(active|draft|disabled|stale)$")


class PromoteRequest(BaseModel):
    chunk_ids: list[str] = Field(min_length=1)


class StatsResponse(BaseModel):
    total_hits: int
    by_layer: dict[str, int]
    hit_rate: float
    avg_duration_ms: dict[str, float]
    total_entries: int
    active_entries: int
    cache_entries: int


def _entry_payload(e: FastPathEntry) -> dict:
    return {
        "id": e.id,
        "kb_id": e.kb_id,
        "canonical_query": e.canonical_query,
        "example_queries": e.example_queries or [],
        "answer_template": e.answer_template,
        "variables": e.variables or [],
        "target_chunk_id": e.target_chunk_id,
        "target_document_id": e.target_document_id,
        "status": e.status,
        "source": e.source,
        "source_chunk_id": e.source_chunk_id,
        "hit_count": e.hit_count,
        "last_hit_at": e.last_hit_at.isoformat() if e.last_hit_at else None,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


@router.get("/{kb_id}/entries")
def list_entries(
    kb_id: str,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
    session: Session = Depends(get_session),
):
    q = select(FastPathEntry).where(FastPathEntry.kb_id == kb_id, FastPathEntry.deleted_at.is_(None))
    if status:
        q = q.where(FastPathEntry.status == status)
    total = session.execute(select(func.count()).select_from(q.subquery())).scalar_one()
    rows = (
        session.execute(q.order_by(FastPathEntry.created_at.desc()).offset((page - 1) * page_size).limit(page_size))
        .scalars()
        .all()
    )
    return {"total": total, "items": [_entry_payload(e) for e in rows]}


@router.post("/entries")
def create_entry(body: EntryCreate, session: Session = Depends(get_session)):
    if body.target_chunk_id:
        chunk = session.get(Chunk, body.target_chunk_id)
        if chunk is None or chunk.deleted_at is not None:
            raise HTTPException(400, "target chunk 不存在或已删除")
    entry = FastPathEntry(
        kb_id=body.kb_id,
        canonical_query=body.canonical_query,
        example_queries=body.example_queries,
        answer_template=body.answer_template,
        variables=body.variables or None,
        target_chunk_id=body.target_chunk_id,
        target_document_id=body.target_document_id,
        status="active",
        source="manual",
    )
    session.add(entry)
    session.commit()
    return _entry_payload(entry)


@router.patch("/entries/{entry_id}")
async def update_entry(entry_id: str, body: EntryUpdate, session: Session = Depends(get_session)):
    entry = session.get(FastPathEntry, entry_id)
    if entry is None or entry.deleted_at is not None:
        raise HTTPException(404, "entry 不存在")
    query_fields_changed = False
    for key in ("canonical_query", "example_queries", "answer_template", "variables",
                "target_chunk_id", "target_document_id", "status"):
        value = getattr(body, key, None)
        if value is not None:
            if key in ("canonical_query", "example_queries"):
                query_fields_changed = True
            setattr(entry, key, value)
    session.commit()
    if query_fields_changed and entry.status == "active":
        await _activate_entry(session, entry)
    return _entry_payload(entry)


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str, session: Session = Depends(get_session)):
    entry = session.get(FastPathEntry, entry_id)
    if entry is None or entry.deleted_at is not None:
        raise HTTPException(404, "entry 不存在")
    entry.deleted_at = datetime.now(timezone.utc)
    session.commit()
    await _clear_entry_vectors(entry.kb_id, entry.id)
    return {"ok": True}


async def _activate_entry(session: Session, entry: FastPathEntry) -> None:
    """条目激活：清除旧向量后嵌入问法向量并落库（幂等）。"""
    from ...indexing.embedder import embed_texts

    entry.status = "active"
    session.commit()
    await _clear_entry_vectors(entry.kb_id, entry.id)
    texts = [entry.canonical_query, *list(entry.example_queries or [])]
    if not texts:
        return
    try:
        vectors = await embed_texts(texts)
        ids = [f"{entry.id}:{i}" for i in range(len(texts))]
        metadata = [{"entry_id": entry.id} for _ in texts]
        await get_storage().vector.upsert(f"{ENTRY_NS_PREFIX}{entry.kb_id}", ids, vectors, metadata)
    except Exception as e:
        log.warning("快路径条目嵌入失败_仍激活", entry_id=entry.id, error=str(e))


async def _clear_entry_vectors(kb_id: str, entry_id: str) -> None:
    """按 id 前缀清理某条目全部问法向量（向量 id 为 {entry_id}:{i}）。"""
    try:
        ns = f"{ENTRY_NS_PREFIX}{kb_id}"
        list_ids = getattr(get_storage().vector, "list_ids", None)
        if list_ids is None:
            return
        ids = [cid for cid in await list_ids(ns) if cid.startswith(f"{entry_id}:")]
        if ids:
            await get_storage().vector.delete_chunks(ns, ids)
    except Exception as e:
        log.warning("快路径旧向量清理失败", entry_id=entry_id, error=str(e))


@router.post("/entries/{entry_id}/activate")
async def activate_entry(entry_id: str, session: Session = Depends(get_session)):
    entry = session.get(FastPathEntry, entry_id)
    if entry is None or entry.deleted_at is not None:
        raise HTTPException(404, "entry 不存在")
    await _activate_entry(session, entry)
    return _entry_payload(entry)


@router.post("/entries/{entry_id}/disable")
def disable_entry(entry_id: str, session: Session = Depends(get_session)):
    entry = session.get(FastPathEntry, entry_id)
    if entry is None or entry.deleted_at is not None:
        raise HTTPException(404, "entry 不存在")
    entry.status = "disabled"
    session.commit()
    return _entry_payload(entry)


@router.get("/{kb_id}/faq-candidates")
def faq_candidates(kb_id: str, limit: int = 200, session: Session = Depends(get_session)):
    """未升级为快路径的 FAQ 问句列表（chunk_type=faq 且 metadata.generated）。"""
    promoted = set(
        session.execute(
            select(FastPathEntry.source_chunk_id).where(
                FastPathEntry.kb_id == kb_id,
                FastPathEntry.source == "faq",
                FastPathEntry.deleted_at.is_(None),
            )
        ).scalars()
    )
    rows = (
        session.execute(
            select(Chunk).where(
                Chunk.knowledge_base_id == kb_id,
                Chunk.chunk_type == "faq",
                Chunk.deleted_at.is_(None),
            ).limit(limit)
        )
        .scalars()
        .all()
    )
    items = []
    for c in rows:
        meta = c.metadata_ or {}
        if not meta.get("generated"):
            continue
        items.append({
            "chunk_id": c.id,
            "question": c.content,
            "source_chunk_id": meta.get("source_chunk_id"),
            "promoted": c.id in promoted,
        })
    return {"items": items}


@router.post("/{kb_id}/promote")
async def promote_faq(
    kb_id: str,
    body: PromoteRequest,
    session: Session = Depends(get_session),
):
    """FAQ 问句升级为快路径：定位源 chunk → LLM 生成模板 → 激活。"""
    from ...fastpath.filler import generate_template_from_chunk

    created = []
    for chunk_id in body.chunk_ids:
        chunk = session.get(Chunk, chunk_id)
        if chunk is None or chunk.knowledge_base_id != kb_id or chunk.chunk_type != "faq" or chunk.deleted_at is not None:
            continue
        meta = chunk.metadata_ or {}
        source_chunk_id = meta.get("source_chunk_id")
        source = session.get(Chunk, source_chunk_id) if source_chunk_id else None
        content = source.content if source and source.deleted_at is None else chunk.content
        template, variables = await generate_template_from_chunk(chunk.content, content)
        entry = FastPathEntry(
            kb_id=kb_id,
            canonical_query=chunk.content,
            example_queries=[],
            answer_template=template,
            variables=variables or None,
            target_chunk_id=source_chunk_id or chunk.id,
            target_document_id=chunk.knowledge_id,
            status="draft",
            source="faq",
            source_chunk_id=chunk.id,
        )
        session.add(entry)
        session.flush()
        await _activate_entry(session, entry)
        created.append(_entry_payload(entry))
    return {"created": created, "count": len(created)}


@router.post("/{kb_id}/mine")
async def mine(kb_id: str, session: Session = Depends(get_session)):
    """离线挖掘：历史问句聚类 → 定位 → 模板生成 → draft 条目。"""
    created = await mine_candidates(session, kb_id)
    return {"created": created}


@router.get("/{kb_id}/stats")
def stats(kb_id: str, session: Session = Depends(get_session)):
    q = select(FastPathHit).where(FastPathHit.kb_ids.contains(kb_id))
    hits = session.execute(q).scalars().all()
    total = len(hits)
    by_layer: dict[str, int] = {}
    durations: dict[str, list[int]] = {}
    for h in hits:
        by_layer[h.layer] = by_layer.get(h.layer, 0) + 1
        if h.duration_ms is not None:
            durations.setdefault(h.layer, []).append(h.duration_ms)
    hits_layers = sum(v for k, v in by_layer.items() if k in ("exact", "semantic", "cache"))
    hit_rate = round(hits_layers / total, 4) if total else 0.0
    avg_duration = {k: round(mean(v), 1) for k, v in durations.items()}
    entries = session.execute(
        select(func.count()).select_from(FastPathEntry).where(
            FastPathEntry.kb_id == kb_id, FastPathEntry.deleted_at.is_(None)
        )
    ).scalar_one()
    active = session.execute(
        select(func.count()).select_from(FastPathEntry).where(
            FastPathEntry.kb_id == kb_id,
            FastPathEntry.status == "active",
            FastPathEntry.deleted_at.is_(None),
        )
    ).scalar_one()
    cache_entries = session.execute(
        select(func.count()).select_from(SemanticCacheEntry)
    ).scalar_one()
    return StatsResponse(
        total_hits=total,
        by_layer=by_layer,
        hit_rate=hit_rate,
        avg_duration_ms=avg_duration,
        total_entries=entries,
        active_entries=active,
        cache_entries=cache_entries,
    )


@router.get("/{kb_id}/cache")
def list_cache(kb_id: str, session: Session = Depends(get_session)):
    rows = (
        session.execute(select(SemanticCacheEntry).order_by(SemanticCacheEntry.created_at.desc()).limit(100))
        .scalars()
        .all()
    )
    return {
        "items": [
            {
                "id": e.id,
                "kb_key": e.kb_key,
                "query_text": e.query_text,
                "answer_text": e.answer_text[:200],
                "source_chunk_ids": e.source_chunk_ids,
                "expires_at": e.expires_at.isoformat() if e.expires_at else None,
                "hit_count": e.hit_count,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in rows
        ]
    }


@router.delete("/cache/{entry_id}")
async def delete_cache_entry(entry_id: str, session: Session = Depends(get_session)):
    from ...fastpath.cache import _delete_entry

    entry = session.get(SemanticCacheEntry, entry_id)
    if entry is None:
        raise HTTPException(404, "缓存条目不存在")
    await _delete_entry(session, entry)
    return {"ok": True}


@router.post("/stale/document/{doc_id}")
def stale_document(doc_id: str, session: Session = Depends(get_session)):
    mark_entries_stale_for_document(session, doc_id)
    return {"ok": True}


@router.post("/stale/chunks")
def stale_chunks(body: PromoteRequest, session: Session = Depends(get_session)):
    mark_entries_stale_for_chunks(session, body.chunk_ids)
    return {"ok": True}
