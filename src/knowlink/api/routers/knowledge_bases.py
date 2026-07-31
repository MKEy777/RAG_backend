from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...models.database import get_session
from ...models.knowledge_base import KnowledgeBase

router = APIRouter()


class RetrievalConfigModel(BaseModel):
    search_mode: str | None = Field(None, pattern="^(hybrid|vector|fulltext)$")
    top_k: int | None = Field(None, ge=1, le=50)
    recall_top_k: int | None = Field(None, ge=1, le=200)
    vector_weight: float | None = Field(None, ge=0.0, le=1.0)
    keyword_weight: float | None = Field(None, ge=0.0, le=1.0)
    vector_threshold: float | None = Field(None, ge=0.0, le=1.0)
    keyword_threshold: float | None = Field(None, ge=0.0, le=1.0)
    rrf_k: int | None = Field(None, ge=1, le=1000)
    enable_rerank: bool | None = None
    rerank_threshold: float | None = Field(None, ge=0.0, le=1.0)
    enable_graph: bool | None = None
    enable_wiki_boost: bool | None = None
    wiki_boost: float | None = Field(None, ge=1.0, le=3.0)
    expand_parent: bool | None = None


class KBCreate(BaseModel):
    name: str
    type: str = "document"
    description: str | None = None
    group_name: str | None = None
    icon: str | None = None
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    chunking_config: dict = {}
    indexing_strategy: dict = {}
    retrieval_config: RetrievalConfigModel | None = None
    enable_wiki: bool = True
    enable_graph: bool = True


class KBResponse(BaseModel):
    id: str
    name: str
    type: str
    description: str | None
    group_name: str | None = None
    icon: str | None = None
    enable_wiki: bool = True
    enable_graph: bool = True
    retrieval_config: dict | None = None
    doc_count: int = 0
    total_size: int = 0
    chunk_count: int = 0

    class Config:
        from_attributes = True


class KBUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    group_name: str | None = None
    icon: str | None = None
    retrieval_config: RetrievalConfigModel | None = None
    enable_wiki: bool | None = None
    enable_graph: bool | None = None


def _kb_response(kb: KnowledgeBase, stats: dict | None = None) -> KBResponse:
    stats = stats or {}
    return KBResponse(
        id=kb.id,
        name=kb.name,
        type=kb.type,
        description=kb.description,
        group_name=kb.group_name,
        icon=kb.icon,
        enable_wiki=(kb.wiki_config or {}).get("enabled", True),
        enable_graph=(kb.graph_config or {}).get("enabled", True),
        retrieval_config=kb.retrieval_config,
        doc_count=stats.get("doc_count", 0),
        total_size=stats.get("total_size", 0),
        chunk_count=stats.get("chunk_count", 0),
    )


@router.get("/embedding-models")
def list_embedding_models():
    from ...settings import settings

    return {
        "models": [
            {"id": mid, "model": m.get("model"), "dims": m.get("dims")}
            for mid, m in settings.embedding_models.items()
        ]
    }


@router.post("", response_model=KBResponse)
def create_kb(body: KBCreate, session: Session = Depends(get_session)):
    from ...settings import settings

    model_info = settings.embedding_models.get(body.embedding_model_id)
    if model_info is None:
        raise HTTPException(422, f"未知向量模型 '{body.embedding_model_id}'，可选：{list(settings.embedding_models)}")
    indexing_strategy = {**body.indexing_strategy, "embedding_dims": model_info["dims"]}

    kb = KnowledgeBase(
        id=uuid.uuid4().hex,
        tenant_id=1,
        name=body.name,
        type=body.type,
        description=body.description,
        group_name=body.group_name,
        icon=body.icon,
        embedding_model_id=body.embedding_model_id,
        chunking_config=body.chunking_config,
        indexing_strategy=indexing_strategy,
        retrieval_config=body.retrieval_config.model_dump(exclude_none=True) if body.retrieval_config else None,
        wiki_config={"enabled": body.enable_wiki},
        graph_config={"enabled": body.enable_graph},
    )
    session.add(kb)
    session.commit()
    session.refresh(kb)
    return _kb_response(kb)


def _kb_stats(session: Session, kb_ids: list[str]) -> dict[str, dict]:
    from sqlalchemy import func

    from ...models.chunk import Chunk
    from ...models.knowledge import Knowledge

    if not kb_ids:
        return {}
    doc_rows = (
        session.query(
            Knowledge.knowledge_base_id,
            func.count(Knowledge.id),
            func.coalesce(func.sum(Knowledge.file_size), 0),
        )
        .filter(Knowledge.knowledge_base_id.in_(kb_ids), Knowledge.deleted_at.is_(None))
        .group_by(Knowledge.knowledge_base_id)
        .all()
    )
    chunk_rows = (
        session.query(Chunk.knowledge_base_id, func.count(Chunk.id))
        .filter(Chunk.knowledge_base_id.in_(kb_ids), Chunk.deleted_at.is_(None))
        .group_by(Chunk.knowledge_base_id)
        .all()
    )
    stats = {kb_id: {"doc_count": c, "total_size": int(s or 0)} for kb_id, c, s in doc_rows}
    for kb_id, c in chunk_rows:
        stats.setdefault(kb_id, {})["chunk_count"] = c
    return stats


@router.get("/groups")
def list_groups(session: Session = Depends(get_session)):
    rows = (
        session.query(KnowledgeBase.group_name)
        .filter(KnowledgeBase.group_name.is_not(None))
        .distinct()
        .all()
    )
    return {"groups": sorted(r[0] for r in rows if r[0])}


@router.get("")
def list_kbs(
    page: int = 1,
    size: int = 20,
    keyword: str | None = None,
    group: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    session: Session = Depends(get_session),
):
    query = session.query(KnowledgeBase)
    if keyword:
        pattern = f"%{keyword}%"
        query = query.filter(
            KnowledgeBase.name.ilike(pattern) | KnowledgeBase.description.ilike(pattern)
        )
    if group:
        query = query.filter(KnowledgeBase.group_name == group)

    sort_col = {
        "created_at": KnowledgeBase.created_at,
        "updated_at": KnowledgeBase.updated_at,
        "name": KnowledgeBase.name,
    }.get(sort_by, KnowledgeBase.created_at)
    query = query.order_by(sort_col.asc() if order == "asc" else sort_col.desc())

    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    stats = _kb_stats(session, [kb.id for kb in items])
    return {
        "items": [_kb_response(kb, stats.get(kb.id)) for kb in items],
        "total": total,
        "page": page,
        "size": size,
    }


class IconSuggestResponse(BaseModel):
    icon: str


@router.post("/{kb_id}/icon/suggest", response_model=IconSuggestResponse)
async def suggest_icon(kb_id: str, session: Session = Depends(get_session)):
    from ...llm.client import get_llm_client

    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    prompt = (
        f"知识库名称：{kb.name}\n描述：{kb.description or '无'}\n"
        "请为该知识库挑选一个最贴切的 emoji 作为图标。只输出这一个 emoji，不要任何其他文字。"
    )
    try:
        reply = await get_llm_client().chat(
            [{"role": "user", "content": prompt}], max_tokens=16, purpose="icon_suggest"
        )
    except Exception:
        raise HTTPException(502, "图标建议服务暂不可用")
    icon = reply.strip().split()[0][:8] if reply.strip() else "📚"
    kb.icon = icon
    session.commit()
    return IconSuggestResponse(icon=icon)


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(kb_id: str, session: Session = Depends(get_session)):
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    stats = _kb_stats(session, [kb.id])
    return _kb_response(kb, stats.get(kb.id))


@router.patch("/{kb_id}", response_model=KBResponse)
def update_kb(kb_id: str, body: KBUpdate, session: Session = Depends(get_session)):
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "名称不能为空")
        kb.name = name
    if body.description is not None:
        kb.description = body.description
    if body.group_name is not None:
        kb.group_name = body.group_name or None
    if body.icon is not None:
        kb.icon = body.icon or None
    if body.retrieval_config is not None:
        kb.retrieval_config = body.retrieval_config.model_dump(exclude_none=True) or None
    if body.enable_wiki is not None:
        kb.wiki_config = {**(kb.wiki_config or {}), "enabled": body.enable_wiki}
    if body.enable_graph is not None:
        kb.graph_config = {**(kb.graph_config or {}), "enabled": body.enable_graph}
    session.commit()
    session.refresh(kb)
    return _kb_response(kb)


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, session: Session = Depends(get_session)):
    from ...models.chunk import Chunk
    from ...models.knowledge import Knowledge
    from datetime import datetime, timezone

    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    now = datetime.now(timezone.utc)
    # 软删除知识库下的所有文档及切块
    session.query(Knowledge).filter(
        Knowledge.knowledge_base_id == kb_id,
        Knowledge.deleted_at.is_(None)
    ).update({"deleted_at": now})
    session.query(Chunk).filter(
        Chunk.knowledge_base_id == kb_id,
        Chunk.deleted_at.is_(None)
    ).update({"deleted_at": now})

    # 删除知识库
    session.delete(kb)
    session.commit()
    return {"status": "deleted"}
