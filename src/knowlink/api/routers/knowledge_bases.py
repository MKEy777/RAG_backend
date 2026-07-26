from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models.database import get_session
from ...models.knowledge_base import KnowledgeBase

router = APIRouter()


class KBCreate(BaseModel):
    name: str
    type: str = "document"
    description: str | None = None
    embedding_model_id: str = "qwen3-embedding-4b"
    chunking_config: dict = {}
    indexing_strategy: dict = {}
    enable_wiki: bool = True
    enable_graph: bool = True


class KBResponse(BaseModel):
    id: str
    name: str
    type: str
    description: str | None
    enable_wiki: bool = True
    enable_graph: bool = True

    class Config:
        from_attributes = True


class KBUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    enable_wiki: bool | None = None
    enable_graph: bool | None = None


def _kb_response(kb: KnowledgeBase) -> KBResponse:
    return KBResponse(
        id=kb.id,
        name=kb.name,
        type=kb.type,
        description=kb.description,
        enable_wiki=(kb.wiki_config or {}).get("enabled", True),
        enable_graph=(kb.graph_config or {}).get("enabled", True),
    )


@router.post("", response_model=KBResponse)
def create_kb(body: KBCreate, session: Session = Depends(get_session)):
    kb = KnowledgeBase(
        id=uuid.uuid4().hex,
        tenant_id=1,
        name=body.name,
        type=body.type,
        description=body.description,
        embedding_model_id=body.embedding_model_id,
        chunking_config=body.chunking_config,
        indexing_strategy=body.indexing_strategy,
        wiki_config={"enabled": body.enable_wiki},
        graph_config={"enabled": body.enable_graph},
    )
    session.add(kb)
    session.commit()
    session.refresh(kb)
    return _kb_response(kb)


@router.get("")
def list_kbs(page: int = 1, size: int = 20, session: Session = Depends(get_session)):
    query = session.query(KnowledgeBase)
    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    return {
        "items": [_kb_response(kb) for kb in items],
        "total": total,
        "page": page,
        "size": size,
    }


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(kb_id: str, session: Session = Depends(get_session)):
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return _kb_response(kb)


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
