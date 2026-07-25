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


class KBResponse(BaseModel):
    id: str
    name: str
    type: str
    description: str | None

    class Config:
        from_attributes = True


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
    )
    session.add(kb)
    session.commit()
    session.refresh(kb)
    return kb


@router.get("")
def list_kbs(page: int = 1, size: int = 20, session: Session = Depends(get_session)):
    query = session.query(KnowledgeBase)
    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    return {
        "items": [KBResponse.model_validate(kb) for kb in items],
        "total": total,
        "page": page,
        "size": size,
    }


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(kb_id: str, session: Session = Depends(get_session)):
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, session: Session = Depends(get_session)):
    from ...models.knowledge import Knowledge
    from datetime import datetime, timezone

    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    # 软删除知识库下的所有文档
    session.query(Knowledge).filter(
        Knowledge.knowledge_base_id == kb_id,
        Knowledge.deleted_at.is_(None)
    ).update({"deleted_at": datetime.now(timezone.utc)})

    # 删除知识库
    session.delete(kb)
    session.commit()
    return {"status": "deleted"}
