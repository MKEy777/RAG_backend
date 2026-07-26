from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...indexing.embedder import content_hash, embed_texts
from ...logging import get_logger
from ...models.chunk import Chunk
from ...models.database import get_session
from ...models.knowledge_base import KnowledgeBase
from ...storage.factory import get_storage

log = get_logger("faq")
router = APIRouter()


class FaqItemBody(BaseModel):
    question: str
    similar_questions: list[str] = []
    answer: str
    negative_questions: list[str] = []
    tags: list[str] = []


@router.post("/{kb_id}/items")
async def import_faq_item(kb_id: str, body: FaqItemBody, session: Session = Depends(get_session)):
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    rows = _add_item_chunks(session, kb, kb_id, body)
    session.commit()
    vectors = await embed_texts([q for _, q, _, _ in rows])
    metadata = [{"knowledge_id": fid, "chunk_type": "faq", "tags": tags} for _, _, fid, tags in rows]
    await get_storage().vector.upsert(kb_id, [cid for cid, _, _, _ in rows], vectors, metadata)
    log.info("FAQ导入", kb_id=kb_id, faq_id=rows[0][2], questions=len(rows))
    return {"faq_id": rows[0][2], "chunk_count": len(rows)}


def _add_item_chunks(session: Session, kb: KnowledgeBase, kb_id: str, body: FaqItemBody) -> list[tuple[str, str, str, list[str]]]:
    faq_id = uuid.uuid4().hex
    rows: list[tuple[str, str, str, list[str]]] = []
    questions = [("standard", body.question)] + [("similar", q) for q in body.similar_questions if q.strip()]
    for role, q in questions:
        cid = uuid.uuid4().hex
        session.add(Chunk(
            id=cid, tenant_id=kb.tenant_id, knowledge_base_id=kb_id,
            knowledge_id=faq_id, content=q, chunk_type="faq",
            content_hash=content_hash(q),
            metadata_={
                "faq_id": faq_id, "role": role, "standard_question": body.question,
                "answer": body.answer, "tags": body.tags,
                "negative_questions": body.negative_questions,
            },
        ))
        rows.append((cid, q, faq_id, body.tags))
    return rows


@router.get("/{kb_id}/items")
def list_faq_items(kb_id: str, session: Session = Depends(get_session)):
    chunks = (
        session.query(Chunk)
        .filter(
            Chunk.knowledge_base_id == kb_id,
            Chunk.chunk_type == "faq",
            Chunk.deleted_at.is_(None),
        )
        .all()
    )
    items: dict[str, dict] = {}
    for c in chunks:
        meta = c.metadata_ or {}
        fid = meta.get("faq_id", c.knowledge_id)
        entry = items.setdefault(fid, {
            "faq_id": fid,
            "question": meta.get("standard_question", c.content),
            "similar_questions": [],
            "answer": meta.get("answer", ""),
            "tags": meta.get("tags", []),
        })
        if meta.get("role") == "similar":
            entry["similar_questions"].append(c.content)
    return {"items": list(items.values()), "total": len(items)}


@router.delete("/{kb_id}/items/{faq_id}")
async def delete_faq_item(kb_id: str, faq_id: str, session: Session = Depends(get_session)):
    now = datetime.now(timezone.utc)
    chunks = (
        session.query(Chunk)
        .filter(
            Chunk.knowledge_base_id == kb_id,
            Chunk.knowledge_id == faq_id,
            Chunk.chunk_type == "faq",
            Chunk.deleted_at.is_(None),
        )
        .all()
    )
    if not chunks:
        raise HTTPException(404, "FAQ item not found")
    ids = [c.id for c in chunks]
    for c in chunks:
        c.deleted_at = now
    session.commit()
    await get_storage().vector.delete_chunks(kb_id, ids)
    return {"status": "deleted", "chunk_count": len(ids)}


class FaqBatchBody(BaseModel):
    items: list[FaqItemBody]
    dry_run: bool = False


@router.post("/{kb_id}/items/batch")
async def import_faq_batch(kb_id: str, body: FaqBatchBody, session: Session = Depends(get_session)):
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    errors = [
        {"index": i, "error": "question/answer 不能为空"}
        for i, item in enumerate(body.items)
        if not item.question.strip() or not item.answer.strip()
    ]
    if body.dry_run or errors:
        return {"valid": len(body.items) - len(errors), "errors": errors, "imported": 0}

    rows: list[tuple[str, str, str, list[str]]] = []
    for item in body.items:
        rows.extend(_add_item_chunks(session, kb, kb_id, item))
    session.commit()

    vectors = await embed_texts([q for _, q, _, _ in rows])
    metadata = [{"knowledge_id": fid, "chunk_type": "faq", "tags": tags} for _, _, fid, tags in rows]
    await get_storage().vector.upsert(kb_id, [cid for cid, _, _, _ in rows], vectors, metadata)
    log.info("FAQ批量导入", kb_id=kb_id, items=len(body.items), chunks=len(rows))
    return {"valid": len(body.items), "errors": [], "imported": len(body.items)}


@router.get("/{kb_id}/export")
def export_faq(kb_id: str, session: Session = Depends(get_session)):
    from fastapi.responses import JSONResponse

    data = list_faq_items(kb_id, session)
    return JSONResponse(
        content=data["items"],
        headers={"Content-Disposition": f'attachment; filename="faq_{kb_id}.json"'},
    )
