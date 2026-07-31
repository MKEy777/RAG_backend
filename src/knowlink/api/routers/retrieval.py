from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from ...logging import get_logger
from ...models.database import get_session
from ...models.retrieval_test import RetrievalTest
from ...retrieval.pipeline import retrieve_candidates
from ...settings import settings

log = get_logger("retrieval_api")
router = APIRouter()

# 端点历史默认：/retrieval 图谱/wiki 加权默认关（与旧版请求模型一致）
_ENDPOINT_DEFAULTS = {"enable_graph": False, "enable_wiki_boost": False}


class RetrievalRequest(BaseModel):
    kb_ids: list[str]
    query: str
    search_mode: str | None = Field(None, pattern="^(hybrid|vector|fulltext)$")
    top_k: int | None = Field(None, ge=1, le=50)
    enable_graph: bool | None = None
    enable_wiki_boost: bool | None = None
    enable_rerank: bool | None = None
    expand_parent: bool | None = None
    recall_top_k: int | None = Field(None, ge=1, le=200)
    vector_threshold: float | None = Field(None, ge=0.0, le=1.0)
    keyword_threshold: float | None = Field(None, ge=0.0, le=1.0)
    rrf_k: int | None = Field(None, ge=1, le=1000)
    vector_weight: float | None = Field(None, ge=0.0, le=1.0)
    keyword_weight: float | None = Field(None, ge=0.0, le=1.0)
    rerank_threshold: float | None = Field(None, ge=0.0, le=1.0)
    wiki_boost: float | None = Field(None, ge=1.0, le=3.0)
    save_history: bool = True

    @field_validator("kb_ids")
    @classmethod
    def validate_kb_ids(cls, v):
        if not v:
            raise ValueError("kb_ids must not be empty")
        if len(v) > settings.max_kb_ids_per_query:
            raise ValueError(f"kb_ids exceeds max {settings.max_kb_ids_per_query}")
        return v


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    channel: str
    chunk_type: str
    title_path: str
    document_id: str
    content: str


class RetrievalResponse(BaseModel):
    hits: list[RetrievalHit]


def _save_test_history(session: Session, body: RetrievalRequest, hit_count: int, top_score: float | None, duration_ms: int) -> None:
    try:
        params = body.model_dump(exclude={"kb_ids", "query", "save_history"}, exclude_none=True)
        session.add(
            RetrievalTest(
                id=uuid.uuid4().hex,
                tenant_id=1,
                kb_ids=body.kb_ids,
                query=body.query,
                params_snapshot=params,
                hit_count=hit_count,
                top_score=top_score,
                duration_ms=duration_ms,
            )
        )
        session.commit()
    except Exception as e:
        session.rollback()
        log.warning("命中历史写入失败", error=str(e)[:200])


@router.post("", response_model=RetrievalResponse)
async def retrieve(body: RetrievalRequest, session: Session = Depends(get_session)):
    start = time.perf_counter()
    result = await retrieve_candidates(
        session, body.kb_ids, body.query,
        top_k=body.top_k,
        search_mode=body.search_mode,
        enable_graph=body.enable_graph,
        enable_rerank=body.enable_rerank,
        expand_parent=body.expand_parent,
        enable_wiki_boost=body.enable_wiki_boost,
        recall_top_k=body.recall_top_k,
        vector_threshold=body.vector_threshold,
        keyword_threshold=body.keyword_threshold,
        rrf_k=body.rrf_k,
        vector_weight=body.vector_weight,
        keyword_weight=body.keyword_weight,
        rerank_threshold=body.rerank_threshold,
        wiki_boost=body.wiki_boost,
        endpoint_defaults=_ENDPOINT_DEFAULTS,
    )
    duration_ms = int((time.perf_counter() - start) * 1000)

    hits = [
        RetrievalHit(
            chunk_id=c.chunk_id,
            score=c.score,
            channel=c.channel,
            chunk_type=c.chunk_type,
            title_path=c.title_path,
            document_id=c.document_id,
            content=c.content[:500],
        )
        for c in result.hits
    ]

    if body.save_history:
        top_score = hits[0].score if hits else None
        _save_test_history(session, body, len(hits), top_score, duration_ms)

    return RetrievalResponse(hits=hits)


@router.get("/history")
def list_history(
    kb_id: str | None = None,
    page: int = 1,
    size: int = 20,
    session: Session = Depends(get_session),
):
    from sqlalchemy import cast, String as SAString

    query = session.query(RetrievalTest)
    if kb_id:
        query = query.filter(cast(RetrievalTest.kb_ids, SAString).like(f'%"{kb_id}"%'))
    query = query.order_by(RetrievalTest.created_at.desc())
    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    return {
        "items": [
            {
                "id": t.id,
                "kb_ids": t.kb_ids,
                "query": t.query,
                "params_snapshot": t.params_snapshot,
                "hit_count": t.hit_count,
                "top_score": t.top_score,
                "duration_ms": t.duration_ms,
                "created_at": t.created_at,
            }
            for t in items
        ],
        "total": total,
        "page": page,
        "size": size,
    }


@router.delete("/history/{test_id}")
def delete_history(test_id: str, session: Session = Depends(get_session)):
    row = session.get(RetrievalTest, test_id)
    if not row:
        raise HTTPException(404, "History not found")
    session.delete(row)
    session.commit()
    return {"status": "deleted"}
