from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from ...models.database import get_session
from ...retrieval.pipeline import enrich_faq_candidates, retrieve_candidates
from ...settings import settings

router = APIRouter()


class RetrievalRequest(BaseModel):
    kb_ids: list[str]
    query: str
    top_k: int = Field(10, ge=1, le=50)
    enable_graph: bool = False
    enable_wiki_boost: bool = False
    enable_rerank: bool = True
    expand_parent: bool = True
    recall_top_k: int | None = Field(None, ge=1, le=200)
    vector_threshold: float | None = Field(None, ge=0.0, le=1.0)
    keyword_threshold: float | None = Field(None, ge=0.0, le=1.0)
    rrf_k: int | None = Field(None, ge=1, le=1000)
    vector_weight: float | None = Field(None, ge=0.0, le=1.0)
    keyword_weight: float | None = Field(None, ge=0.0, le=1.0)
    rerank_threshold: float | None = Field(None, ge=0.0, le=1.0)
    wiki_boost: float | None = Field(None, ge=1.0, le=3.0)

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
    faq_hits: list[RetrievalHit] = []


@router.post("", response_model=RetrievalResponse)
async def retrieve(body: RetrievalRequest, session: Session = Depends(get_session)):
    result = await retrieve_candidates(
        session, body.kb_ids, body.query,
        top_k=body.top_k,
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
    )

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

    faq_out = [
        RetrievalHit(
            chunk_id=h.chunk_id, score=h.score, channel="faq", chunk_type="faq",
            title_path="", document_id=h.document_id,
            content=h.content[:500],
        )
        for h in enrich_faq_candidates(session, result.faq_hits)
    ]
    return RetrievalResponse(hits=hits, faq_hits=faq_out)
