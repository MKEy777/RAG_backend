from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from ...generation.context import expand_parent_chunks
from ...indexing.embedder import embed_texts
from ...llm.circuit_breaker import CircuitOpenError
from ...logging import get_logger
from ...models.database import get_session
from ...retrieval.fusion import rrf_fusion
from ...retrieval.hybrid import hybrid_search
from ...retrieval.rerank_stage import apply_wiki_boost, rerank_stage
from ...settings import settings

log = get_logger("retrieval_api")
router = APIRouter()


class RetrievalRequest(BaseModel):
    kb_ids: list[str]
    query: str
    top_k: int = 10
    enable_graph: bool = False
    enable_wiki_boost: bool = False

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


@router.post("", response_model=RetrievalResponse)
async def retrieve(body: RetrievalRequest, session: Session = Depends(get_session)):
    query_vector: np.ndarray | None = None
    try:
        vectors = await embed_texts([body.query])
        query_vector = vectors[0]
    except (CircuitOpenError, Exception) as e:
        log.warning("嵌入服务不可用_回退BM25", error=str(e))

    all_vec = []
    all_kw = []
    for kb_id in body.kb_ids:
        vec, kw = await hybrid_search(kb_id, body.query, query_vector)
        all_vec.extend(vec)
        all_kw.extend(kw)

    fused = rrf_fusion(all_vec, all_kw)

    if body.enable_graph:
        from ...retrieval.graph_boost import graph_boost

        existing_ids = {c.chunk_id for c in fused}
        for kb_id in body.kb_ids:
            graph_hits = await graph_boost(kb_id, body.query, existing_ids)
            fused.extend(graph_hits)
            fused.sort(key=lambda x: x.score, reverse=True)

    fused = expand_parent_chunks(session, fused)
    reranked = await rerank_stage(body.query, fused, top_k=body.top_k)

    if body.enable_wiki_boost:
        reranked = apply_wiki_boost(reranked)

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
        for c in reranked
    ]
    return RetrievalResponse(hits=hits)
