from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ...generation.answer import generate_answer, generate_answer_stream
from ...generation.context import expand_parent_chunks
from ...indexing.embedder import embed_texts
from ...models.database import get_session
from ...retrieval.fusion import rrf_fusion
from ...retrieval.hybrid import hybrid_search
from ...retrieval.rerank_stage import apply_wiki_boost, rerank_stage

router = APIRouter()


class ChatRequest(BaseModel):
    kb_ids: list[str]
    query: str
    history: list[dict] = []
    top_k: int = 10
    stream: bool = True
    enable_wiki_boost: bool = False


class ChatResponse(BaseModel):
    answer: str
    references: list[dict]


@router.post("")
async def chat(body: ChatRequest, session: Session = Depends(get_session)):
    vectors = await embed_texts([body.query])
    query_vector = vectors[0]

    all_vec = []
    all_kw = []
    for kb_id in body.kb_ids:
        vec, kw = await hybrid_search(kb_id, body.query, query_vector)
        all_vec.extend(vec)
        all_kw.extend(kw)

    fused = rrf_fusion(all_vec, all_kw)
    fused = expand_parent_chunks(session, fused)
    reranked = await rerank_stage(body.query, fused, top_k=body.top_k)
    if body.enable_wiki_boost:
        reranked = apply_wiki_boost(reranked)

    if body.stream:
        async def event_gen():
            async for event in generate_answer_stream(body.query, reranked, body.history):
                yield event

        return EventSourceResponse(event_gen(), media_type="text/event-stream")

    result = await generate_answer(body.query, reranked, body.history)
    return ChatResponse(answer=result["answer"], references=result["references"])
