from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ...generation.answer import generate_answer, generate_answer_stream
from ...models.database import get_session
from ...retrieval.pipeline import enrich_faq_candidates, retrieve_candidates

router = APIRouter()


class ChatRequest(BaseModel):
    kb_ids: list[str]
    query: str
    history: list[dict] = []
    top_k: int = 10
    stream: bool = True
    enable_wiki_boost: bool = False
    enable_graph: bool = True


class ChatResponse(BaseModel):
    answer: str
    references: list[dict]


@router.post("")
async def chat(body: ChatRequest, session: Session = Depends(get_session)):
    result = await retrieve_candidates(
        session, body.kb_ids, body.query,
        top_k=body.top_k,
        enable_graph=body.enable_graph,
        enable_wiki_boost=body.enable_wiki_boost,
    )
    # FAQ 命中置于候选前部：generation/context.py 的 build_context_xml 按 chunk_type 标注来源
    candidates = enrich_faq_candidates(session, result.faq_hits) + result.hits

    if body.stream:
        async def event_gen():
            async for event in generate_answer_stream(body.query, candidates, body.history):
                yield event

        return EventSourceResponse(event_gen(), media_type="text/event-stream")

    answer = await generate_answer(body.query, candidates, body.history)
    return ChatResponse(answer=answer["answer"], references=answer["references"])
