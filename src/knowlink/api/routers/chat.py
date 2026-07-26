from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ...generation.answer import generate_answer, generate_answer_stream
from ...models.chat import ChatMessage, ChatSession
from ...models.database import get_session
from ...retrieval.pipeline import enrich_faq_candidates, retrieve_candidates
from ...retrieval.rewrite import rewrite_query
from ...settings import settings

router = APIRouter()


class ChatRequest(BaseModel):
    kb_ids: list[str]
    query: str
    history: list[dict] = []
    top_k: int = 10
    stream: bool = True
    enable_wiki_boost: bool = False
    enable_graph: bool = True
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    references: list[dict]


@router.post("")
async def chat(body: ChatRequest, session: Session = Depends(get_session)):
    history = body.history
    if body.session_id:
        history = _load_history(session, body.session_id, settings.chat_history_rounds)
    search_query = await rewrite_query(body.query, history)
    result = await retrieve_candidates(
        session, body.kb_ids, search_query,
        top_k=body.top_k,
        enable_graph=body.enable_graph,
        enable_wiki_boost=body.enable_wiki_boost,
    )
    candidates = enrich_faq_candidates(session, result.faq_hits) + result.hits

    if body.stream:
        async def event_gen():
            answer_parts: list[str] = []
            refs: list = []
            async for event in generate_answer_stream(body.query, candidates, history):
                if event["event"] == "message":
                    answer_parts.append(json.loads(event["data"]).get("content", ""))
                elif event["event"] == "references":
                    refs = json.loads(event["data"]).get("references", [])
                yield event
            if body.session_id and answer_parts:
                _persist_turn(body.session_id, body.query, "".join(answer_parts), refs)

        return EventSourceResponse(event_gen(), media_type="text/event-stream")

    answer = await generate_answer(body.query, candidates, history)
    if body.session_id:
        _persist_turn(body.session_id, body.query, answer["answer"], answer["references"])
    return ChatResponse(answer=answer["answer"], references=answer["references"])


def _load_history(session: Session, session_id: str, rounds: int) -> list[dict]:
    rows = (
        session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(rounds * 2)
        .all()
    )
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def _session_factory():
    from ...models.database import SessionLocal

    return SessionLocal()


def _persist_turn(session_id: str, query: str, answer: str, references: list) -> None:
    db = _session_factory()
    try:
        db.add(ChatMessage(session_id=session_id, role="user", content=query))
        db.add(ChatMessage(session_id=session_id, role="assistant", content=answer, references_=references))
        row = db.get(ChatSession, session_id)
        if row and row.title == "新对话":
            row.title = query[:30]
        db.commit()
    finally:
        db.close()
