from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ...fastpath.service import (
    FastPathResult,
    record_hit,
    try_fastpath,
    write_cache_background,
)
from ...generation.answer import generate_answer, generate_answer_stream
from ...models.chat import ChatMessage, ChatSession
from ...models.database import get_session
from ...retrieval.pipeline import retrieve_candidates
from ...retrieval.rewrite import rewrite_query
from ...settings import settings

router = APIRouter()


class ChatRequest(BaseModel):
    kb_ids: list[str]
    query: str
    history: list[dict] = []
    top_k: int | None = None
    stream: bool = True
    enable_wiki_boost: bool | None = None
    enable_graph: bool | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    references: list[dict]


def _sse_event(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def _fastpath_event_gen(result: FastPathResult):
    yield _sse_event("via", {"via": result.via, "layer": result.layer})
    yield _sse_event("message", {"content": result.answer})
    yield _sse_event("references", {"references": result.references})
    yield _sse_event("done", {})


@router.post("")
async def chat(body: ChatRequest, session: Session = Depends(get_session)):
    history = body.history
    if body.session_id:
        history = _load_history(session, body.session_id, settings.chat_history_rounds)
    search_query = await rewrite_query(body.query, history)

    fast = await try_fastpath(session, body.kb_ids, search_query)
    if fast is not None:
        if body.stream:
            async def fast_gen():
                async for ev in _fastpath_event_gen(fast):
                    yield ev
                if body.session_id:
                    _persist_turn(body.session_id, body.query, fast.answer, fast.references)

            return EventSourceResponse(fast_gen(), media_type="text/event-stream")
        if body.session_id:
            _persist_turn(body.session_id, body.query, fast.answer, fast.references)
        return ChatResponse(answer=fast.answer, references=fast.references)

    result = await retrieve_candidates(
        session, body.kb_ids, search_query,
        top_k=body.top_k,
        enable_graph=body.enable_graph,
        enable_wiki_boost=body.enable_wiki_boost,
        endpoint_defaults={"enable_graph": True, "enable_wiki_boost": False},
    )
    candidates = result.hits

    if body.stream:
        async def event_gen():
            answer_parts: list[str] = []
            refs: list = []
            async for event in generate_answer_stream(body.query, candidates, history, suggest=True):
                if event["event"] == "message":
                    answer_parts.append(json.loads(event["data"]).get("content", ""))
                elif event["event"] == "references":
                    refs = json.loads(event["data"]).get("references", [])
                yield event
            if body.session_id and answer_parts:
                _persist_turn(body.session_id, body.query, "".join(answer_parts), refs)
            source_ids = [r.get("chunk_id") for r in refs if r.get("chunk_id")]
            if source_ids:
                write_cache_background(_session_factory, body.kb_ids, body.query, "".join(answer_parts), refs, source_ids)

        return EventSourceResponse(event_gen(), media_type="text/event-stream")

    answer = await generate_answer(body.query, candidates, history)
    if body.session_id:
        _persist_turn(body.session_id, body.query, answer["answer"], answer["references"])
    source_ids = [r.get("chunk_id") for r in answer["references"] if r.get("chunk_id")]
    if source_ids:
        write_cache_background(_session_factory, body.kb_ids, body.query, answer["answer"], answer["references"], source_ids)
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
