from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models.chat import ChatMessage, ChatSession
from ...models.database import get_session

router = APIRouter()


class SessionCreateBody(BaseModel):
    kb_ids: list[str] = []
    title: str = "新对话"


@router.post("")
def create_chat_session(body: SessionCreateBody, session: Session = Depends(get_session)):
    row = ChatSession(title=body.title, kb_ids=body.kb_ids, tenant_id=1)
    session.add(row)
    session.commit()
    return {"id": row.id, "title": row.title}


@router.get("")
def list_chat_sessions(session: Session = Depends(get_session)):
    rows = (
        session.query(ChatSession)
        .filter(ChatSession.deleted_at.is_(None))
        .order_by(ChatSession.updated_at.desc())
        .limit(100)
        .all()
    )
    return {"sessions": [
        {"id": r.id, "title": r.title, "kb_ids": r.kb_ids, "updated_at": str(r.updated_at)}
        for r in rows
    ]}


@router.get("/{session_id}/messages")
def list_chat_messages(session_id: str, session: Session = Depends(get_session)):
    rows = (
        session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return {"messages": [
        {"id": r.id, "role": r.role, "content": r.content, "references": r.references_}
        for r in rows
    ]}


@router.delete("/{session_id}")
def delete_chat_session(session_id: str, session: Session = Depends(get_session)):
    row = session.get(ChatSession, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    row.deleted_at = datetime.now(timezone.utc)
    session.commit()
    return {"status": "deleted"}
