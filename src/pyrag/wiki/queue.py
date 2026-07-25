from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..models.wiki import WikiPendingOperation
from ..settings import settings


def enqueue_ingest(
    session: Session, kb_id: str, document_id: str, op: str = "build"
) -> None:
    dedup_key = f"{kb_id}/{document_id}"
    now = datetime.now(timezone.utc)
    available_at = now + timedelta(seconds=settings.wiki_debounce_ingest_s)

    existing = (
        session.query(WikiPendingOperation)
        .filter(
            WikiPendingOperation.knowledge_base_id == kb_id,
            WikiPendingOperation.document_id == document_id,
            WikiPendingOperation.completed_at.is_(None),
        )
        .first()
    )
    if existing:
        existing.operation = op
        existing.available_at = available_at
        existing.retry_count = 0
    else:
        session.add(WikiPendingOperation(
            document_id=document_id,
            knowledge_base_id=kb_id,
            operation=op,
            dedup_key=dedup_key,
            available_at=available_at,
            created_at=now,
        ))
    session.commit()


def peek_batch(session: Session, kb_id: str, limit: int | None = None) -> list[WikiPendingOperation]:
    limit = limit or settings.wiki_batch_max_docs
    now = datetime.now(timezone.utc)
    ops = (
        session.query(WikiPendingOperation)
        .filter(
            WikiPendingOperation.knowledge_base_id == kb_id,
            WikiPendingOperation.completed_at.is_(None),
            WikiPendingOperation.available_at <= now,
        )
        .order_by(WikiPendingOperation.created_at.asc())
        .limit(limit)
        .all()
    )

    seen: dict[str, WikiPendingOperation] = {}
    for op in ops:
        key = op.dedup_key or op.document_id
        seen[key] = op
    return list(seen.values())


def mark_completed(session: Session, op: WikiPendingOperation) -> None:
    op.completed_at = datetime.now(timezone.utc)
    session.commit()


def mark_failed(session: Session, op: WikiPendingOperation, error: str) -> None:
    from ..models.wiki import WikiDeadLetter

    op.retry_count += 1
    if op.retry_count >= settings.wiki_dead_letter_threshold:
        session.add(WikiDeadLetter(
            operation_id=op.operation_id,
            error=error[:2000],
            retry_count=op.retry_count,
            created_at=datetime.now(timezone.utc),
        ))
        session.delete(op)
    session.commit()
