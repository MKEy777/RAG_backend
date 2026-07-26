from __future__ import annotations

import socket
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...logging import get_logger
from ...models.database import get_session
from ...models.wiki import WikiFolder, WikiPage, WikiPendingOperation

log = get_logger("wiki")
router = APIRouter()

GROUP_ORDER = ["overview", "concept", "entity", "synthesis", "comparison"]  # detail.md §C4


def build_index_payload(session: Session, kb_id: str) -> dict:
    pages = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.deleted_at.is_(None),
            WikiPage.status == "published",
        )
        .order_by(WikiPage.title)
        .all()
    )
    if not pages:
        return {"intro": "", "groups": [], "folders": []}

    intro = ""
    by_kind: dict[str, list[WikiPage]] = {}
    for p in pages:
        if p.kind == "index":
            intro = p.content or ""
            continue
        by_kind.setdefault(p.kind, []).append(p)

    groups = []
    for kind in GROUP_ORDER:
        items = by_kind.get(kind)
        if not items:
            continue
        groups.append({
            "type": kind,
            "total": len(items),
            "items": [
                {"slug": p.slug, "title": p.title, "summary": p.summary or "", "folder_id": p.folder_id}
                for p in items
            ],
        })

    folder_rows = session.query(WikiFolder).filter(WikiFolder.knowledge_base_id == kb_id).all()
    page_counts: dict[str, int] = {}
    for p in pages:
        if p.folder_id:
            page_counts[p.folder_id] = page_counts.get(p.folder_id, 0) + 1
    folders = [
        {"id": f.id, "name": f.name, "parent_id": f.parent_id, "page_count": page_counts.get(f.id, 0)}
        for f in folder_rows
    ]
    return {"intro": intro, "groups": groups, "folders": folders}


@router.get("/{kb_id}/index")
def wiki_index(kb_id: str, session: Session = Depends(get_session)):
    return build_index_payload(session, kb_id)


class WikiPageSummary(BaseModel):
    page_id: str
    title: str
    slug: str
    kind: str
    summary: str | None
    status: str
    version: int
    folder_id: str | None
    updated_at: datetime | None

    class Config:
        from_attributes = True


class WikiPageDetail(BaseModel):
    page_id: str
    title: str
    slug: str
    kind: str
    summary: str | None
    content: str | None
    source_refs: list | None
    out_links: list | None
    in_links: list | None
    aliases: list | None
    status: str
    version: int
    folder_id: str | None
    updated_at: datetime | None

    class Config:
        from_attributes = True


def _is_redis_available() -> bool:
    try:
        with socket.create_connection(("localhost", 6379), timeout=1):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


@router.get("/{kb_id}/pages")
def list_wiki_pages(
    kb_id: str,
    page: int = 1,
    size: int = 20,
    session: Session = Depends(get_session),
):
    query = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.deleted_at.is_(None),
        )
    )
    total = query.count()
    items = query.order_by(WikiPage.kind, WikiPage.title).offset((page - 1) * size).limit(size).all()
    return {
        "items": [WikiPageSummary.model_validate(p) for p in items],
        "total": total,
        "page": page,
        "size": size,
    }


@router.get("/{kb_id}/pages/{page_id}")
def get_wiki_page(
    kb_id: str,
    page_id: str,
    session: Session = Depends(get_session),
):
    page = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.page_id == page_id,
            WikiPage.deleted_at.is_(None),
        )
        .first()
    )
    if not page:
        raise HTTPException(404, "Wiki page not found")
    return page


@router.post("/{kb_id}/build")
def build_wiki(kb_id: str):
    if _is_redis_available():
        from ...tasks.parse_tasks import wiki_build_task

        wiki_build_task.delay(kb_id)
        log.info("Wiki 构建任务已派发", kb_id=kb_id)
    else:
        import threading

        from ...models.database import SessionLocal
        from ...wiki.build import run_wiki_build

        def _run_wiki_sync():
            session = SessionLocal()
            try:
                run_wiki_build(session, kb_id, enqueue_missing=True)
            except Exception as e:
                log.error("同步 Wiki 构建失败", kb_id=kb_id, error=str(e))
            finally:
                session.close()

        threading.Thread(target=_run_wiki_sync, daemon=True).start()
        log.info("使用同步 Wiki 构建（Redis 不可用）", kb_id=kb_id)

    return {"status": "started"}


@router.get("/{kb_id}/status")
def wiki_status(kb_id: str, session: Session = Depends(get_session)):
    page_count = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.deleted_at.is_(None),
        )
        .count()
    )
    pending_count = (
        session.query(WikiPendingOperation)
        .filter(
            WikiPendingOperation.knowledge_base_id == kb_id,
            WikiPendingOperation.completed_at.is_(None),
        )
        .count()
    )
    return {"page_count": page_count, "pending_count": pending_count}
