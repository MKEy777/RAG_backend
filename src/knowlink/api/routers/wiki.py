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


class PageUpdateBody(BaseModel):
    title: str | None = None
    content: str | None = None
    folder_id: str | None = None


@router.put("/{kb_id}/pages/{page_id}")
async def update_wiki_page(kb_id: str, page_id: str, body: PageUpdateBody, session: Session = Depends(get_session)):
    import uuid as _uuid
    from datetime import datetime, timezone

    from ...indexing.embedder import content_hash as _hash, embed_texts
    from ...models.chunk import Chunk
    from ...storage.factory import get_storage

    page = (
        session.query(WikiPage)
        .filter(WikiPage.page_id == page_id, WikiPage.knowledge_base_id == kb_id, WikiPage.deleted_at.is_(None))
        .first()
    )
    if not page:
        raise HTTPException(404, "Page not found")

    content_changed = body.content is not None and body.content != page.content
    if body.title is not None:
        page.title = body.title
    if body.content is not None:
        page.content = body.content
    if body.folder_id is not None:
        page.folder_id = body.folder_id or None
    if content_changed:
        page.version += 1
    session.commit()

    if content_changed and page.content:
        storage = get_storage()
        old = (
            session.query(Chunk)
            .filter(Chunk.knowledge_base_id == kb_id, Chunk.chunk_type == "wiki_page",
                    Chunk.knowledge_id == page.page_id, Chunk.deleted_at.is_(None))
            .all()
        )
        now = datetime.now(timezone.utc)
        old_ids = [c.id for c in old]
        for c in old:
            c.deleted_at = now
        new_id = _uuid.uuid4().hex
        session.add(Chunk(
            id=new_id, tenant_id=page.tenant_id, knowledge_base_id=kb_id,
            knowledge_id=page.page_id, content=page.content, chunk_type="wiki_page",
            content_hash=_hash(page.content),
            metadata_={"title_path": [page.title], "doc_title": page.title, "slug": page.slug},
        ))
        session.commit()
        text = f"{page.title}\n{page.content}"
        vectors = await embed_texts([text])
        meta = [{"knowledge_id": kb_id, "chunk_type": "wiki_page"}]
        await storage.vector.upsert(kb_id, [new_id], vectors, meta)
        await storage.keyword.index_chunks(kb_id, [new_id], [text], meta)
        if old_ids:
            await storage.vector.delete_chunks(kb_id, old_ids)
            await storage.keyword.delete_chunks(kb_id, old_ids)
    return {"status": "updated", "version": page.version}


class FolderBody(BaseModel):
    name: str
    parent_id: str | None = None


@router.get("/{kb_id}/folders")
def list_folders(kb_id: str, session: Session = Depends(get_session)):
    rows = session.query(WikiFolder).filter(WikiFolder.knowledge_base_id == kb_id).all()
    return {"folders": [{"id": f.id, "name": f.name, "parent_id": f.parent_id} for f in rows]}


@router.post("/{kb_id}/folders")
def create_folder(kb_id: str, body: FolderBody, session: Session = Depends(get_session)):
    row = WikiFolder(knowledge_base_id=kb_id, name=body.name, parent_id=body.parent_id)
    session.add(row)
    session.commit()
    return {"id": row.id, "name": row.name, "parent_id": row.parent_id}


@router.delete("/{kb_id}/folders/{folder_id}")
def delete_folder(kb_id: str, folder_id: str, session: Session = Depends(get_session)):
    row = session.get(WikiFolder, folder_id)
    if not row or row.knowledge_base_id != kb_id:
        raise HTTPException(404, "Folder not found")
    session.query(WikiPage).filter(WikiPage.folder_id == folder_id).update({"folder_id": None})
    session.query(WikiFolder).filter(WikiFolder.parent_id == folder_id).update({"parent_id": row.parent_id})
    session.delete(row)
    session.commit()
    return {"status": "deleted"}
