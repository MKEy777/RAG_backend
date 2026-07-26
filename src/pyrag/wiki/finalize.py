from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.retry import wiki_llm_retry
from ..models.wiki import WikiFolder, WikiPage
from ..prompts.loader import get_prompt
from ..settings import settings
from ..storage.factory import get_storage

WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


async def finalize(session: Session, kb_id: str, tenant_id: int) -> None:
    storage = get_storage()
    lock = storage.lock
    lock_key = f"wiki:finalize:{kb_id}"
    acquired = await lock.acquire(lock_key, ttl_s=settings.wiki_finalize_lock_ttl_s)
    if not acquired:
        return

    try:
        await _rebuild_index_page(session, kb_id, tenant_id)
        await _assign_taxonomy(session, kb_id)

        from .synthesis import generate_comparisons, generate_synthesis

        await generate_synthesis(session, kb_id, tenant_id)
        await generate_comparisons(session, kb_id, tenant_id)

        await _inject_cross_links(session, kb_id)
        await _clean_dead_links(session, kb_id)
    finally:
        await lock.release(lock_key)


async def _rebuild_index_page(session: Session, kb_id: str, tenant_id: int) -> None:
    pages = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.kind != "index",
            WikiPage.deleted_at.is_(None),
        )
        .all()
    )
    if not pages:
        return

    summaries = [f"- {p.title}: {p.summary or ''}" for p in pages if p.summary]
    if not summaries:
        return

    client = get_llm_client()
    doc_summaries = "\n".join(summaries[:50])

    index_page = (
        session.query(WikiPage)
        .filter(WikiPage.knowledge_base_id == kb_id, WikiPage.kind == "index")
        .first()
    )

    try:
        if index_page and index_page.content:
            prompt = get_prompt("wiki", "WikiIndexIntroUpdatePrompt")
            user_msg = f"<current_intro>\n{index_page.content}\n</current_intro>\n<document_summaries>\n{doc_summaries}\n</document_summaries>"
        else:
            prompt = get_prompt("wiki", "WikiIndexIntroPrompt")
            user_msg = f"<document_summaries>\n{doc_summaries}\n</document_summaries>"

        @wiki_llm_retry()
        async def _call():
            return await client.chat(
                [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
                temperature=0.3,
                max_tokens=1024,
                purpose="wiki:generate_intro",
            )

        intro = await _call()

        if index_page is None:
            index_page = WikiPage(
                page_id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                kind="index",
                topic_key=f"{kb_id}:index:main",
                title="Index",
                slug="index/main",
                content=intro,
                summary="Knowledge base index",
                status="published",
                version=1,
            )
            session.add(index_page)
        else:
            index_page.content = intro

        session.commit()
    except Exception:
        pass


async def _assign_taxonomy(session: Session, kb_id: str) -> None:
    from ..llm.json_repair import parse_json_object

    unassigned = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.folder_id.is_(None),
            WikiPage.kind.notin_(["index"]),
            WikiPage.deleted_at.is_(None),
        )
        .all()
    )
    if not unassigned:
        return

    client = get_llm_client()
    items = [{"slug": p.slug, "title": p.title, "type": p.kind} for p in unassigned]
    user_msg = f"<items>\n{items}\n</items>"

    try:
        @wiki_llm_retry()
        async def _call():
            return await client.chat(
                [{"role": "system", "content": get_prompt("wiki", "WikiTaxonomyPlanPrompt")}, {"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=2048,
                purpose="wiki:classify",
            )

        resp = await _call()
        data = parse_json_object(resp)
        assignments = data.get("assignments", [])

        for assignment in assignments:
            slug = assignment.get("slug", "")
            path = assignment.get("path", [])
            if not slug or not path:
                continue

            folder = _get_or_create_folder(session, kb_id, path)
            page = next((p for p in unassigned if p.slug == slug), None)
            if page and folder:
                page.folder_id = folder.id

        session.commit()
    except Exception:
        pass


def _get_or_create_folder(session: Session, kb_id: str, path: list[str]) -> WikiFolder | None:
    parent_id = None
    folder = None
    for name in path:
        folder = (
            session.query(WikiFolder)
            .filter(
                WikiFolder.knowledge_base_id == kb_id,
                WikiFolder.name == name,
                WikiFolder.parent_id == parent_id,
            )
            .first()
        )
        if folder is None:
            folder = WikiFolder(
                id=uuid.uuid4().hex,
                knowledge_base_id=kb_id,
                name=name,
                parent_id=parent_id,
            )
            session.add(folder)
            session.flush()
        parent_id = folder.id
    return folder


async def _inject_cross_links(session: Session, kb_id: str) -> None:
    pages = (
        session.query(WikiPage)
        .filter(WikiPage.knowledge_base_id == kb_id, WikiPage.deleted_at.is_(None))
        .all()
    )
    slug_set = {p.slug for p in pages}

    for page in pages:
        if not page.content:
            continue
        links = WIKI_LINK_RE.findall(page.content)
        valid_links = [l for l in links if l in slug_set]
        page.out_links = valid_links

    for page in pages:
        in_links = []
        for other in pages:
            if other.page_id != page.page_id and page.slug in (other.out_links or []):
                in_links.append(other.slug)
        page.in_links = in_links

    session.commit()


async def _clean_dead_links(session: Session, kb_id: str) -> None:
    pages = (
        session.query(WikiPage)
        .filter(WikiPage.knowledge_base_id == kb_id, WikiPage.deleted_at.is_(None))
        .all()
    )
    slug_set = {p.slug for p in pages}

    for page in pages:
        if not page.out_links:
            continue
        cleaned = [l for l in page.out_links if l in slug_set]
        if len(cleaned) != len(page.out_links):
            page.out_links = cleaned

    session.commit()
