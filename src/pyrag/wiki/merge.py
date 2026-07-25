from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.retry import wiki_llm_retry
from ..models.wiki import WikiPage, WikiPageRevision
from ..settings import settings
from ..storage.factory import get_storage
from .ingest import MapResult

WIKI_PAGE_MODIFY_PROMPT = (
    "You are a wiki editor tasked with updating an existing wiki page. "
    "Process NEW information to add AND/OR deleted documents whose contributions must be REMOVED.\n"
    "First line: SUMMARY: {one sentence summary}\n"
    "Stay close to source wording. Do not add filler phrases. "
    "Use [[slug|DisplayName]] for wiki links only to valid pages.\n"
    "If page becomes empty after removal, output: SUMMARY: (empty page)"
)

WIKI_DEDUP_PROMPT = (
    "You are a strict deduplication system. Given newly extracted items and existing wiki pages, "
    "determine which new items refer to the exact same real-world entity or concept.\n"
    "Three conditions must ALL hold: (1) same referent; (2) name variant (abbreviation, translation, spelling); "
    "(3) type compatible.\n"
    "related ≠ same. Prefer separate pages over wrong merges.\n"
    "Output: {\"merges\": {\"new_slug\": \"existing_slug\", ...}}"
)


async def reduce_pages(
    session: Session,
    kb_id: str,
    tenant_id: int,
    document_id: str,
    doc_title: str,
    map_result: MapResult,
) -> None:
    storage = get_storage()
    lock = storage.lock

    if map_result.overview_content:
        slug = f"overview/{doc_title}"
        await _write_page(
            session, lock, kb_id, tenant_id, slug,
            kind="overview", title=doc_title,
            content=map_result.overview_content,
            summary=map_result.overview_summary,
            source_doc=document_id, doc_title=doc_title,
            chunk_refs=[],
        )

    for candidate in map_result.candidates:
        slug = candidate.get("slug", "")
        name = candidate.get("name", "")
        if not slug or not name:
            continue
        kind = candidate.get("_type", "entity")
        details = candidate.get("details", candidate.get("description", ""))
        chunk_ids = map_result.citations.get(slug, [])

        await _write_page(
            session, lock, kb_id, tenant_id, slug,
            kind=kind, title=name,
            content=details,
            summary=candidate.get("description", ""),
            source_doc=document_id, doc_title=doc_title,
            chunk_refs=chunk_ids,
            aliases=candidate.get("aliases", []),
        )


async def _write_page(
    session: Session,
    lock,
    kb_id: str,
    tenant_id: int,
    slug: str,
    kind: str,
    title: str,
    content: str,
    summary: str,
    source_doc: str,
    doc_title: str,
    chunk_refs: list[str],
    aliases: list[str] | None = None,
) -> None:
    lock_key = f"wiki:slug:{kb_id}:{slug}"
    acquired = await lock.acquire(lock_key, ttl_s=settings.wiki_slug_lock_ttl_s)

    try:
        page = (
            session.query(WikiPage)
            .filter(WikiPage.knowledge_base_id == kb_id, WikiPage.slug == slug)
            .first()
        )

        source_ref = f"{source_doc}|{doc_title}"

        if page is None:
            page = WikiPage(
                page_id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                kind=kind,
                topic_key=f"{kb_id}:{kind}:{slug}",
                title=title,
                slug=slug,
                content=content,
                summary=summary,
                source_refs=[source_ref],
                chunk_refs=chunk_refs,
                aliases=aliases or [],
                status="published",
                version=1,
            )
            session.add(page)
        else:
            if content and page.content:
                merged = await _merge_content(page.content, content, title, slug)
                page.content = merged
            elif content:
                page.content = content

            if summary:
                page.summary = summary
            if source_ref not in (page.source_refs or []):
                page.source_refs = (page.source_refs or []) + [source_ref]
            existing_refs = set(page.chunk_refs or [])
            existing_refs.update(chunk_refs)
            page.chunk_refs = list(existing_refs)
            if aliases:
                existing_aliases = set(page.aliases or [])
                existing_aliases.update(aliases)
                page.aliases = list(existing_aliases)
            page.version = (page.version or 0) + 1

        session.commit()

        revision = WikiPageRevision(
            revision_id=uuid.uuid4().hex,
            page_id=page.page_id,
            title=page.title,
            content=page.content or "",
            content_hash=hashlib.sha256((page.content or "").encode()).hexdigest(),
            created_at=datetime.now(timezone.utc),
        )
        session.add(revision)
        session.commit()

    finally:
        if acquired:
            await lock.release(lock_key)


async def _merge_content(existing: str, new_info: str, title: str, slug: str) -> str:
    client = get_llm_client()
    user_msg = (
        f"Page: {title} (slug: {slug})\n"
        f"<existing_page_content>\n{existing}\n</existing_page_content>\n"
        f"<new_information>\n{new_info}\n</new_information>"
    )
    try:
        @wiki_llm_retry()
        async def _call():
            return await client.chat(
                [
                    {"role": "system", "content": WIKI_PAGE_MODIFY_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

        resp = await _call()
        lines = resp.strip().split("\n")
        if lines and lines[0].startswith("SUMMARY:"):
            return "\n".join(lines[1:]).strip()
        return resp.strip()
    except Exception:
        return f"{existing}\n\n{new_info}"
