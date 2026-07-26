from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.retry import wiki_llm_retry
from ..models.wiki import WikiPage
from ..prompts.loader import get_prompt


async def retract_document(
    session: Session, kb_id: str, document_id: str, doc_title: str
) -> None:
    pages = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.deleted_at.is_(None),
        )
        .all()
    )

    source_ref_key = f"{document_id}|{doc_title}"
    affected = [
        p for p in pages
        if any(document_id in ref for ref in (p.source_refs or []))
    ]

    if not affected:
        return

    for page in affected:
        remaining_refs = [
            ref for ref in (page.source_refs or [])
            if document_id not in ref
        ]

        if not remaining_refs:
            page.deleted_at = datetime.now(timezone.utc)
            page.status = "archived"
            session.commit()
            continue

        new_content = await _rewrite_without_source(page.content or "", doc_title)

        page.content = new_content
        page.source_refs = remaining_refs
        page.chunk_refs = [
            cid for cid in (page.chunk_refs or [])
            if not cid.startswith(document_id)
        ]
        page.version = (page.version or 0) + 1
        session.commit()

    from .finalize import finalize

    kb_page = (
        session.query(WikiPage)
        .filter(WikiPage.knowledge_base_id == kb_id)
        .first()
    )
    tenant_id = kb_page.tenant_id if kb_page else 1
    await finalize(session, kb_id, tenant_id)


async def _rewrite_without_source(content: str, deleted_title: str) -> str:
    client = get_llm_client()
    user_msg = (
        f"<deleted_documents>\n{deleted_title}\n</deleted_documents>\n"
        f"<existing_page_content>\n{content}\n</existing_page_content>\n"
        f"<remaining_source_documents>\n(see page context)\n</remaining_source_documents>"
    )
    try:
        @wiki_llm_retry()
        async def _call():
            return await client.chat(
                [
                    {"role": "system", "content": get_prompt("wiki", "WikiPageModifyPrompt")},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4096,
                purpose="wiki:retract",
            )

        resp = await _call()
        lines = resp.strip().split("\n")
        if lines and lines[0].startswith("SUMMARY:"):
            if "(empty page)" in lines[0]:
                return ""
            return "\n".join(lines[1:]).strip()
        return resp.strip()
    except Exception:
        return content
