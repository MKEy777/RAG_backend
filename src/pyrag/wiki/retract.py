from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.retry import wiki_llm_retry
from ..models.chunk import Chunk
from ..models.wiki import WikiPage
from ..prompts.loader import get_prompt


def _doc_ref_match(ref: str, document_id: str) -> bool:
    return ref.split("|", 1)[0] == document_id


async def retract_document(
    session: Session, kb_id: str, document_id: str, doc_title: str
) -> None:
    pages = (
        session.query(WikiPage)
        .filter(WikiPage.knowledge_base_id == kb_id, WikiPage.deleted_at.is_(None))
        .all()
    )
    affected = [
        p for p in pages
        if any(_doc_ref_match(ref, document_id) for ref in (p.source_refs or []))
    ]
    await retract_pages(session, kb_id, document_id, doc_title, affected)


async def retract_pages(
    session: Session, kb_id: str, document_id: str, doc_title: str,
    pages: list[WikiPage],
) -> None:
    if not pages:
        return
    doc_chunk_ids = {
        cid for (cid,) in session.query(Chunk.id).filter(Chunk.knowledge_id == document_id).all()
    }

    for page in pages:
        remaining_refs = [
            ref for ref in (page.source_refs or [])
            if not _doc_ref_match(ref, document_id)
        ]

        if not remaining_refs:
            page.deleted_at = datetime.now(timezone.utc)
            page.status = "archived"
            session.commit()
            continue

        page.content = await _rewrite_without_source(page.content or "", doc_title)
        page.source_refs = remaining_refs
        page.chunk_refs = [cid for cid in (page.chunk_refs or []) if cid not in doc_chunk_ids]
        page.version = (page.version or 0) + 1
        session.commit()


async def _rewrite_without_source(content: str, deleted_title: str) -> str:
    client = get_llm_client()
    user_msg = (
        f"<deleted_documents>\n{deleted_title}\n</deleted_documents>\n"
        f"<existing_page_content>\n{content}\n</existing_page_content>"
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
