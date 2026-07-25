from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.chunk import Chunk
from ..retrieval.hybrid import Candidate


def expand_parent_chunks(session: Session, candidates: list[Candidate]) -> list[Candidate]:
    valid_candidates = []
    for c in candidates:
        if c.chunk_type in ("parent_text", "wiki_page", "faq"):
            valid_candidates.append(c)
            continue
        chunk = session.get(Chunk, c.chunk_id)
        if chunk is None:
            # chunk 不存在（可能是旧索引残留），跳过
            continue
        if chunk.deleted_at is not None:
            # chunk 已删除，跳过
            continue
        c.content = chunk.content
        meta = chunk.metadata_ or {}
        c.title_path = " > ".join(meta.get("title_path", []))
        c.document_id = chunk.knowledge_id

        if chunk.parent_chunk_id:
            parent = session.get(Chunk, chunk.parent_chunk_id)
            if parent and parent.deleted_at is None:
                c.content = parent.content
        valid_candidates.append(c)
    return valid_candidates


def build_context_xml(candidates: list[Candidate]) -> str:
    parts: list[str] = []
    for i, c in enumerate(candidates):
        ref_id = f"c{i + 1}"
        source_type = c.chunk_type or "text"
        priority = f"{c.score:.4f}"
        title_info = f' title_path="{c.title_path}"' if c.title_path else ""
        parts.append(
            f'<source id="{ref_id}" type="{source_type}" priority="{priority}"{title_info}>\n'
            f"{c.content}\n"
            f"</source>"
        )
    return "\n\n".join(parts)
