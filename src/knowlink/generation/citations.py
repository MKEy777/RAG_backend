from __future__ import annotations

import re

from ..retrieval.hybrid import Candidate

CITATION_PROTOCOL = (
    "Retrieved content uses request-local source handles: cN identifies a knowledge chunk. "
    "Source citations are enabled. Cite a knowledge chunk with exactly `<ref id=\"cN\"/>`. "
    "Keep each `<ref/>` inline on the same line as the claim it supports. "
    "Do not group citations at the end."
)


_REF_PATTERN = re.compile(r'<ref\s+id="(c\d+)"\s*/>')


def extract_cited_ids(answer: str) -> set[str]:
    """从答案文本中解析实际引用的 cN 标记"""
    return set(_REF_PATTERN.findall(answer or ""))


def build_references(candidates: list[Candidate], cited_ids: set[str] | None = None) -> list[dict]:
    refs = []
    for i, c in enumerate(candidates):
        ref_id = f"c{i + 1}"
        if cited_ids is not None and ref_id not in cited_ids:
            continue
        refs.append({
            "id": ref_id,
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "title_path": c.title_path,
            "score": c.score,
            "channel": c.channel,
            "chunk_type": c.chunk_type,
        })
    return refs
