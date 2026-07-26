from __future__ import annotations

from ..retrieval.hybrid import Candidate

CITATION_PROTOCOL = (
    "Retrieved content uses request-local source handles: cN identifies a knowledge chunk. "
    "Source citations are enabled. Cite a knowledge chunk with exactly `<ref id=\"cN\"/>`. "
    "Keep each `<ref/>` inline on the same line as the claim it supports. "
    "Do not group citations at the end."
)


def build_references(candidates: list[Candidate]) -> list[dict]:
    refs = []
    for i, c in enumerate(candidates):
        refs.append({
            "id": f"c{i + 1}",
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "title_path": c.title_path,
            "score": c.score,
            "channel": c.channel,
            "chunk_type": c.chunk_type,
        })
    return refs
