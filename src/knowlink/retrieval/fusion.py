from __future__ import annotations

from ..settings import settings
from .hybrid import Candidate


def rrf_fusion(
    vector_candidates: list[Candidate],
    keyword_candidates: list[Candidate],
    k: int | None = None,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
) -> list[Candidate]:
    k = k or settings.rrf_k
    vector_weight = vector_weight if vector_weight is not None else settings.rrf_vector_weight
    keyword_weight = keyword_weight if keyword_weight is not None else settings.rrf_keyword_weight

    if not vector_candidates and not keyword_candidates:
        return []

    if not keyword_candidates:
        seen: dict[str, Candidate] = {}
        for c in vector_candidates:
            if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
                seen[c.chunk_id] = c
        result = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return result

    if not vector_candidates:
        seen = {}
        for c in keyword_candidates:
            if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
                seen[c.chunk_id] = c
        result = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return result

    vec_rank: dict[str, int] = {}
    for rank, c in enumerate(vector_candidates):
        if c.chunk_id not in vec_rank:
            vec_rank[c.chunk_id] = rank + 1

    kw_rank: dict[str, int] = {}
    for rank, c in enumerate(keyword_candidates):
        if c.chunk_id not in kw_rank:
            kw_rank[c.chunk_id] = rank + 1

    all_ids = set(vec_rank.keys()) | set(kw_rank.keys())
    candidates_map: dict[str, Candidate] = {}
    for c in vector_candidates:
        candidates_map.setdefault(c.chunk_id, c)
    for c in keyword_candidates:
        if c.chunk_id not in candidates_map:
            candidates_map[c.chunk_id] = c

    fused: list[Candidate] = []
    for cid in all_ids:
        vr = vec_rank.get(cid)
        kr = kw_rank.get(cid)
        score = 0.0
        if vr is not None:
            score += vector_weight / (k + vr)
        if kr is not None:
            score += keyword_weight / (k + kr)

        c = candidates_map[cid]
        channel = "vector"
        if vr is not None and kr is not None:
            channel = "hybrid"
        elif kr is not None:
            channel = "keyword"

        fused.append(Candidate(
            chunk_id=c.chunk_id,
            score=score,
            channel=channel,
            chunk_type=c.chunk_type,
            title_path=c.title_path,
            document_id=c.document_id,
            content=c.content,
            metadata=c.metadata,
        ))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused
