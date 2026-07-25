from __future__ import annotations

from ..llm.rerank import get_reranker
from ..settings import settings
from .hybrid import Candidate

RERANK_THRESHOLD_FALLBACK_RATIO = 0.7
RERANK_THRESHOLD_FLOOR = 0.3


async def rerank_stage(
    query: str,
    candidates: list[Candidate],
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[Candidate]:
    top_k = top_k or settings.rerank_top_k
    threshold = threshold if threshold is not None else settings.rerank_threshold

    if not candidates:
        return []

    documents = [c.content or c.chunk_id for c in candidates]
    reranker = get_reranker()

    try:
        results = await reranker.rerank(query, documents, top_k=len(candidates))
    except Exception:
        return candidates[:top_k]

    scored: list[tuple[Candidate, float]] = []
    for item in results:
        idx = item["index"]
        score = item["score"]
        if idx < len(candidates):
            scored.append((candidates[idx], score))

    filtered = [(c, s) for c, s in scored if s >= threshold]

    if not filtered and scored:
        fallback_threshold = max(threshold * RERANK_THRESHOLD_FALLBACK_RATIO, RERANK_THRESHOLD_FLOOR)
        filtered = [(c, s) for c, s in scored if s >= fallback_threshold]

    filtered.sort(key=lambda x: x[1], reverse=True)
    result = []
    for c, s in filtered[:top_k]:
        c.score = s
        result.append(c)
    return result


def apply_wiki_boost(candidates: list[Candidate], boost: float | None = None) -> list[Candidate]:
    boost = boost if boost is not None else settings.wiki_boost
    for c in candidates:
        if c.chunk_type == "wiki_page":
            c.score *= boost
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates
