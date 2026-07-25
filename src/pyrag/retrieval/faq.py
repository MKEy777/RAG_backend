from __future__ import annotations

import numpy as np

from ..settings import settings
from ..storage.factory import get_storage
from .hybrid import Candidate


async def faq_search(
    kb_id: str,
    query: str,
    query_vector: np.ndarray,
    match_count: int = 5,
) -> list[Candidate]:
    storage = get_storage()
    max_iterations = settings.faq_max_iterations
    initial_top_k = match_count * settings.faq_initial_top_k_multiplier

    all_results: list[Candidate] = []
    seen_ids: set[str] = set()

    for iteration in range(max_iterations):
        top_k = initial_top_k + iteration * match_count
        results = await storage.vector.search(kb_id, query_vector, top_k, threshold=0.0)

        for r in results:
            cid = r["chunk_id"]
            if cid in seen_ids:
                continue
            meta = r.get("metadata", {})
            if meta.get("chunk_type") != "faq":
                continue
            seen_ids.add(cid)
            all_results.append(Candidate(
                chunk_id=cid,
                score=r["score"],
                channel="faq",
                chunk_type="faq",
                document_id=meta.get("knowledge_id", ""),
                metadata=meta,
            ))

        if len(all_results) >= match_count:
            break

    all_results.sort(key=lambda x: x.score, reverse=True)
    return all_results[:match_count]
