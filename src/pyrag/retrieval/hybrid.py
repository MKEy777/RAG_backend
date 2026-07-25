from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger
from ..settings import settings
from ..storage.factory import get_storage

log = get_logger("retrieval")


@dataclass
class Candidate:
    chunk_id: str
    score: float
    channel: str
    chunk_type: str = "text"
    title_path: str = ""
    document_id: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)


async def hybrid_search(
    kb_id: str,
    query: str,
    query_vector: np.ndarray | None,
    top_k: int | None = None,
    vector_threshold: float | None = None,
    keyword_threshold: float | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    top_k = top_k or settings.embedding_top_k
    vector_threshold = vector_threshold if vector_threshold is not None else settings.vector_threshold
    keyword_threshold = keyword_threshold if keyword_threshold is not None else settings.keyword_threshold

    storage = get_storage()

    vector_candidates: list[Candidate] = []
    keyword_candidates: list[Candidate] = []

    kw_task = storage.keyword.search(kb_id, query, top_k, keyword_threshold)

    if query_vector is not None:
        vec_task = storage.vector.search(kb_id, query_vector, top_k, vector_threshold)
        try:
            vec_results, kw_results = await asyncio.gather(vec_task, kw_task)
        except Exception as e:
            log.warning("向量搜索失败_回退BM25", kb_id=kb_id, error=str(e))
            kw_results = await kw_task
            vec_results = []
    else:
        log.info("嵌入服务不可用_仅BM25", kb_id=kb_id)
        kw_results = await kw_task
        vec_results = []

    vector_candidates = [
        Candidate(
            chunk_id=r["chunk_id"],
            score=r["score"],
            channel="vector",
            chunk_type=r.get("metadata", {}).get("chunk_type", "text"),
            document_id=r.get("metadata", {}).get("knowledge_id", ""),
            metadata=r.get("metadata", {}),
        )
        for r in vec_results
    ]

    keyword_candidates = [
        Candidate(
            chunk_id=r["chunk_id"],
            score=r["score"],
            channel="keyword",
            chunk_type=r.get("metadata", {}).get("chunk_type", "text"),
            document_id=r.get("metadata", {}).get("knowledge_id", ""),
            metadata=r.get("metadata", {}),
        )
        for r in kw_results
    ]

    return vector_candidates, keyword_candidates
