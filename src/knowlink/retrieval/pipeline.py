from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from ..generation.context import expand_parent_chunks
from ..indexing.embedder import embed_texts
from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.knowledge_base import KnowledgeBase
from .dedup import dedup_candidates
from .faq import faq_search
from .fusion import rrf_fusion
from .graph_boost import graph_boost
from .hybrid import Candidate, hybrid_search
from .rerank_stage import apply_wiki_boost, rerank_stage

log = get_logger("retrieval")


@dataclass
class RetrievalResult:
    hits: list[Candidate] = field(default_factory=list)
    faq_hits: list[Candidate] = field(default_factory=list)


async def retrieve_candidates(
    session: Session,
    kb_ids: list[str],
    query: str,
    top_k: int = 10,
    enable_graph: bool = True,
    enable_rerank: bool = True,
    expand_parent: bool = True,
    enable_wiki_boost: bool = False,
    recall_top_k: int | None = None,
    vector_threshold: float | None = None,
    keyword_threshold: float | None = None,
    rrf_k: int | None = None,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
    rerank_threshold: float | None = None,
    wiki_boost: float | None = None,
) -> RetrievalResult:
    query_vector: np.ndarray | None = None
    try:
        vectors = await embed_texts([query])
        query_vector = vectors[0]
    except Exception as e:
        log.warning("嵌入服务不可用_回退BM25", error=str(e))

    faq_kbs: list[KnowledgeBase] = []
    doc_kb_ids: list[str] = []
    for kb_id in kb_ids:
        kb = session.get(KnowledgeBase, kb_id)
        if kb and kb.type == "faq":
            faq_kbs.append(kb)
        else:
            doc_kb_ids.append(kb_id)

    result = RetrievalResult()

    # FAQ 独立路径：不参与融合与 wiki 加权（detail.md §FAQ 检索方式）
    if faq_kbs and query_vector is not None:
        faq_tasks = []
        for kb in faq_kbs:
            tag_priority = (kb.faq_config or {}).get("tag_priority", [])
            faq_tasks.append(faq_search(kb.id, query, query_vector, top_k, tag_priority))
        for hits in await asyncio.gather(*faq_tasks, return_exceptions=True):
            if isinstance(hits, Exception):
                log.warning("FAQ检索失败", error=str(hits))
                continue
            result.faq_hits.extend(hits)

    pairs = await asyncio.gather(
        *[
            hybrid_search(
                kb_id, query, query_vector,
                top_k=recall_top_k,
                vector_threshold=vector_threshold,
                keyword_threshold=keyword_threshold,
            )
            for kb_id in doc_kb_ids
        ],
        return_exceptions=True,
    )
    all_vec: list[Candidate] = []
    all_kw: list[Candidate] = []
    for pair in pairs:
        if isinstance(pair, BaseException):
            log.warning("混合召回失败", error=str(pair))
            continue
        vec, kw = pair
        all_vec.extend(vec)
        all_kw.extend(kw)

    fused = rrf_fusion(all_vec, all_kw, k=rrf_k, vector_weight=vector_weight, keyword_weight=keyword_weight)

    if enable_graph and doc_kb_ids:
        existing_ids = {c.chunk_id for c in fused}
        graph_results = await asyncio.gather(
            *[graph_boost(kb_id, query, existing_ids) for kb_id in doc_kb_ids],
            return_exceptions=True,
        )
        for hits in graph_results:
            if isinstance(hits, Exception):
                log.warning("图谱通道失败", error=str(hits))
                continue
            fused.extend(hits)
        fused.sort(key=lambda x: x.score, reverse=True)

    if expand_parent:
        fused = expand_parent_chunks(session, fused)

    fused = dedup_candidates(fused)

    if enable_rerank:
        reranked = await rerank_stage(query, fused, top_k=top_k, threshold=rerank_threshold)
    else:
        reranked = fused[:top_k]

    if enable_wiki_boost:
        reranked = apply_wiki_boost(reranked, boost=wiki_boost)

    result.hits = reranked
    return result


def enrich_faq_candidates(session: Session, faq_hits: list[Candidate]) -> list[Candidate]:
    """FAQ 候选内容回填 Q/A（向量索引 metadata 无 answer，需回 DB 取）。"""
    if not faq_hits:
        return []
    rows = session.query(Chunk).filter(Chunk.id.in_([h.chunk_id for h in faq_hits])).all()
    by_id = {c.id: c for c in rows}
    out: list[Candidate] = []
    for h in faq_hits:
        ck = by_id.get(h.chunk_id)
        if ck is None:
            continue
        meta = ck.metadata_ or {}
        h.content = f"Q: {meta.get('standard_question', ck.content)}\nA: {meta.get('answer', '')}"
        out.append(h)
    return out
