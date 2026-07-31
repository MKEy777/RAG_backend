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
from ..settings import settings
from .dedup import dedup_candidates
from .expansion import expand_queries
from .fusion import rrf_fusion
from .graph_boost import graph_boost
from .hybrid import Candidate, hybrid_search
from .rerank_stage import apply_authority_boost, apply_wiki_boost, rerank_stage

log = get_logger("retrieval")


@dataclass
class RetrievalResult:
    hits: list[Candidate] = field(default_factory=list)


# 三层解析兜底值：与历史行为一致（ARCHITECTURE.md §4.4）
_CONFIG_DEFAULTS: dict = {
    "search_mode": "hybrid",
    "top_k": 10,
    "enable_graph": True,
    "enable_rerank": True,
    "expand_parent": True,
    "enable_wiki_boost": False,
    "recall_top_k": None,
    "vector_threshold": None,
    "keyword_threshold": None,
    "rrf_k": None,
    "vector_weight": None,
    "keyword_weight": None,
    "rerank_threshold": None,
    "wiki_boost": None,
}


def _resolve_config(session: Session, kb_ids: list[str], overrides: dict, endpoint_defaults: dict | None = None) -> dict:
    kb_cfg: dict = {}
    for kb_id in kb_ids:
        kb = session.get(KnowledgeBase, kb_id)
        if kb and kb.retrieval_config:
            kb_cfg = kb.retrieval_config
            break
    endpoint_defaults = endpoint_defaults or {}
    resolved: dict = {}
    for key, fallback in _CONFIG_DEFAULTS.items():
        if overrides.get(key) is not None:
            resolved[key] = overrides[key]
        elif kb_cfg.get(key) is not None:
            resolved[key] = kb_cfg[key]
        elif key in endpoint_defaults:
            resolved[key] = endpoint_defaults[key]
        else:
            resolved[key] = fallback
    return resolved


async def retrieve_candidates(
    session: Session,
    kb_ids: list[str],
    query: str,
    top_k: int | None = None,
    search_mode: str | None = None,
    enable_graph: bool | None = None,
    enable_rerank: bool | None = None,
    expand_parent: bool | None = None,
    enable_wiki_boost: bool | None = None,
    recall_top_k: int | None = None,
    vector_threshold: float | None = None,
    keyword_threshold: float | None = None,
    rrf_k: int | None = None,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
    rerank_threshold: float | None = None,
    wiki_boost: float | None = None,
    endpoint_defaults: dict | None = None,
) -> RetrievalResult:
    cfg = _resolve_config(
        session, kb_ids,
        {
            "search_mode": search_mode,
            "top_k": top_k,
            "enable_graph": enable_graph,
            "enable_rerank": enable_rerank,
            "expand_parent": expand_parent,
            "enable_wiki_boost": enable_wiki_boost,
            "recall_top_k": recall_top_k,
            "vector_threshold": vector_threshold,
            "keyword_threshold": keyword_threshold,
            "rrf_k": rrf_k,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "rerank_threshold": rerank_threshold,
            "wiki_boost": wiki_boost,
        },
        endpoint_defaults,
    )
    mode = cfg["search_mode"]
    top_k = cfg["top_k"]
    recall_top_k = cfg["recall_top_k"]
    vector_threshold = cfg["vector_threshold"]
    keyword_threshold = cfg["keyword_threshold"]
    rrf_k = cfg["rrf_k"]
    vector_weight = cfg["vector_weight"]
    keyword_weight = cfg["keyword_weight"]
    rerank_threshold = cfg["rerank_threshold"]
    wiki_boost = cfg["wiki_boost"]

    query_vector: np.ndarray | None = None
    if mode != "fulltext":
        try:
            vectors = await embed_texts([query])
            query_vector = vectors[0]
        except Exception as e:
            log.warning("嵌入服务不可用_回退BM25", error=str(e))

    doc_kb_ids = list(kb_ids)

    result = RetrievalResult()

    pairs = await asyncio.gather(
        *[
            hybrid_search(
                kb_id, query, query_vector,
                top_k=recall_top_k,
                vector_threshold=vector_threshold,
                keyword_threshold=keyword_threshold,
                mode=mode,
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

    if cfg["enable_graph"] and doc_kb_ids:
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

    # vector 模式下低召回扩展是纯关键词通道，跳过以保持模式语义
    if doc_kb_ids and len(fused) < top_k and mode != "vector":
        variants = expand_queries(query)
        if variants:
            kw_th = (keyword_threshold if keyword_threshold is not None else settings.keyword_threshold) * 0.8
            exp = await asyncio.gather(*[
                hybrid_search(kb_id, v, None, top_k=recall_top_k, keyword_threshold=kw_th)
                for v in variants for kb_id in doc_kb_ids
            ], return_exceptions=True)
            seen_ids = {c.chunk_id for c in fused}
            added = 0
            for pair in exp:
                if isinstance(pair, Exception):
                    continue
                for c in pair[0] + pair[1]:
                    if c.chunk_id not in seen_ids:
                        seen_ids.add(c.chunk_id)
                        fused.append(c)
                        added += 1
            if added:
                log.info("低召回扩展补充", variants=len(variants), added=added)

    if cfg["expand_parent"]:
        fused = expand_parent_chunks(session, fused)

    fused = dedup_candidates(fused)

    if cfg["enable_rerank"]:
        reranked = await rerank_stage(query, fused, top_k=top_k, threshold=rerank_threshold)
    else:
        reranked = fused[:top_k]

    if cfg["enable_wiki_boost"]:
        reranked = apply_wiki_boost(reranked, boost=wiki_boost)

    reranked = apply_authority_boost(session, reranked)

    result.hits = reranked
    return result


