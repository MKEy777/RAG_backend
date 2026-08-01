from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..indexing.embedder import embed_texts
from ..logging import get_logger
from ..models.chat import ChatMessage, ChatSession
from ..models.fastpath import FastPathEntry
from ..models.retrieval_test import RetrievalTest
from ..retrieval.pipeline import retrieve_candidates
from ..settings import settings
from .filler import generate_template_from_chunk

log = get_logger("fastpath")


def _gather_queries(session: Session, kb_id: str) -> list[str]:
    """历史真实问句：chat_messages(user) + retrieval_tests，按 KB 过滤。"""
    queries: list[str] = []

    session_ids = set(
        session.execute(
            select(ChatSession.id).where(ChatSession.kb_ids.contains(kb_id))
        ).scalars()
    )
    if session_ids:
        rows = (
            session.execute(
                select(ChatMessage.content).where(
                    ChatMessage.session_id.in_(session_ids),
                    ChatMessage.role == "user",
                )
            )
            .scalars()
            .all()
        )
        queries.extend(rows)

    # retrieval_tests 的 kb_ids 是 JSON，SQLite JSON contains 走 Python 过滤
    tests = session.execute(select(RetrievalTest)).scalars().all()
    queries.extend(t.query for t in tests if kb_id in (t.kb_ids or []))

    seen: set[str] = set()
    cleaned: list[str] = []
    for q in queries:
        q = (q or "").strip()
        if q and q not in seen:
            seen.add(q)
            cleaned.append(q)
    return cleaned


def _greedy_cluster(
    queries: list[str], vectors: np.ndarray, threshold: float
) -> list[tuple[str, list[str]]]:
    """贪心聚类：按出现顺序取代表点，相似度 ≥ 阈值的归入同簇。"""
    norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    assigned: list[bool] = [False] * len(queries)
    clusters: list[tuple[str, list[str]]] = []
    for i in range(len(queries)):
        if assigned[i]:
            continue
        members = [i]
        assigned[i] = True
        for j in range(i + 1, len(queries)):
            if assigned[j]:
                continue
            sim = float(norms[i] @ norms[j])
            if sim >= threshold:
                members.append(j)
                assigned[j] = True
        centroid = norms[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-10
        rep_idx = min(members, key=lambda m: -float(norms[m] @ centroid))
        rep = queries[rep_idx]
        examples = [queries[m] for m in members if m != rep_idx]
        clusters.append((rep, examples))
    return clusters


async def mine_candidates(session: Session, kb_id: str) -> int:
    """离线挖掘：历史问句聚类 → 完整链路定位 → LLM 生成模板 → 入库为 draft。"""
    queries = _gather_queries(session, kb_id)
    if len(queries) < settings.mining_min_cluster_size:
        log.info("快路径挖掘_问句不足", kb_id=kb_id, count=len(queries))
        return 0

    vectors = await embed_texts(queries)
    clusters = _greedy_cluster(queries, vectors, settings.mining_cluster_threshold)
    created = 0
    for rep, examples in clusters:
        if len(examples) + 1 < settings.mining_min_cluster_size:
            continue
        try:
            result = await retrieve_candidates(session, [kb_id], rep, top_k=3)
        except Exception as e:
            log.warning("快路径挖掘_定位失败", query=rep[:40], error=str(e))
            continue
        if not result.hits:
            continue
        top = result.hits[0]
        if not top.content:
            continue
        template, variables = await generate_template_from_chunk(rep, top.content)
        if not template:
            continue
        entry = FastPathEntry(
            kb_id=kb_id,
            canonical_query=rep,
            example_queries=examples[:10],
            answer_template=template,
            variables=variables,
            target_chunk_id=top.chunk_id,
            target_document_id=top.document_id,
            status="draft",
            source="mined",
        )
        session.add(entry)
        created += 1
    session.commit()
    log.info("快路径挖掘完成", kb_id=kb_id, created=created)
    return created
