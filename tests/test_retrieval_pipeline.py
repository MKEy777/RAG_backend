from __future__ import annotations

import numpy as np
import pytest

from knowlink.models.knowledge_base import KnowledgeBase
from knowlink.retrieval.hybrid import Candidate


def _kb(id, type="document"):
    return KnowledgeBase(id=id, tenant_id=1, name=id, type=type, embedding_model_id="m")


def _cand(cid, score=0.5, channel="vector", chunk_type="text", content=""):
    return Candidate(chunk_id=cid, score=score, channel=channel, chunk_type=chunk_type, content=content)


@pytest.fixture
def patched(monkeypatch):
    from knowlink.retrieval import pipeline as rp

    calls = {"hybrid": [], "faq": [], "graph": []}

    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    async def fake_hybrid(kb_id, query, qv, top_k=None, vector_threshold=None, keyword_threshold=None):
        calls["hybrid"].append((kb_id, query))
        return [_cand(f"{kb_id}-v")], [_cand(f"{kb_id}-k", channel="keyword")]

    async def fake_faq(kb_id, query, qv, top_k, tag_priority):
        calls["faq"].append(kb_id)
        return [_cand(f"{kb_id}-faq", channel="faq", chunk_type="faq")]

    async def fake_graph(kb_id, query, existing_ids):
        calls["graph"].append(kb_id)
        return []

    async def fake_rerank(query, cands, top_k=None, threshold=None):
        return cands[:top_k]

    monkeypatch.setattr(rp, "embed_texts", fake_embed)
    monkeypatch.setattr(rp, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(rp, "faq_search", fake_faq)
    monkeypatch.setattr(rp, "graph_boost", fake_graph)
    monkeypatch.setattr(rp, "rerank_stage", fake_rerank)
    monkeypatch.setattr(rp, "expand_parent_chunks", lambda session, c: c)
    return calls


async def test_faq_kb_split_and_doc_concurrent(db_session, patched):
    from knowlink.retrieval.pipeline import retrieve_candidates

    db_session.add(_kb("kb-doc1"))
    db_session.add(_kb("kb-doc2"))
    db_session.add(_kb("kb-faq", type="faq"))
    db_session.commit()

    result = await retrieve_candidates(db_session, ["kb-doc1", "kb-faq", "kb-doc2"], "查询", top_k=10)
    assert patched["faq"] == ["kb-faq"]
    assert {kb for kb, _ in patched["hybrid"]} == {"kb-doc1", "kb-doc2"}
    assert {c.chunk_id for c in result.faq_hits} == {"kb-faq-faq"}
    assert len(result.hits) == 4
    assert set(patched["graph"]) == {"kb-doc1", "kb-doc2"}


async def test_rerank_disabled_truncates(db_session, patched):
    from knowlink.retrieval.pipeline import retrieve_candidates

    db_session.add(_kb("kb1"))
    db_session.commit()
    result = await retrieve_candidates(db_session, ["kb1"], "q", top_k=1, enable_rerank=False, enable_graph=False)
    assert len(result.hits) == 1


async def test_partial_kb_failure_keeps_other_results(db_session, patched, monkeypatch):
    from knowlink.retrieval import pipeline as rp
    from knowlink.retrieval.pipeline import retrieve_candidates

    db_session.add(_kb("kb-ok"))
    db_session.add(_kb("kb-bad"))
    db_session.commit()

    async def flaky_hybrid(kb_id, query, qv, top_k=None, vector_threshold=None, keyword_threshold=None):
        if kb_id == "kb-bad":
            raise RuntimeError("boom")
        return [_cand(f"{kb_id}-v")], [_cand(f"{kb_id}-k", channel="keyword")]

    async def failing_graph(kb_id, query, existing_ids):
        raise RuntimeError("graph down")

    monkeypatch.setattr(rp, "hybrid_search", flaky_hybrid)
    monkeypatch.setattr(rp, "graph_boost", failing_graph)

    result = await retrieve_candidates(db_session, ["kb-ok", "kb-bad"], "q", top_k=10)
    assert result.hits
    assert {c.chunk_id for c in result.hits} == {"kb-ok-v", "kb-ok-k"}
