import numpy as np
import pytest
from pydantic import ValidationError

from knowlink.api.routers import retrieval as r
from knowlink.models.knowledge_base import KnowledgeBase
from knowlink.retrieval import pipeline as rp


def test_request_accepts_override_params():
    body = r.RetrievalRequest(
        kb_ids=["kb1"], query="q",
        recall_top_k=80, vector_threshold=0.2, keyword_threshold=0.4,
        rrf_k=30, vector_weight=0.6, keyword_weight=0.4,
        enable_rerank=False, rerank_threshold=0.5,
        wiki_boost=1.5, expand_parent=False,
    )
    assert body.recall_top_k == 80 and body.enable_rerank is False


def test_request_defaults_are_none_or_enabled():
    body = r.RetrievalRequest(kb_ids=["kb1"], query="q")
    assert body.recall_top_k is None
    assert body.rrf_k is None
    assert body.enable_rerank is True
    assert body.expand_parent is True


def test_request_rejects_out_of_bounds():
    with pytest.raises(ValidationError):
        r.RetrievalRequest(kb_ids=["kb1"], query="q", vector_threshold=1.5)
    with pytest.raises(ValidationError):
        r.RetrievalRequest(kb_ids=["kb1"], query="q", wiki_boost=0.5)


async def test_route_passes_overrides_through(db_session, monkeypatch):
    db_session.add(KnowledgeBase(
        id="kb1", tenant_id=1, name="kb", type="document", embedding_model_id="m1",
    ))
    db_session.commit()

    captured = {}

    async def fake_embed(texts):
        return np.zeros((1, 4), dtype=np.float32)

    async def fake_hybrid(kb_id, query, qv, top_k=None, vector_threshold=None, keyword_threshold=None):
        captured["hybrid"] = (top_k, vector_threshold, keyword_threshold)
        return [], []

    def fake_fusion(vec, kw, k=None, vector_weight=None, keyword_weight=None):
        captured["fusion"] = (k, vector_weight, keyword_weight)
        return []

    async def fake_rerank(query, candidates, top_k=None, threshold=None):
        captured["rerank"] = (top_k, threshold)
        return []

    def fake_expand(session, candidates):
        captured["expand"] = True
        return candidates

    # 路由已委托共享管线，打到 pipeline 模块
    monkeypatch.setattr(rp, "embed_texts", fake_embed)
    monkeypatch.setattr(rp, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(rp, "rrf_fusion", fake_fusion)
    monkeypatch.setattr(rp, "rerank_stage", fake_rerank)
    monkeypatch.setattr(rp, "expand_parent_chunks", fake_expand)

    body = r.RetrievalRequest(
        kb_ids=["kb1"], query="q", top_k=7,
        recall_top_k=80, vector_threshold=0.2, keyword_threshold=0.4,
        rrf_k=30, vector_weight=0.6, keyword_weight=0.4,
        rerank_threshold=0.5,
    )
    await r.retrieve(body, db_session)

    assert captured["hybrid"] == (80, 0.2, 0.4)
    assert captured["fusion"] == (30, 0.6, 0.4)
    assert captured["rerank"] == (7, 0.5)
    assert captured["expand"] is True


async def test_route_skips_rerank_and_expand_when_disabled(db_session, monkeypatch):
    db_session.add(KnowledgeBase(
        id="kb1", tenant_id=1, name="kb", type="document", embedding_model_id="m1",
    ))
    db_session.commit()

    called = {"rerank": False, "expand": False}

    async def fake_embed(texts):
        return np.zeros((1, 4), dtype=np.float32)

    async def fake_hybrid(kb_id, query, qv, top_k=None, vector_threshold=None, keyword_threshold=None):
        return [], []

    def fake_fusion(vec, kw, k=None, vector_weight=None, keyword_weight=None):
        from knowlink.retrieval.hybrid import Candidate
        return [Candidate(chunk_id=f"c{i}", score=1.0 - i * 0.1, channel="vector") for i in range(5)]

    async def fake_rerank(query, candidates, top_k=None, threshold=None):
        called["rerank"] = True
        return candidates

    def fake_expand(session, candidates):
        called["expand"] = True
        return candidates

    monkeypatch.setattr(rp, "embed_texts", fake_embed)
    monkeypatch.setattr(rp, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(rp, "rrf_fusion", fake_fusion)
    monkeypatch.setattr(rp, "rerank_stage", fake_rerank)
    monkeypatch.setattr(rp, "expand_parent_chunks", fake_expand)

    body = r.RetrievalRequest(
        kb_ids=["kb1"], query="q", top_k=3,
        enable_rerank=False, expand_parent=False,
    )
    resp = await r.retrieve(body, db_session)

    assert called["rerank"] is False
    assert called["expand"] is False
    assert len(resp.hits) == 3
