from __future__ import annotations

from knowlink.retrieval import rerank_stage as rs
from knowlink.retrieval.hybrid import Candidate


def _c(cid, score):
    return Candidate(chunk_id=cid, score=score, channel="vector", chunk_type="text", content=f"内容{cid}")


async def test_composite_blends_base_and_model(monkeypatch):
    class FakeReranker:
        async def rerank(self, query, documents, top_k=None):
            return [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.8}]

    monkeypatch.setattr(rs, "get_reranker", lambda: FakeReranker())
    cands = [_c("a", 0.2), _c("b", 0.95)]
    out = await rs.rerank_stage("q", cands, top_k=2, threshold=0.0)
    scores = {c.chunk_id: c.score for c in out}
    assert scores["a"] == round(0.8 * 0.9 + 0.2 * (0.2 / 0.95), 6)
    assert scores["b"] == round(0.8 * 0.8 + 0.2 * (0.95 / 0.95), 6)


async def test_rerank_api_failure_falls_back(monkeypatch):
    class FailReranker:
        async def rerank(self, query, documents, top_k=None):
            raise RuntimeError("api down")

    monkeypatch.setattr(rs, "get_reranker", lambda: FailReranker())
    out = await rs.rerank_stage("q", [_c("a", 0.5), _c("b", 0.3)], top_k=1, threshold=0.0)
    assert [c.chunk_id for c in out] == ["a"]
