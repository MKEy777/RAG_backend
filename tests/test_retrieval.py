from __future__ import annotations

from pyrag.retrieval.fusion import rrf_fusion
from pyrag.retrieval.hybrid import Candidate


def _make_candidate(cid: str, score: float, channel: str) -> Candidate:
    return Candidate(chunk_id=cid, score=score, channel=channel)


def test_rrf_both_channels():
    vec = [_make_candidate("a", 0.9, "vector"), _make_candidate("b", 0.7, "vector")]
    kw = [_make_candidate("b", 5.0, "keyword"), _make_candidate("c", 3.0, "keyword")]
    fused = rrf_fusion(vec, kw, k=60, vector_weight=0.7, keyword_weight=0.3)
    assert len(fused) == 3
    ids = [c.chunk_id for c in fused]
    assert "b" in ids
    b = next(c for c in fused if c.chunk_id == "b")
    assert b.channel == "hybrid"
    expected_b = 0.7 / (60 + 2) + 0.3 / (60 + 1)
    assert abs(b.score - expected_b) < 1e-6


def test_rrf_vector_only():
    vec = [_make_candidate("x", 0.8, "vector"), _make_candidate("y", 0.5, "vector")]
    fused = rrf_fusion(vec, [], k=60)
    assert len(fused) == 2
    assert fused[0].chunk_id == "x"


def test_rrf_empty():
    fused = rrf_fusion([], [])
    assert fused == []


def test_rrf_dedup():
    vec = [_make_candidate("a", 0.9, "vector"), _make_candidate("a", 0.5, "vector")]
    fused = rrf_fusion(vec, [])
    assert len(fused) == 1
    assert fused[0].score == 0.9
