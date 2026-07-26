"""Golden regression tests: fixed query → expected retrieval behavior.

These tests verify that the RRF fusion, chunking, and retrieval pipeline
produces stable, deterministic results for known inputs without a real LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from knowlink.models.base import Base
from knowlink.chunking import chunk_document
from knowlink.chunking.parent_child import ParentChildPair
from knowlink.retrieval.fusion import rrf_fusion
from knowlink.retrieval.hybrid import Candidate


GOLDEN_DOC = """# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Supervised Learning

Supervised learning uses labeled training data. Common algorithms include linear regression, decision trees, and neural networks.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are typical tasks.

## Reinforcement Learning

Reinforcement learning trains agents through reward signals. The agent learns optimal actions through trial and error.
"""


class TestChunkingGolden:
    def test_heading_strategy_selected(self):
        result = chunk_document(GOLDEN_DOC, strategy="auto", enable_parent_child=True)
        assert len(result) > 0
        assert isinstance(result[0], ParentChildPair)

    def test_parent_child_structure(self):
        result = chunk_document(GOLDEN_DOC, strategy="auto", enable_parent_child=True)
        for pair in result:
            assert pair.parent_text
            assert len(pair.children) > 0
            for child in pair.children:
                assert child.text
                assert len(child.text) <= 512

    def test_title_path_preserved(self):
        result = chunk_document(GOLDEN_DOC, strategy="auto", enable_parent_child=True)
        paths_found = [pair.parent_title_path for pair in result if pair.parent_title_path]
        assert len(paths_found) > 0
        has_section = any("Supervised Learning" in " > ".join(p) for p in paths_found)
        assert has_section

    def test_flat_mode(self):
        result = chunk_document(GOLDEN_DOC, strategy="heading", enable_parent_child=False)
        assert len(result) > 0
        assert not isinstance(result[0], ParentChildPair)


class TestRRFGolden:
    def test_rrf_ranking_stability(self):
        vec = [
            Candidate(chunk_id="a", score=0.95, channel="vector"),
            Candidate(chunk_id="b", score=0.80, channel="vector"),
            Candidate(chunk_id="c", score=0.70, channel="vector"),
        ]
        kw = [
            Candidate(chunk_id="b", score=12.0, channel="keyword"),
            Candidate(chunk_id="d", score=8.0, channel="keyword"),
            Candidate(chunk_id="a", score=5.0, channel="keyword"),
        ]
        fused = rrf_fusion(vec, kw, k=60, vector_weight=0.7, keyword_weight=0.3)

        ids = [c.chunk_id for c in fused]
        assert "a" in ids
        assert "b" in ids
        assert len(fused) == 4

        # b appears in both channels at good ranks → should score high
        b_candidate = next(c for c in fused if c.chunk_id == "b")
        assert b_candidate.channel == "hybrid"
        expected_b = 0.7 / (60 + 2) + 0.3 / (60 + 1)
        assert abs(b_candidate.score - expected_b) < 1e-9

    def test_rrf_single_channel_passthrough(self):
        vec = [
            Candidate(chunk_id="x", score=0.9, channel="vector"),
            Candidate(chunk_id="y", score=0.5, channel="vector"),
        ]
        fused = rrf_fusion(vec, [])
        assert len(fused) == 2
        assert fused[0].chunk_id == "x"
        assert fused[0].score == 0.9

    def test_rrf_empty_inputs(self):
        assert rrf_fusion([], []) == []

    def test_rrf_dedup_same_channel(self):
        vec = [
            Candidate(chunk_id="a", score=0.9, channel="vector"),
            Candidate(chunk_id="a", score=0.4, channel="vector"),
        ]
        fused = rrf_fusion(vec, [])
        assert len(fused) == 1
        assert fused[0].score == 0.9
