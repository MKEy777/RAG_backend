from __future__ import annotations

from knowlink.retrieval.dedup import dedup_candidates
from knowlink.retrieval.hybrid import Candidate


def _c(cid, content, score):
    return Candidate(chunk_id=cid, score=score, channel="vector", chunk_type="text", content=content)


def test_exact_signature_dedup_keeps_first():
    a = _c("a", "深度学习是机器学习的分支。", 0.9)
    b = _c("b", "深度学习是机器学习的分支。", 0.6)
    assert [c.chunk_id for c in dedup_candidates([a, b])] == ["a"]


def test_containment_drops_lower_score():
    long = _c("long", "检索增强生成结合了向量检索与大模型生成能力，显著提升问答质量。", 0.5)
    short = _c("short", "检索增强生成结合了向量检索", 0.3)
    assert [c.chunk_id for c in dedup_candidates([long, short])] == ["long"]


def test_distinct_content_kept():
    a = _c("a", "苹果公司发布新品。", 0.9)
    b = _c("b", "谷歌发布搜索更新。", 0.8)
    assert len(dedup_candidates([a, b])) == 2


def test_empty_content_never_merged():
    assert len(dedup_candidates([_c("a", "", 0.9), _c("b", "", 0.8)])) == 2


def test_faq_and_wiki_skipped():
    a = Candidate(chunk_id="f1", score=0.9, channel="faq", chunk_type="faq", content="同一句话")
    b = Candidate(chunk_id="f2", score=0.8, channel="faq", chunk_type="faq", content="同一句话")
    assert len(dedup_candidates([a, b])) == 2
