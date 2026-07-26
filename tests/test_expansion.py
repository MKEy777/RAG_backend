from __future__ import annotations

from knowlink.retrieval.expansion import expand_queries


def test_question_words_removed():
    out = expand_queries("什么是脑电信号的情绪识别方法？")
    assert out
    assert any("脑电信号" in v and "什么是" not in v for v in out)


def test_short_query_no_expansion():
    assert expand_queries("脑电") == []


def test_dedup_and_limit():
    out = expand_queries("为什么睡眠舱干预方法采用生物反馈，而不是药物干预，优势哪些？")
    assert len(out) <= 5
    assert len(set(v.lower() for v in out)) == len(out)
