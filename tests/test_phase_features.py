from __future__ import annotations

import numpy as np
import pytest

from knowlink.models.chunk import Chunk
from knowlink.models.knowledge import Knowledge
from knowlink.models.knowledge_base import KnowledgeBase


def _kb(id="kb1", **kw):
    return KnowledgeBase(id=id, tenant_id=1, name=id, type="document", embedding_model_id="m1", **kw)


def _doc(id="doc1", kb_id="kb1", **kw):
    kw.setdefault("parse_status", "completed")
    kw.setdefault("title", id)
    return Knowledge(id=id, tenant_id=1, knowledge_base_id=kb_id, type="file", **kw)


def _chunk(id, kb_id="kb1", doc_id="doc1", **kw):
    return Chunk(id=id, tenant_id=1, knowledge_base_id=kb_id, knowledge_id=doc_id, content=f"内容{id}", chunk_type="text", **kw)


# ---------- 阶段1：列表筛选/统计 ----------

def test_kb_list_keyword_group_and_stats(db_session):
    from knowlink.api.routers.knowledge_bases import list_kbs

    db_session.add(_kb("kb1", group_name="法务", description="合同库"))
    db_session.add(_kb("kb2", group_name="医疗"))
    db_session.add(_doc("d1", "kb1", file_size=100))
    db_session.add(_doc("d2", "kb1", file_size=200))
    db_session.add(_chunk("c1"))
    db_session.commit()

    out = list_kbs(keyword="合同", session=db_session)
    assert [i.id for i in out["items"]] == ["kb1"]
    assert out["items"][0].doc_count == 2
    assert out["items"][0].total_size == 300
    assert out["items"][0].chunk_count == 1

    out = list_kbs(group="医疗", session=db_session)
    assert [i.id for i in out["items"]] == ["kb2"]


def test_document_list_filters(db_session):
    from knowlink.api.routers.documents import list_documents

    db_session.add(_kb())
    db_session.add(_doc("d1", tags=["合同", "重要"], parse_status="completed"))
    db_session.add(_doc("d2", title="应急预案", parse_status="failed"))
    db_session.commit()

    out = list_documents("kb1", tag="合同", session=db_session)
    assert [i.id for i in out["items"]] == ["d1"]
    out = list_documents("kb1", status="failed", session=db_session)
    assert [i.id for i in out["items"]] == ["d2"]
    out = list_documents("kb1", keyword="应急", session=db_session)
    assert [i.id for i in out["items"]] == ["d2"]


# ---------- 阶段2：三层配置解析 / search_mode / chunk 启停 ----------

def test_resolve_config_three_layers(db_session):
    from knowlink.retrieval.pipeline import _resolve_config

    db_session.add(_kb("kb1", retrieval_config={"search_mode": "vector", "top_k": 5, "enable_rerank": False}))
    db_session.commit()

    # KB 配置生效
    cfg = _resolve_config(db_session, ["kb1"], {})
    assert cfg["search_mode"] == "vector" and cfg["top_k"] == 5 and cfg["enable_rerank"] is False

    # 请求覆写优先
    cfg = _resolve_config(db_session, ["kb1"], {"search_mode": "fulltext", "top_k": 3})
    assert cfg["search_mode"] == "fulltext" and cfg["top_k"] == 3

    # NULL 配置回落默认（旧 KB 行为不变）
    db_session.add(_kb("kb2"))
    db_session.commit()
    cfg = _resolve_config(db_session, ["kb2"], {})
    assert cfg["search_mode"] == "hybrid" and cfg["top_k"] == 10 and cfg["enable_rerank"] is True

    # 端点默认仅在无覆写且无 KB 配置时生效
    cfg = _resolve_config(db_session, ["kb2"], {}, {"enable_graph": False})
    assert cfg["enable_graph"] is False


async def test_search_mode_vector_skips_keyword_channel(monkeypatch):
    from knowlink.retrieval import hybrid as h

    calls = {"vector": 0, "keyword": 0}

    class FakeVec:
        async def search(self, kb_id, qv, top_k, threshold):
            calls["vector"] += 1
            return [{"chunk_id": "v1", "score": 0.9, "metadata": {}}]

    class FakeKw:
        async def search(self, kb_id, q, top_k, threshold):
            calls["keyword"] += 1
            return [{"chunk_id": "k1", "score": 0.8, "metadata": {}}]

    class FakeStorage:
        vector = FakeVec()
        keyword = FakeKw()

    monkeypatch.setattr(h, "get_storage", lambda: FakeStorage())

    qv = np.zeros(4, dtype=np.float32)
    vec, kw = await h.hybrid_search("kb1", "q", qv, mode="vector")
    assert calls == {"vector": 1, "keyword": 0}
    assert [c.chunk_id for c in vec] == ["v1"] and kw == []

    vec, kw = await h.hybrid_search("kb1", "q", qv, mode="fulltext")
    assert calls == {"vector": 1, "keyword": 1}
    assert vec == [] and [c.chunk_id for c in kw] == ["k1"]

    vec, kw = await h.hybrid_search("kb1", "q", qv, mode="hybrid")
    assert calls == {"vector": 2, "keyword": 2}


def test_disabled_chunk_excluded_from_live_sets(db_session):
    from knowlink.indexing.reconcile import _live_index_entries, live_text_chunk_ids

    db_session.add(_kb())
    db_session.add(_doc())
    db_session.add(_chunk("c-on"))
    db_session.add(_chunk("c-off", enabled=False))
    db_session.commit()

    assert live_text_chunk_ids(db_session, "kb1") == {"c-on"}
    live_texts, _ = _live_index_entries(db_session, "kb1")
    assert set(live_texts) == {"c-on"}


def test_disabled_chunk_skipped_in_parent_expand(db_session):
    from knowlink.generation.context import expand_parent_chunks
    from knowlink.retrieval.hybrid import Candidate

    db_session.add(_kb())
    db_session.add(_doc())
    db_session.add(_chunk("c-on"))
    db_session.add(_chunk("c-off", enabled=False))
    db_session.commit()

    cands = [
        Candidate(chunk_id="c-on", score=1.0, channel="vector"),
        Candidate(chunk_id="c-off", score=0.9, channel="vector"),
    ]
    out = expand_parent_chunks(db_session, cands)
    assert [c.chunk_id for c in out] == ["c-on"]


def test_authority_boost(db_session):
    from knowlink.retrieval.hybrid import Candidate
    from knowlink.retrieval.rerank_stage import apply_authority_boost

    db_session.add(_kb())
    db_session.add(_doc("d-auth", authority=1))
    db_session.add(_doc("d-plain"))
    db_session.commit()

    cands = [
        Candidate(chunk_id="c1", score=0.5, channel="vector", document_id="d-plain"),
        Candidate(chunk_id="c2", score=0.5, channel="vector", document_id="d-auth"),
    ]
    out = apply_authority_boost(db_session, cands)
    assert out[0].chunk_id == "c2"
    assert out[0].score == pytest.approx(0.6)


# ---------- 阶段3：自定义分隔符 ----------

def test_custom_separators_chunking():
    from knowlink.chunking import chunk_document
    from knowlink.chunking.recursive import split_recursive

    # 直测：自定义分隔符生效
    spans = split_recursive("第一段◆第二段◆第三段", chunk_size=6, overlap=0, separators=["◆"])
    assert [s.text for s in spans] == ["第一段", "第二段", "第三段"]

    # 贯通：chunk_document 透传 separators（段长 ≥50 避免 validator 小块回退）
    p1, p2 = "甲" * 60, "乙" * 60
    chunks = chunk_document(
        f"{p1}◆{p2}", strategy="recursive", chunk_size=80, overlap=0,
        enable_parent_child=False, separators=["◆"],
    )
    assert [c.text for c in chunks] == [p1, p2]
