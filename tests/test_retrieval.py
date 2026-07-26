from __future__ import annotations

import numpy as np

from knowlink.retrieval.fusion import rrf_fusion
from knowlink.retrieval.hybrid import Candidate
from knowlink.storage.keyword_bm25s import BM25KeywordIndex
from knowlink.storage.vector_local import LocalVectorStore


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


async def test_vector_delete_document_removes_hits(tmp_path):
    store = LocalVectorStore(root=tmp_path)
    vecs = np.eye(3, 8, dtype=np.float32)
    await store.upsert(
        "kb1",
        ["c1", "c2", "c3"],
        vecs,
        [{"knowledge_id": "docA"}, {"knowledge_id": "docA"}, {"knowledge_id": "docB"}],
    )
    await store.delete_document("kb1", "docA")
    hits = await store.search("kb1", vecs[0], top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c3"}


async def test_vector_delete_document_survives_restart(tmp_path):
    store = LocalVectorStore(root=tmp_path)
    vecs = np.eye(2, 8, dtype=np.float32)
    await store.upsert(
        "kb1", ["c1", "c2"], vecs,
        [{"knowledge_id": "docA"}, {"knowledge_id": "docB"}],
    )
    # 新实例模拟进程重启：meta 必须落盘才能按 knowledge_id 删除
    store2 = LocalVectorStore(root=tmp_path)
    await store2.delete_document("kb1", "docA")
    hits = await store2.search("kb1", vecs[0], top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c2"}
    assert hits[0]["metadata"]["knowledge_id"] == "docB"


async def test_keyword_delete_document_removes_hits(tmp_path):
    index = BM25KeywordIndex(root=tmp_path)
    await index.index_chunks(
        "kb1",
        ["c1", "c2"],
        ["深度学习模型训练", "深度学习推理加速"],
        [{"knowledge_id": "docA"}, {"knowledge_id": "docB"}],
    )
    await index.delete_document("kb1", "docA")
    hits = await index.search("kb1", "深度学习", top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c2"}


async def test_keyword_search_survives_restart(tmp_path):
    index = BM25KeywordIndex(root=tmp_path)
    await index.index_chunks(
        "kb1",
        ["c1", "c2"],
        ["向量检索系统", "关键词检索系统"],
        [{"knowledge_id": "docA"}, {"knowledge_id": "docB"}],
    )
    index2 = BM25KeywordIndex(root=tmp_path)
    hits = await index2.search("kb1", "检索", top_k=10)
    assert len(hits) == 2
    await index2.delete_document("kb1", "docA")
    hits = await index2.search("kb1", "检索", top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c2"}


async def test_vector_reconcile_removes_stale_and_backfills_meta(tmp_path):
    store = LocalVectorStore(root=tmp_path)
    vecs = np.eye(3, 8, dtype=np.float32)
    # 模拟旧版本：meta 未落盘（直接清空内存 meta 后落盘）
    await store.upsert("kb1", ["c1", "c2", "stale"], vecs, [{}, {}, {}])
    live_meta = {
        "c1": {"knowledge_id": "docA", "chunk_type": "text"},
        "c2": {"knowledge_id": "docB", "chunk_type": "text"},
    }
    removed = await store.reconcile("kb1", live_meta)
    assert removed == 1
    hits = await store.search("kb1", vecs[0], top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c1", "c2"}
    assert all(h["metadata"].get("knowledge_id") for h in hits)
    # 对账后按文档删除可用
    await store.reconcile("kb1", live_meta)
    await store.delete_document("kb1", "docA")
    hits = await store.search("kb1", vecs[0], top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c2"}


async def test_keyword_reconcile_rebuilds_legacy_corpus(tmp_path):
    index = BM25KeywordIndex(root=tmp_path)
    await index.index_chunks(
        "kb1", ["c1", "stale"],
        ["深度学习模型", "已删除文档内容"],
        [{"knowledge_id": "docA"}, {"knowledge_id": "dead"}],
    )
    await index.search("kb1", "模型", top_k=5)
    # 模拟旧版本：只有 index 目录、无 corpus.json
    (tmp_path / "kb1" / "corpus.json").unlink()
    (tmp_path / "kb1" / "indexed_ids.json").unlink()

    index2 = BM25KeywordIndex(root=tmp_path)
    live_texts = {"c1": "深度学习模型", "c3": "新增存活块"}
    live_meta = {"c1": {"knowledge_id": "docA"}, "c3": {"knowledge_id": "docB"}}
    removed, added = await index2.reconcile("kb1", live_texts, live_meta)
    assert removed == 0 and added == 2
    hits = await index2.search("kb1", "深度学习", top_k=10, threshold=0.1)
    assert {h["chunk_id"] for h in hits} == {"c1"}
    hits = await index2.search("kb1", "存活", top_k=10, threshold=0.1)
    assert {h["chunk_id"] for h in hits} == {"c3"}


async def test_reconcile_kb_indexes_uses_db_truth(tmp_path, db_session, monkeypatch):
    from dataclasses import dataclass

    from knowlink.indexing import reconcile as reconcile_mod
    from knowlink.models.chunk import Chunk
    from knowlink.models.knowledge import Knowledge

    live_doc = Knowledge(id="docLive", tenant_id=1, knowledge_base_id="kb1", type="file")
    dead_doc = Knowledge(id="docDead", tenant_id=1, knowledge_base_id="kb1", type="file")
    db_session.add_all([live_doc, dead_doc])
    db_session.add_all([
        Chunk(id="c1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="docLive",
              content="存活内容", chunk_type="text", metadata_={"title_path": ["A"], "doc_title": "T"}),
        Chunk(id="c2", tenant_id=1, knowledge_base_id="kb1", knowledge_id="docDead",
              content="死文档内容", chunk_type="text", metadata_={}),
        Chunk(id="p1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="docLive",
              content="父块", chunk_type="parent_text", metadata_={}),
    ])
    from datetime import datetime, timezone
    dead_doc.deleted_at = datetime.now(timezone.utc)
    db_session.commit()

    vector = LocalVectorStore(root=tmp_path / "vec")
    keyword = BM25KeywordIndex(root=tmp_path / "kw")
    vecs = np.eye(2, 8, dtype=np.float32)
    await vector.upsert("kb1", ["c1", "c2"], vecs, [{}, {}])
    await keyword.index_chunks("kb1", ["c1", "c2"], ["存活内容", "死文档内容"], [{}, {}])

    @dataclass
    class Bundle:
        vector: object
        keyword: object

    monkeypatch.setattr(reconcile_mod, "get_storage", lambda: Bundle(vector=vector, keyword=keyword))
    result = await reconcile_mod.reconcile_kb_indexes(db_session, "kb1")
    assert result["live"] == 1
    assert result["vector_removed"] == 1
    assert result["keyword_removed"] == 1

    hits = await vector.search("kb1", vecs[0], top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c1"}
    hits = await keyword.search("kb1", "内容", top_k=10)
    assert {h["chunk_id"] for h in hits} == {"c1"}
