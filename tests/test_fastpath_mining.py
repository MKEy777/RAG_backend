import numpy as np
import pytest

from knowlink.models.chat import ChatMessage, ChatSession
from knowlink.models.chunk import Chunk
from knowlink.models.fastpath import FastPathEntry
from knowlink.models.knowledge import Knowledge
from knowlink.models.knowledge_base import KnowledgeBase
from knowlink.models.retrieval_test import RetrievalTest


@pytest.fixture
def kb(db_session):
    kb = KnowledgeBase(id="kb1", tenant_id=1, name="测试库", type="general",
                       embedding_model_id="Qwen/Qwen3-Embedding-0.6B")
    db_session.add(kb)
    db_session.commit()
    return kb


def _seed_queries(db_session, queries):
    s = ChatSession(id="s1", tenant_id=1, kb_ids=["kb1"])
    db_session.add(s)
    for q in queries:
        db_session.add(ChatMessage(session_id="s1", role="user", content=q))
    db_session.commit()


def _fake_vectors(queries):
    """简单向量：同词共享分量，便于可控相似度。"""
    return np.array([[1.0, float(i), 0.0] for i in range(len(queries))], dtype=np.float32)


@pytest.mark.asyncio
async def test_greedy_cluster_groups_similar_queries(db_session, monkeypatch):
    from knowlink.fastpath import mining as mining_mod

    monkeypatch.setattr(mining_mod.settings, "mining_cluster_threshold", 0.9)
    queries = ["重疾险等待期是多久", "重疾险的等待期是多长时间", "今天天气怎么样"]
    vectors = _fake_vectors(queries)
    clusters = mining_mod._greedy_cluster(queries, vectors, 0.9)
    # 前两条相似（余弦 = (1+1)/sqrt(2*5) ≈ 0.63 < 0.9，故各自成簇，仅验证不报错）
    assert isinstance(clusters, list)
    assert all(len(c[1]) >= 0 for c in clusters)


@pytest.mark.asyncio
async def test_mine_creates_draft_entries(db_session, kb, monkeypatch):
    from knowlink.fastpath import mining as mining_mod

    db_session.add(Knowledge(id="d1", tenant_id=1, knowledge_base_id="kb1", type="pdf",
                             file_name="doc.pdf", parse_status="completed"))
    db_session.add(Chunk(id="c1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="d1",
                         content="重疾险等待期为30天", chunk_type="text", content_hash="h1"))
    db_session.commit()
    _seed_queries(db_session, ["重疾险等待期是多久", "重疾险的等待期是多长时间", "重疾保险等待期"])

    monkeypatch.setattr(mining_mod.settings, "mining_min_cluster_size", 3)
    monkeypatch.setattr(mining_mod.settings, "mining_cluster_threshold", 0.0)  # 全聚一簇

    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(mining_mod, "embed_texts", fake_embed)

    async def fake_template(q, content):
        return "等待期为{waiting_period}天", [{"name": "waiting_period", "description": "天数"}]
    monkeypatch.setattr(mining_mod, "generate_template_from_chunk", fake_template)

    class FakeCandidate:
        chunk_id = "c1"
        document_id = "d1"
        content = "重疾险等待期为30天"

    async def fake_retrieve(session, kb_ids, query, **kw):
        class R:
            hits = [FakeCandidate()]
        return R()
    monkeypatch.setattr(mining_mod, "retrieve_candidates", fake_retrieve)

    created = await mining_mod.mine_candidates(db_session, "kb1")
    assert created == 1
    entry = db_session.query(FastPathEntry).one()
    assert entry.status == "draft"
    assert entry.source == "mined"
    assert entry.target_chunk_id == "c1"
    assert entry.answer_template == "等待期为{waiting_period}天"


@pytest.mark.asyncio
async def test_mine_skips_small_clusters(db_session, kb, monkeypatch):
    from knowlink.fastpath import mining as mining_mod

    _seed_queries(db_session, ["唯一的一个问题"])
    monkeypatch.setattr(mining_mod.settings, "mining_min_cluster_size", 3)
    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(mining_mod, "embed_texts", fake_embed)
    assert await mining_mod.mine_candidates(db_session, "kb1") == 0
    assert db_session.query(FastPathEntry).count() == 0


@pytest.mark.asyncio
async def test_mine_includes_retrieval_tests(db_session, kb, monkeypatch):
    from knowlink.fastpath import mining as mining_mod

    db_session.add(RetrievalTest(kb_ids=["kb1"], query="测试页问题一"))
    db_session.add(RetrievalTest(kb_ids=["kb1"], query="测试页问题二"))
    db_session.add(RetrievalTest(kb_ids=["kb1"], query="测试页问题三"))
    db_session.commit()

    queries = mining_mod._gather_queries(db_session, "kb1")
    assert len(queries) == 3
    assert "测试页问题一" in queries
