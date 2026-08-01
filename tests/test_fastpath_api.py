import numpy as np
import pytest

from knowlink.models.chunk import Chunk
from knowlink.models.fastpath import FastPathEntry, FastPathHit
from knowlink.models.knowledge import Knowledge
from knowlink.models.knowledge_base import KnowledgeBase


class FakeVector:
    def __init__(self):
        self.upserted = []
        self.deleted = []
        self._ids: set[str] = set()

    async def upsert(self, kb_id, chunk_ids, vectors, metadata):
        self.upserted.append((kb_id, list(chunk_ids)))
        self._ids.update(chunk_ids)

    async def search(self, kb_id, query_vector, top_k, threshold=0.0):
        return []

    async def delete_chunks(self, kb_id, chunk_ids):
        self.deleted.append((kb_id, list(chunk_ids)))
        self._ids.difference_update(chunk_ids)

    async def list_ids(self, kb_id):
        return list(self._ids)


class FakeStorage:
    def __init__(self, vector):
        self.vector = vector


@pytest.fixture
def kb(db_session):
    kb = KnowledgeBase(id="kb1", tenant_id=1, name="测试库", type="general",
                       embedding_model_id="Qwen/Qwen3-Embedding-0.6B")
    db_session.add(kb)
    db_session.commit()
    return kb


def _mk_doc_and_chunks(db_session, doc_id="doc1", chunk_ids=("c1",)):
    db_session.add(Knowledge(id=doc_id, tenant_id=1, knowledge_base_id="kb1", type="pdf",
                             file_name="doc1.pdf", parse_status="completed"))
    for i, cid in enumerate(chunk_ids):
        db_session.add(Chunk(id=cid, tenant_id=1, knowledge_base_id="kb1", knowledge_id=doc_id,
                             content=f"等待期为30天 {i}", chunk_type="text", content_hash=f"h{i}"))
    db_session.commit()


def test_create_entry(db_session, kb):
    from knowlink.api.routers import fastpath as fp_router

    _mk_doc_and_chunks(db_session)
    payload = fp_router.EntryCreate(
        kb_id="kb1", canonical_query="重疾险等待期是多久",
        answer_template="等待期为30天", target_chunk_id="c1", target_document_id="doc1",
    )
    result = fp_router.create_entry(payload, db_session)
    assert result["status"] == "active"
    assert result["source"] == "manual"
    assert db_session.query(FastPathEntry).count() == 1


def test_create_entry_with_bad_chunk_rejected(db_session, kb):
    from fastapi import HTTPException

    from knowlink.api.routers import fastpath as fp_router

    payload = fp_router.EntryCreate(
        kb_id="kb1", canonical_query="q", answer_template="t", target_chunk_id="missing",
    )
    with pytest.raises(HTTPException):
        fp_router.create_entry(payload, db_session)


@pytest.mark.asyncio
async def test_update_entry(db_session, kb, monkeypatch):
    from knowlink.api.routers import fastpath as fp_router

    entry = FastPathEntry(kb_id="kb1", canonical_query="q1", answer_template="t1")
    db_session.add(entry)
    db_session.commit()
    fake_vec = FakeVector()
    monkeypatch.setattr(fp_router, "get_storage", lambda: FakeStorage(fake_vec))
    result = await fp_router.update_entry(entry.id, fp_router.EntryUpdate(status="disabled"), db_session)
    assert result["status"] == "disabled"


@pytest.mark.asyncio
async def test_delete_entry_soft_delete(db_session, kb, monkeypatch):
    from knowlink.api.routers import fastpath as fp_router

    entry = FastPathEntry(kb_id="kb1", canonical_query="q1", answer_template="t1")
    db_session.add(entry)
    db_session.commit()
    fake_vec = FakeVector()
    monkeypatch.setattr(fp_router, "get_storage", lambda: FakeStorage(fake_vec))
    result = await fp_router.delete_entry(entry.id, db_session)
    assert result == {"ok": True}
    assert db_session.get(FastPathEntry, entry.id).deleted_at is not None


def test_faq_candidates_lists_only_generated(db_session, kb):
    from knowlink.api.routers import fastpath as fp_router

    _mk_doc_and_chunks(db_session)
    db_session.add(Chunk(id="faq1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="doc1",
                         content="等待期多久能赔", chunk_type="faq",
                         metadata_={"generated": True, "source_chunk_id": "c1"}))
    db_session.add(Chunk(id="faq2", tenant_id=1, knowledge_base_id="kb1", knowledge_id="doc1",
                         content="手工导入的问句", chunk_type="faq"))
    db_session.commit()
    res = fp_router.faq_candidates("kb1", session=db_session)
    assert len(res["items"]) == 1
    assert res["items"][0]["question"] == "等待期多久能赔"
    assert res["items"][0]["source_chunk_id"] == "c1"
    assert res["items"][0]["promoted"] is False


def test_faq_candidates_marks_promoted(db_session, kb):
    from knowlink.api.routers import fastpath as fp_router

    _mk_doc_and_chunks(db_session)
    db_session.add(Chunk(id="faq1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="doc1",
                         content="等待期多久能赔", chunk_type="faq",
                         metadata_={"generated": True, "source_chunk_id": "c1"}))
    db_session.add(FastPathEntry(kb_id="kb1", canonical_query="等待期多久能赔", answer_template="t",
                                 source="faq", source_chunk_id="faq1"))
    db_session.commit()
    res = fp_router.faq_candidates("kb1", session=db_session)
    assert res["items"][0]["promoted"] is True


@pytest.mark.asyncio
async def test_promote_faq_creates_active_entry(db_session, kb, monkeypatch):
    from knowlink.api.routers import fastpath as fp_router

    _mk_doc_and_chunks(db_session)
    db_session.add(Chunk(id="faq1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="doc1",
                         content="等待期多久能赔", chunk_type="faq",
                         metadata_={"generated": True, "source_chunk_id": "c1"}))
    db_session.commit()

    fake_vec = FakeVector()
    monkeypatch.setattr(fp_router, "get_storage", lambda: FakeStorage(fake_vec))

    async def fake_template(q, content):
        return "等待期为{waiting_period}天", [{"name": "waiting_period", "description": "天数"}]
    monkeypatch.setattr("knowlink.fastpath.filler.generate_template_from_chunk", fake_template)

    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr("knowlink.indexing.embedder.embed_texts", _embed)

    res = await fp_router.promote_faq("kb1", fp_router.PromoteRequest(chunk_ids=["faq1"]), db_session)
    assert res["count"] == 1
    entry = db_session.query(FastPathEntry).one()
    assert entry.source == "faq"
    assert entry.status == "active"
    assert entry.target_chunk_id == "c1"
    assert entry.answer_template == "等待期为{waiting_period}天"
    assert fake_vec.upserted and fake_vec.upserted[0][0] == "fp_kb1"


def test_stats_aggregates(db_session, kb):
    from knowlink.api.routers import fastpath as fp_router

    db_session.add_all([
        FastPathHit(kb_ids=["kb1"], layer="exact", query="q1", duration_ms=10),
        FastPathHit(kb_ids=["kb1"], layer="semantic", query="q2", duration_ms=200),
        FastPathHit(kb_ids=["kb1"], layer="miss", query="q3", duration_ms=2000),
        FastPathEntry(kb_id="kb1", canonical_query="q", answer_template="t", status="active"),
        FastPathEntry(kb_id="kb1", canonical_query="q2", answer_template="t", status="draft"),
    ])
    db_session.commit()
    res = fp_router.stats("kb1", db_session)
    assert res.total_hits == 3
    assert res.hit_rate == pytest.approx(2 / 3, abs=1e-4)
    assert res.by_layer["exact"] == 1
    assert res.total_entries == 2
    assert res.active_entries == 1
