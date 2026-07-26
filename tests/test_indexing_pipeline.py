from __future__ import annotations

import numpy as np

from knowlink.indexing import pipeline as ip
from knowlink.models.chunk import Chunk
from knowlink.models.knowledge import Knowledge


async def _fake_embed(texts):
    return np.ones((len(texts), 4), dtype=np.float32)


def _make_storage(order):
    class V:
        async def upsert(self, kb_id, ids, vectors, metadata):
            order.append("upsert")

        async def delete_chunks(self, kb_id, ids):
            order.append("v_delete")

        async def delete_document(self, kb_id, knowledge_id):
            order.append("v_delete_doc")

    class K:
        async def index_chunks(self, kb_id, ids, texts, metadata):
            order.append("kw_index")

        async def delete_chunks(self, kb_id, ids):
            order.append("kw_delete")

        async def delete_document(self, kb_id, knowledge_id):
            order.append("kw_delete_doc")

    class S:
        vector = V()
        keyword = K()

    return S()


def _add_doc(db_session, doc_id):
    db_session.add(Knowledge(
        id=doc_id, tenant_id=1, knowledge_base_id="kb1", type="file",
        title="t", file_name="t.md", parse_status="processing",
    ))
    db_session.commit()


_CFG = {"enable_parent_child": False, "strategy": "recursive"}


async def test_new_index_written_before_old_deleted(db_session, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(ip, "get_storage", lambda: _make_storage(order))
    monkeypatch.setattr(ip, "embed_texts", _fake_embed)
    _add_doc(db_session, "doc1")

    await ip.run_indexing_pipeline(
        session=db_session, knowledge_id="doc1", kb_id="kb1", tenant_id=1,
        markdown="第一版内容。" * 20, doc_title="t", chunking_config=_CFG,
    )
    order.clear()
    await ip.run_indexing_pipeline(
        session=db_session, knowledge_id="doc1", kb_id="kb1", tenant_id=1,
        markdown="第二版内容。" * 20, doc_title="t", chunking_config=_CFG,
    )
    assert "upsert" in order and "v_delete" in order
    assert order.index("v_delete") > order.index("upsert")


async def test_failure_keeps_old_chunks_live(db_session, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(ip, "get_storage", lambda: _make_storage(order))
    monkeypatch.setattr(ip, "embed_texts", _fake_embed)
    _add_doc(db_session, "doc2")
    await ip.run_indexing_pipeline(
        session=db_session, knowledge_id="doc2", kb_id="kb1", tenant_id=1,
        markdown="旧版本内容。" * 20, doc_title="t", chunking_config=_CFG,
    )
    live_before = db_session.query(Chunk).filter(
        Chunk.knowledge_id == "doc2", Chunk.deleted_at.is_(None)).count()

    async def boom(texts):
        raise RuntimeError("embed down")

    monkeypatch.setattr(ip, "embed_texts", boom)
    try:
        await ip.run_indexing_pipeline(
            session=db_session, knowledge_id="doc2", kb_id="kb1", tenant_id=1,
            markdown="新版本内容。" * 20, doc_title="t", chunking_config=_CFG,
        )
    except RuntimeError:
        pass
    live_after = db_session.query(Chunk).filter(
        Chunk.knowledge_id == "doc2", Chunk.deleted_at.is_(None)).count()
    assert live_after == live_before
