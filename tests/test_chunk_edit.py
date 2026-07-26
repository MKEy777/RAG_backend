from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from knowlink.models.chunk import Chunk


@pytest.fixture
def fake_env(monkeypatch):
    calls = {"upsert": [], "kw": [], "v_del": [], "kw_del": []}

    class V:
        async def upsert(self, kb_id, ids, vectors, metadata):
            calls["upsert"].extend(ids)

        async def delete_chunks(self, kb_id, ids):
            calls["v_del"].extend(ids)

    class K:
        async def index_chunks(self, kb_id, ids, texts, metadata):
            calls["kw"].extend(ids)

        async def delete_chunks(self, kb_id, ids):
            calls["kw_del"].extend(ids)

    class S:
        vector = V()
        keyword = K()

    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("knowlink.storage.factory.get_storage", lambda: S())
    monkeypatch.setattr("knowlink.indexing.embedder.embed_texts", fake_embed)
    return calls


def _chunk(db_session, cid="c1", doc="doc1"):
    row = Chunk(id=cid, tenant_id=1, knowledge_base_id="kb1", knowledge_id=doc,
                content="原内容", chunk_type="text", content_hash="h",
                metadata_={"title_path": ["章节"], "doc_title": "文档"})
    db_session.add(row)
    db_session.commit()
    return row


async def test_update_chunk_reindexes(db_session, fake_env):
    from knowlink.api.routers.documents import ChunkUpdateBody, update_chunk

    _chunk(db_session)
    await update_chunk("doc1", "c1", ChunkUpdateBody(content="新内容"), session=db_session)
    row = db_session.get(Chunk, "c1")
    assert row.content == "新内容"
    assert fake_env["upsert"] == ["c1"] and fake_env["kw"] == ["c1"]


async def test_delete_chunk_soft_deletes_and_cleans_index(db_session, fake_env):
    from knowlink.api.routers.documents import delete_chunk

    _chunk(db_session, cid="c2")
    await delete_chunk("doc1", "c2", session=db_session)
    assert db_session.get(Chunk, "c2").deleted_at is not None
    assert fake_env["v_del"] == ["c2"] and fake_env["kw_del"] == ["c2"]


async def test_update_wrong_doc_404(db_session, fake_env):
    from knowlink.api.routers.documents import ChunkUpdateBody, update_chunk

    _chunk(db_session, cid="c3")
    with pytest.raises(HTTPException):
        await update_chunk("other-doc", "c3", ChunkUpdateBody(content="x"), session=db_session)
