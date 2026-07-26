from __future__ import annotations

import numpy as np
import pytest

from knowlink.models.chunk import Chunk
from knowlink.models.wiki import WikiFolder, WikiPage


@pytest.fixture
def wiki_env(db_session, monkeypatch):
    calls = {"upsert": [], "deleted": []}

    class V:
        async def upsert(self, kb_id, ids, vectors, metadata):
            calls["upsert"].extend(ids)

        async def delete_chunks(self, kb_id, ids):
            calls["deleted"].extend(ids)

    class K(V):
        async def index_chunks(self, kb_id, ids, texts, metadata):
            pass

    class S:
        vector = V()
        keyword = K()

    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("knowlink.storage.factory.get_storage", lambda: S())
    monkeypatch.setattr("knowlink.indexing.embedder.embed_texts", fake_embed)

    page = WikiPage(page_id="p1", tenant_id=1, knowledge_base_id="kb1", kind="entity",
                    topic_key="kb1:entity/x", title="实体X", slug="entity/x",
                    content="旧内容", status="published", version=1)
    old_chunk = Chunk(id="wc1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="p1",
                      content="旧内容", chunk_type="wiki_page", content_hash="h")
    db_session.add_all([page, old_chunk])
    db_session.commit()
    return calls


async def test_update_page_bumps_version_and_reembeds(db_session, wiki_env):
    from knowlink.api.routers.wiki import PageUpdateBody, update_wiki_page

    out = await update_wiki_page("kb1", "p1", PageUpdateBody(content="新内容"), session=db_session)
    assert out["version"] == 2
    assert db_session.get(Chunk, "wc1").deleted_at is not None
    assert len(wiki_env["upsert"]) == 1
    assert "wc1" in wiki_env["deleted"]


def test_folder_crud(db_session):
    from knowlink.api.routers.wiki import FolderBody, create_folder, list_folders

    create_folder("kb1", FolderBody(name="分类A"), session=db_session)
    out = list_folders("kb1", session=db_session)
    assert out["folders"][0]["name"] == "分类A"
