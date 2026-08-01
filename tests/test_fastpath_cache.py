from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from knowlink.fastpath.cache import (
    invalidate_cache_for_chunks,
    kb_key,
    lookup_cache,
    mark_entries_stale_for_chunks,
    mark_entries_stale_for_document,
    write_cache,
)
from knowlink.models.chunk import Chunk
from knowlink.models.fastpath import FastPathEntry, SemanticCacheEntry
from knowlink.models.knowledge import Knowledge


class FakeVector:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.upserted = []
        self.deleted = []
        self._store: dict[str, dict[str, np.ndarray]] = {}

    async def upsert(self, kb_id, chunk_ids, vectors, metadata):
        self.upserted.append((kb_id, list(chunk_ids)))
        for cid, vec in zip(chunk_ids, vectors):
            self._store.setdefault(kb_id, {})[cid] = vec
            if cid not in self.search_results:
                self.search_results.append({"chunk_id": cid, "score": 0.99, "metadata": metadata[0]})

    async def search(self, kb_id, query_vector, top_k, threshold=0.0):
        return [r for r in self.search_results if r.get("chunk_id", "").startswith(kb_id.rsplit("_", 1)[-1][:0]) or True][:top_k]

    async def delete_chunks(self, kb_id, chunk_ids):
        self.deleted.append((kb_id, list(chunk_ids)))
        for cid in chunk_ids:
            self.search_results = [r for r in self.search_results if r["chunk_id"] != cid]


class FakeStorage:
    def __init__(self, vector):
        self.vector = vector


def _mk_chunk(db_session, cid, kb_id="kb1", doc_id="doc1", content="等待期为30天", deleted=False):
    db_session.add(Chunk(id=cid, tenant_id=1, knowledge_base_id=kb_id,
                         knowledge_id=doc_id, content=content, chunk_type="text",
                         content_hash=content, deleted_at=datetime.now(timezone.utc) if deleted else None))
    db_session.commit()


def _mk_doc(db_session, doc_id="doc1", kb_id="kb1", deleted=False):
    db_session.add(Knowledge(id=doc_id, tenant_id=1, knowledge_base_id=kb_id, type="pdf",
                             file_name="doc.pdf", parse_status="completed",
                             deleted_at=datetime.now(timezone.utc) if deleted else None))
    db_session.commit()


async def _write(db_session, query="等待期是多久", answer="等待期为30天", chunk_ids=("c1",)):
    for cid in chunk_ids:
        _mk_chunk(db_session, cid)
    await write_cache(db_session, ["kb1"], query, answer, [{"id": "c1", "chunk_id": "c1"}], list(chunk_ids))
    return db_session.query(SemanticCacheEntry).one()


@pytest.mark.asyncio
async def test_kb_key_stable_and_sorted(db_session):
    assert kb_key(["b", "a"]) == kb_key(["a", "b"])
    assert len(kb_key(["kb1"])) == 16


@pytest.mark.asyncio
async def test_write_and_hit(db_session, monkeypatch):
    from knowlink.fastpath import cache as cache_mod
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "cache_enabled", True)
    fake_vec = FakeVector()
    monkeypatch.setattr(cache_mod, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(cache_mod, "embed_texts", _embed)

    entry = await _write(db_session)
    assert entry.answer_text == "等待期为30天"

    hit = await lookup_cache(db_session, ["kb1"], "等待期是多久")
    assert hit is not None
    assert hit.answer_text == "等待期为30天"
    assert hit.hit_count == 1


@pytest.mark.asyncio
async def test_no_answer_or_no_sources_not_cached(db_session, monkeypatch):
    from knowlink.fastpath import cache as cache_mod

    fake_vec = FakeVector()
    monkeypatch.setattr(cache_mod, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(cache_mod, "embed_texts", _embed)

    await write_cache(db_session, ["kb1"], "q", "无法回答", [], [])
    await write_cache(db_session, ["kb1"], "q2", "好答案", [], [])
    assert db_session.query(SemanticCacheEntry).count() == 0


@pytest.mark.asyncio
async def test_source_chunk_changed_invalidates(db_session, monkeypatch):
    from knowlink.fastpath import cache as cache_mod
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "cache_enabled", True)
    fake_vec = FakeVector()
    monkeypatch.setattr(cache_mod, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(cache_mod, "embed_texts", _embed)

    await _write(db_session, chunk_ids=("c1",))
    chunk = db_session.get(Chunk, "c1")
    chunk.content = "等待期变更为60天"
    chunk.content_hash = "等待期变更为60天"
    db_session.commit()

    assert await lookup_cache(db_session, ["kb1"], "等待期是多久") is None
    assert db_session.query(SemanticCacheEntry).count() == 0


@pytest.mark.asyncio
async def test_source_chunk_deleted_invalidates(db_session, monkeypatch):
    from knowlink.fastpath import cache as cache_mod
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "cache_enabled", True)
    fake_vec = FakeVector()
    monkeypatch.setattr(cache_mod, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(cache_mod, "embed_texts", _embed)

    await _write(db_session, chunk_ids=("c1",))
    chunk = db_session.get(Chunk, "c1")
    chunk.deleted_at = datetime.now(timezone.utc)
    db_session.commit()

    assert await lookup_cache(db_session, ["kb1"], "等待期是多久") is None
    assert db_session.query(SemanticCacheEntry).count() == 0


@pytest.mark.asyncio
async def test_expired_entry_skipped(db_session, monkeypatch):
    from knowlink.fastpath import cache as cache_mod
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "cache_enabled", True)
    fake_vec = FakeVector()
    monkeypatch.setattr(cache_mod, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(cache_mod, "embed_texts", _embed)

    entry = await _write(db_session)
    entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.commit()

    assert await lookup_cache(db_session, ["kb1"], "等待期是多久") is None
    assert db_session.query(SemanticCacheEntry).count() == 0


def test_invalidate_cache_for_chunks(db_session, monkeypatch):
    db_session.add(Chunk(id="c1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="d1",
                         content="x", chunk_type="text", content_hash="h1"))
    db_session.add(SemanticCacheEntry(kb_key=kb_key(["kb1"]), query_text="q", answer_text="a",
                                      source_chunk_ids=["c1"], source_chunk_hashes=["h1"]))
    db_session.commit()
    invalidate_cache_for_chunks(db_session, ["c1"])
    assert db_session.query(SemanticCacheEntry).count() == 0


def test_mark_entries_stale_for_chunks(db_session):
    db_session.add(Chunk(id="c1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="d1",
                         content="x", chunk_type="text", content_hash="h1"))
    db_session.add(FastPathEntry(kb_id="kb1", canonical_query="q", answer_template="t",
                                 target_chunk_id="c1", target_document_id="d1"))
    db_session.commit()
    mark_entries_stale_for_chunks(db_session, ["c1"])
    entry = db_session.query(FastPathEntry).one()
    assert entry.status == "stale"


def test_mark_entries_stale_for_document(db_session):
    db_session.add(FastPathEntry(kb_id="kb1", canonical_query="q", answer_template="t",
                                 target_chunk_id="c1", target_document_id="d1"))
    db_session.commit()
    mark_entries_stale_for_document(db_session, "d1")
    assert db_session.query(FastPathEntry).one().status == "stale"
