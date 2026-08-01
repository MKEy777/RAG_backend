import numpy as np
import pytest

from knowlink.fastpath.matcher import lookup_fastpath, normalize_query
from knowlink.models.chunk import Chunk
from knowlink.models.fastpath import FastPathEntry
from knowlink.models.knowledge import Knowledge


def test_normalize_query():
    assert normalize_query("  重疾险 等待期 是多久？ ") == "重疾险等待期是多久"
    assert normalize_query("What IS  the Waiting-period?") == "whatisthewaitingperiod"
    assert normalize_query("ＡＢＣ") == "abc"  # NFKC 全角转半角


class FakeVector:
    def __init__(self, results=None):
        self.results = results or []
        self.search_calls = []

    async def search(self, kb_id, query_vector, top_k, threshold=0.0):
        self.search_calls.append((kb_id, threshold))
        return self.results

    async def delete_chunks(self, kb_id, chunk_ids):
        pass


class FakeStorage:
    def __init__(self, vector):
        self.vector = vector


async def _mk_entry(db_session, kb_id="kb1", canonical="重疾险等待期是多久", **kw):
    entry = FastPathEntry(
        kb_id=kb_id,
        canonical_query=canonical,
        example_queries=kw.pop("examples", []),
        answer_template=kw.pop("template", "等待期为30天"),
        target_chunk_id=kw.pop("target_chunk_id", "chunk1"),
        target_document_id=kw.pop("target_document_id", "doc1"),
        **kw,
    )
    db_session.add(entry)
    db_session.commit()
    return entry


async def _mk_target(db_session, chunk_id="chunk1", doc_id="doc1", kb_id="kb1"):
    db_session.add(Chunk(id=chunk_id, tenant_id=1, knowledge_base_id=kb_id,
                         knowledge_id=doc_id, content="等待期为30天", chunk_type="text"))
    db_session.add(Knowledge(id=doc_id, tenant_id=1, knowledge_base_id=kb_id, type="pdf",
                             file_name="doc1.pdf", parse_status="completed"))
    db_session.commit()


@pytest.mark.asyncio
async def test_exact_match_hits_canonical_and_examples(db_session, monkeypatch):
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", True)
    await _mk_target(db_session)
    entry = await _mk_entry(db_session)
    lookup = await lookup_fastpath(db_session, ["kb1"], "重疾险 等待期 是多久？")
    assert lookup is not None
    assert lookup.layer == "exact"
    assert lookup.entry.id == entry.id


@pytest.mark.asyncio
async def test_exact_match_hits_example_query(db_session, monkeypatch):
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", True)
    await _mk_target(db_session)
    entry = await _mk_entry(db_session, examples=["多久可以理赔重疾"])
    lookup = await lookup_fastpath(db_session, ["kb1"], "多久可以理赔重疾？")
    assert lookup is not None
    assert lookup.layer == "exact"
    assert lookup.entry.id == entry.id


@pytest.mark.asyncio
async def test_disabled_or_draft_entries_never_match(db_session, monkeypatch):
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", True)
    await _mk_target(db_session)
    await _mk_entry(db_session, status="draft")
    await _mk_entry(db_session, canonical="另一个问题", status="disabled")
    assert await lookup_fastpath(db_session, ["kb1"], "重疾险等待期是多久") is None


@pytest.mark.asyncio
async def test_semantic_match_uses_namespace_and_threshold(db_session, monkeypatch):
    from knowlink.fastpath import matcher
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "fastpath_match_threshold", 0.8)
    await _mk_target(db_session)
    entry = await _mk_entry(db_session, canonical="重疾险等待期是多久")
    fake_vec = FakeVector(results=[
        {"chunk_id": f"{entry.id}:0", "score": 0.93,
         "metadata": {"entry_id": entry.id}},
    ])
    monkeypatch.setattr(matcher, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(matcher, "embed_texts", _embed)

    lookup = await lookup_fastpath(db_session, ["kb1"], "重疾保险的等待期")
    assert lookup is not None
    assert lookup.layer == "semantic"
    assert lookup.similarity == 0.93
    assert fake_vec.search_calls[0][0] == "fp_kb1"
    assert fake_vec.search_calls[0][1] == 0.8


@pytest.mark.asyncio
async def test_semantic_below_threshold_misses(db_session, monkeypatch):
    from knowlink.fastpath import matcher
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", True)
    entry = await _mk_entry(db_session)
    fake_vec = FakeVector(results=[])
    monkeypatch.setattr(matcher, "get_storage", lambda: FakeStorage(fake_vec))
    async def _embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(matcher, "embed_texts", _embed)
    assert await lookup_fastpath(db_session, ["kb1"], "完全无关的问题") is None


@pytest.mark.asyncio
async def test_stale_target_marks_entry_stale(db_session, monkeypatch):
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", True)
    entry = await _mk_entry(db_session, target_chunk_id="gone")
    db_session.add(Chunk(id="gone", tenant_id=1, knowledge_base_id="kb1",
                         knowledge_id="doc1", content="x", chunk_type="text",
                         deleted_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)))
    db_session.commit()
    lookup = await lookup_fastpath(db_session, ["kb1"], "重疾险等待期是多久")
    assert lookup is None
    db_session.refresh(entry)
    assert entry.status == "stale"


@pytest.mark.asyncio
async def test_fastpath_disabled_returns_none(db_session, monkeypatch):
    from knowlink import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "fastpath_enabled", False)
    await _mk_target(db_session)
    await _mk_entry(db_session)
    assert await lookup_fastpath(db_session, ["kb1"], "重疾险等待期是多久") is None
