import json

import pytest

from knowlink.api.routers import chat as chat_router
from knowlink.fastpath.service import FastPathResult
from knowlink.models.chunk import Chunk
from knowlink.models.knowledge import Knowledge
from knowlink.models.knowledge_base import KnowledgeBase


def _mk_data(db_session):
    db_session.add(KnowledgeBase(id="kb1", tenant_id=1, name="测试库", type="general",
                                 embedding_model_id="m1"))
    db_session.add(Knowledge(id="d1", tenant_id=1, knowledge_base_id="kb1", type="pdf",
                             file_name="doc.pdf", parse_status="completed"))
    db_session.add(Chunk(id="c1", tenant_id=1, knowledge_base_id="kb1", knowledge_id="d1",
                         content="等待期为30天", chunk_type="text",
                         metadata_={"title_path": ["重疾险"]}, content_hash="h1"))
    db_session.commit()


def _coro(value):
    async def _inner():
        return value
    return _inner()


@pytest.mark.asyncio
async def test_fastpath_hit_skips_retrieval(db_session, monkeypatch):
    """L0 命中时 retrieve_candidates 与 generate_answer_stream 不应被调用。"""
    from knowlink.api.routers import chat as chat_router_mod

    _mk_data(db_session)
    body = chat_router.ChatRequest(kb_ids=["kb1"], query="重疾险等待期是多久", stream=False)

    result = FastPathResult(
        answer="等待期为30天",
        references=[{"id": "c1", "chunk_id": "c1", "document_id": "d1", "title_path": "重疾险"}],
        via="fastpath", layer="exact",
    )
    monkeypatch.setattr(chat_router_mod, "rewrite_query", lambda q, h: _coro(q))
    monkeypatch.setattr(chat_router_mod, "try_fastpath", lambda s, kb_ids, q: _coro(result))
    called = {"retrieve": False, "generate": False}

    async def fake_retrieve(*a, **kw):
        called["retrieve"] = True
        return None
    async def fake_generate(*a, **kw):
        called["generate"] = True
        return {"answer": "x", "references": []}

    monkeypatch.setattr(chat_router_mod, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(chat_router_mod, "generate_answer", fake_generate)

    resp = await chat_router.chat(body, db_session)
    assert resp.answer == "等待期为30天"
    assert resp.references[0]["chunk_id"] == "c1"
    assert not called["retrieve"]
    assert not called["generate"]


def _coro(value):
    async def _inner():
        return value
    return _inner()


@pytest.mark.asyncio
async def test_fastpath_stream_emits_via_event(db_session, monkeypatch):
    from knowlink.api.routers import chat as chat_router_mod

    _mk_data(db_session)
    body = chat_router.ChatRequest(kb_ids=["kb1"], query="重疾险等待期是多久", stream=True)

    result = FastPathResult(
        answer="等待期为30天", references=[], via="cache", layer="cache",
    )
    monkeypatch.setattr(chat_router_mod, "rewrite_query", lambda q, h: _coro(q))
    monkeypatch.setattr(chat_router_mod, "try_fastpath", lambda s, kb_ids, q: _coro(result))
    monkeypatch.setattr(chat_router_mod, "retrieve_candidates", lambda *a, **k: _coro(None))

    resp = await chat_router.chat(body, db_session)
    events = [e async for e in resp.body_iterator]
    parsed = [{"event": e["event"], "data": json.loads(e["data"])} for e in events]
    assert any(p["event"] == "via" and p["data"].get("via") == "cache" for p in parsed)
    assert any(p["event"] == "message" and p["data"].get("content") == "等待期为30天" for p in parsed)
    assert any(p["event"] == "done" for p in parsed)


@pytest.mark.asyncio
async def test_no_fastpath_goes_full_pipeline(db_session, monkeypatch):
    from knowlink.api.routers import chat as chat_router_mod

    _mk_data(db_session)
    body = chat_router.ChatRequest(kb_ids=["kb1"], query="普通问题", stream=False)

    monkeypatch.setattr(chat_router_mod, "rewrite_query", lambda q, h: _coro(q))
    monkeypatch.setattr(chat_router_mod, "try_fastpath", lambda s, kb_ids, q: _coro(None))

    class R:
        hits = []

    async def fake_retrieve(*a, **kw):
        return R()
    async def fake_generate(*a, **kw):
        return {"answer": "完整链路答案", "references": []}
    monkeypatch.setattr(chat_router_mod, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(chat_router_mod, "generate_answer", fake_generate)

    resp = await chat_router.chat(body, db_session)
    assert resp.answer == "完整链路答案"


@pytest.mark.asyncio
async def test_fastpath_fill_failure_falls_back_to_cache(db_session, monkeypatch):
    """L0 命中但取数失败 → 应回落 L1 缓存。"""
    from knowlink.fastpath import service as svc_mod

    _mk_data(db_session)
    seen = {"fill": False, "cache": False}

    class FakeLookup:
        layer = "exact"
        similarity = 1.0
        entry = None
        query_vector = None

    async def fake_lookup_fastpath(s, kb_ids, q):
        return FakeLookup()
    async def fake_fill(s, entry):
        seen["fill"] = True
        return None
    async def fake_lookup_cache(s, kb_ids, q, query_vector=None):
        seen["cache"] = True
        return None

    monkeypatch.setattr(svc_mod, "lookup_fastpath", fake_lookup_fastpath)
    monkeypatch.setattr(svc_mod, "fill_answer", fake_fill)
    monkeypatch.setattr(svc_mod, "lookup_cache", fake_lookup_cache)

    assert await svc_mod.try_fastpath(db_session, ["kb1"], "q") is None
    assert seen["fill"] and seen["cache"]
