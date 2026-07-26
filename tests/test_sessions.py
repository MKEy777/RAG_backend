from __future__ import annotations

from datetime import datetime, timedelta, timezone

from knowlink.models.chat import ChatMessage, ChatSession


def test_models_roundtrip(db_session):
    s = ChatSession(id="s1", tenant_id=1, title="新对话", kb_ids=["kb1"])
    db_session.add(s)
    db_session.add(ChatMessage(session_id="s1", role="user", content="你好"))
    db_session.commit()
    assert db_session.get(ChatSession, "s1").kb_ids == ["kb1"]
    msgs = db_session.query(ChatMessage).filter(ChatMessage.session_id == "s1").all()
    assert msgs[0].role == "user"


def test_load_history_window(db_session):
    from knowlink.api.routers.chat import _load_history

    db_session.add(ChatSession(id="s2", tenant_id=1, title="t", kb_ids=[]))
    base = datetime.now(timezone.utc)
    for i in range(12):
        db_session.add(ChatMessage(
            session_id="s2", role="user" if i % 2 == 0 else "assistant",
            content=f"m{i}", created_at=base + timedelta(seconds=i),
        ))
    db_session.commit()
    history = _load_history(db_session, "s2", rounds=3)
    assert len(history) == 6
    assert history[0]["content"] == "m6"
    assert history[-1]["content"] == "m11"


async def test_chat_persists_turn_with_session_id(db_session, monkeypatch):
    from knowlink.api.routers import chat as chat_mod
    from knowlink.retrieval.pipeline import RetrievalResult
    from knowlink.retrieval.hybrid import Candidate

    db_session.add(ChatSession(id="s3", tenant_id=1, title="新对话", kb_ids=["kb1"]))
    db_session.commit()

    async def fake_retrieve(session, kb_ids, query, **kw):
        return RetrievalResult(hits=[Candidate(chunk_id="c1", score=1.0, channel="vector", chunk_type="text", content="x")])

    async def fake_generate(query, candidates, history):
        return {"answer": "回答", "references": [{"id": "c1"}]}

    class _NoClose:
        def __init__(self, s):
            self._s = s

        def __getattr__(self, n):
            return getattr(self._s, n)

        def close(self):
            pass

    monkeypatch.setattr(chat_mod, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(chat_mod, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_mod, "enrich_faq_candidates", lambda s, h: h)
    monkeypatch.setattr(chat_mod, "_session_factory", lambda: _NoClose(db_session))

    body = chat_mod.ChatRequest(kb_ids=["kb1"], query="问题", stream=False, session_id="s3")
    await chat_mod.chat(body, session=db_session)

    msgs = db_session.query(ChatMessage).filter(ChatMessage.session_id == "s3").all()
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].content == "回答"
    assert db_session.get(ChatSession, "s3").title == "问题"
