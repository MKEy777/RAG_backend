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
