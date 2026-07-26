from __future__ import annotations

from knowlink.generation import answer as ans
from knowlink.retrieval.hybrid import Candidate


def _cand(cid, content="正文"):
    return Candidate(chunk_id=cid, score=0.9, channel="vector", chunk_type="text", content=content)


async def test_stream_error_event_preserves_partial(monkeypatch):
    class FakeClient:
        async def chat_stream(self, messages):
            yield "前半段"
            raise RuntimeError("boom")

    monkeypatch.setattr(ans, "get_llm_client", lambda: FakeClient())
    events = [e async for e in ans.generate_answer_stream("q", [_cand("c1")], [])]
    kinds = [e["event"] for e in events]
    assert kinds == ["message", "error", "references", "done"]


async def test_stream_success_no_error_event(monkeypatch):
    class FakeClient:
        async def chat_stream(self, messages):
            yield "答案"

    monkeypatch.setattr(ans, "get_llm_client", lambda: FakeClient())
    events = [e async for e in ans.generate_answer_stream("q", [_cand("c1")], [])]
    assert [e["event"] for e in events] == ["message", "references", "done"]
