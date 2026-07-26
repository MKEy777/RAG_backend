from __future__ import annotations

from knowlink.generation import suggestions as sg


async def test_generate_suggestions_parses_array(monkeypatch, fake_llm_factory):
    fake = fake_llm_factory(['["追问一", "追问二", "追问三", "多余的"]'])
    monkeypatch.setattr(sg, "get_llm_client", lambda: fake)
    out = await sg.generate_suggestions("问题", "答案")
    assert out == ["追问一", "追问二", "追问三"]


async def test_generate_suggestions_failure_returns_empty(monkeypatch, fake_llm_factory):
    fake = fake_llm_factory([])
    monkeypatch.setattr(sg, "get_llm_client", lambda: fake)
    assert await sg.generate_suggestions("q", "a") == []


async def test_stream_emits_suggestions_before_references(monkeypatch):
    from knowlink.generation import answer as ans
    from knowlink.retrieval.hybrid import Candidate

    class FakeClient:
        async def chat_stream(self, messages):
            yield "答案"

    async def fake_suggest(query, answer):
        return ["追问一"]

    monkeypatch.setattr(ans, "get_llm_client", lambda: FakeClient())
    monkeypatch.setattr(ans, "generate_suggestions", fake_suggest)
    cand = Candidate(chunk_id="c1", score=0.9, channel="vector", chunk_type="text", content="x")
    events = [e async for e in ans.generate_answer_stream("q", [cand], [], suggest=True)]
    kinds = [e["event"] for e in events]
    assert kinds == ["message", "suggestions", "references", "done"]
