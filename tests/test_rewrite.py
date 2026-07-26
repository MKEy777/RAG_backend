from __future__ import annotations

from knowlink.retrieval import rewrite as rw


async def test_rewrite_skips_without_history(monkeypatch):
    def _fail():
        raise AssertionError("no history must not call LLM")

    monkeypatch.setattr(rw, "get_llm_client", _fail)
    assert await rw.rewrite_query("原问题", []) == "原问题"
    assert await rw.rewrite_query("原问题", None) == "原问题"


async def test_rewrite_resolves_reference(monkeypatch, fake_llm_factory):
    fake = fake_llm_factory(['{"rewrite_query": "KnowLink 检索接口的参数说明"}'])
    monkeypatch.setattr(rw, "get_llm_client", lambda: fake)
    out = await rw.rewrite_query("它的参数呢", [{"role": "user", "content": "介绍下检索接口"}])
    assert out == "KnowLink 检索接口的参数说明"
    assert fake.calls[0]["purpose"] == "chat:rewrite"


async def test_rewrite_falls_back_on_error(monkeypatch, fake_llm_factory):
    fake = fake_llm_factory([])  # 耗尽即抛错
    monkeypatch.setattr(rw, "get_llm_client", lambda: fake)
    out = await rw.rewrite_query("它的参数呢", [{"role": "user", "content": "介绍下检索接口"}])
    assert out == "它的参数呢"
