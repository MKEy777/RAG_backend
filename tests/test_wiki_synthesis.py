import pytest

from pyrag.models.knowledge_base import KnowledgeBase
from pyrag.models.wiki import WikiPage


def _page(slug, title, kind="entity", content="内容", out_links=None):
    return WikiPage(
        page_id=slug.replace("/", "-"), tenant_id=1, knowledge_base_id="kb1",
        kind=kind, topic_key=f"kb1:{kind}:{slug}", title=title, slug=slug,
        content=content, out_links=out_links or [], status="published", version=1,
    )


def _kb(wiki_config=None):
    return KnowledgeBase(
        id="kb1", tenant_id=1, name="kb", type="document",
        embedding_model_id="m1", wiki_config=wiki_config or {},
    )


async def test_synthesis_skipped_without_model(db_session, fake_llm_factory, monkeypatch):
    from pyrag.wiki.synthesis import generate_synthesis

    db_session.add(_kb())
    db_session.add(_page("entity/a", "A"))
    db_session.add(_page("entity/b", "B"))
    db_session.commit()

    llm = fake_llm_factory([])
    monkeypatch.setattr("pyrag.wiki.synthesis.get_llm_client", lambda: llm)
    await generate_synthesis(db_session, "kb1", 1)
    assert llm.calls == []  # detail.md §C5 未配置综述模型跳过


async def test_synthesis_uses_configured_model(db_session, fake_llm_factory, monkeypatch):
    from pyrag.wiki.synthesis import generate_synthesis

    db_session.add(_kb({"synthesis_model": "strong-model"}))
    db_session.add(_page("entity/a", "A"))
    db_session.add(_page("entity/b", "B"))
    db_session.commit()

    llm = fake_llm_factory(["SUMMARY: 综览\n## 主题串讲\n[[entity/a|A]] 与 [[entity/b|B]]"])
    monkeypatch.setattr("pyrag.wiki.synthesis.get_llm_client", lambda: llm)
    await generate_synthesis(db_session, "kb1", 1)

    assert llm.calls[0]["model"] == "strong-model"
    page = db_session.query(WikiPage).filter(WikiPage.slug == "synthesis/overview").one()
    assert page.kind == "synthesis" and "主题串讲" in page.content


async def test_comparison_generated_for_colinked_pair(db_session, fake_llm_factory, monkeypatch):
    from pyrag.wiki.synthesis import generate_comparisons

    db_session.add(_kb())
    db_session.add(_page("entity/a", "A"))
    db_session.add(_page("entity/b", "B"))
    # 两个页面同时链接 a 和 b → 共现 2 次，达到阈值
    db_session.add(_page("overview/d1", "D1", kind="overview", out_links=["entity/a", "entity/b"]))
    db_session.add(_page("overview/d2", "D2", kind="overview", out_links=["entity/a", "entity/b"]))
    db_session.commit()

    llm = fake_llm_factory(["SUMMARY: 对照\n| 维度 | A | B |\n|---|---|---|"])
    monkeypatch.setattr("pyrag.wiki.synthesis.get_llm_client", lambda: llm)
    await generate_comparisons(db_session, "kb1", 1)

    page = db_session.query(WikiPage).filter(WikiPage.kind == "comparison").one()
    assert page.slug == "comparison/a-vs-b"


async def test_comparison_skipped_below_threshold(db_session, fake_llm_factory, monkeypatch):
    from pyrag.wiki.synthesis import generate_comparisons

    db_session.add(_kb())
    db_session.add(_page("entity/a", "A"))
    db_session.add(_page("entity/b", "B"))
    db_session.add(_page("overview/d1", "D1", kind="overview", out_links=["entity/a", "entity/b"]))
    db_session.commit()

    llm = fake_llm_factory([])
    monkeypatch.setattr("pyrag.wiki.synthesis.get_llm_client", lambda: llm)
    await generate_comparisons(db_session, "kb1", 1)
    assert llm.calls == []
