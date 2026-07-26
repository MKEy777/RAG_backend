import pytest

from pyrag.models.wiki import WikiPage
from pyrag.wiki.ingest import MapResult


def _page(slug, title, source_refs, content="旧内容", kind="entity"):
    return WikiPage(
        page_id=slug.replace("/", "-"), tenant_id=1, knowledge_base_id="kb1",
        kind=kind, topic_key=f"kb1:{kind}:{slug}", title=title, slug=slug,
        content=content, source_refs=source_refs, status="published", version=1,
    )


async def test_reparse_retracts_stale_slugs(db_session, fake_llm_factory, fake_storage, monkeypatch):
    from pyrag.wiki.merge import reduce_pages

    # doc1 上次贡献了 entity/old；本次抽取不再包含它 → 应撤回归档
    db_session.add(_page("entity/old", "Old", ["doc1|文档一"]))
    db_session.commit()

    monkeypatch.setattr("pyrag.wiki.merge.get_storage", lambda: fake_storage)
    monkeypatch.setattr("pyrag.wiki.dedup.get_llm_client", lambda: fake_llm_factory([]))

    mr = MapResult(overview_content="新概览", overview_summary="s", candidates=[])
    await reduce_pages(db_session, "kb1", 1, "doc1", "文档一", mr)

    old = db_session.query(WikiPage).filter(WikiPage.slug == "entity/old").one()
    assert old.deleted_at is not None and old.status == "archived"


async def test_reparse_same_slug_passes_replace_instruction(db_session, fake_llm_factory, fake_storage, monkeypatch):
    from pyrag.wiki.merge import reduce_pages

    db_session.add(_page("entity/keep", "Keep", ["doc1|文档一"], content="doc1 旧事实"))
    db_session.commit()

    merge_llm = fake_llm_factory(["SUMMARY: 替换后\ndoc1 新事实"])
    monkeypatch.setattr("pyrag.wiki.merge.get_llm_client", lambda: merge_llm)
    monkeypatch.setattr("pyrag.wiki.merge.get_storage", lambda: fake_storage)
    monkeypatch.setattr("pyrag.wiki.dedup.get_llm_client", lambda: fake_llm_factory(['{"merges": {}}']))

    mr = MapResult(candidates=[{
        "slug": "entity/keep", "name": "Keep", "_type": "entity",
        "description": "d", "details": "doc1 新事实",
    }])
    await reduce_pages(db_session, "kb1", 1, "doc1", "文档一", mr)

    user_msg = merge_llm.calls[0]["messages"][1]["content"]
    assert "<replaced_document>" in user_msg and "文档一" in user_msg
    page = db_session.query(WikiPage).filter(WikiPage.slug == "entity/keep").one()
    assert "新事实" in page.content
    # source_refs 不重复追加
    assert page.source_refs.count("doc1|文档一") == 1
