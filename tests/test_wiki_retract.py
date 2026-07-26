import pytest

from pyrag.models.chunk import Chunk
from pyrag.models.wiki import WikiPage


def _page(slug, title, source_refs, chunk_refs=None, content="旧内容"):
    return WikiPage(
        page_id=slug.replace("/", "-"), tenant_id=1, knowledge_base_id="kb1",
        kind="entity", topic_key=f"kb1:entity:{slug}", title=title, slug=slug,
        content=content, source_refs=source_refs, chunk_refs=chunk_refs or [],
        status="published", version=1,
    )


def _chunk(cid, knowledge_id):
    return Chunk(
        id=cid, tenant_id=1, knowledge_base_id="kb1", knowledge_id=knowledge_id,
        content="x", chunk_type="text",
    )


async def test_retract_sole_source_archives_page(db_session, monkeypatch):
    from pyrag.wiki.retract import retract_document

    db_session.add(_page("entity/a", "A", ["doc1|文档一"]))
    db_session.commit()
    await retract_document(db_session, "kb1", "doc1", "文档一")

    page = db_session.query(WikiPage).filter(WikiPage.slug == "entity/a").one()
    assert page.deleted_at is not None and page.status == "archived"


async def test_retract_multi_source_rewrites_and_strips_refs(db_session, fake_llm_factory, monkeypatch):
    from pyrag.wiki.retract import retract_document

    # chunk id 是 uuid 风格，不以 doc id 开头 —— 旧实现的 startswith 过滤会漏删
    db_session.add(_chunk("aaa111", "doc1"))
    db_session.add(_chunk("bbb222", "doc2"))
    db_session.add(_page("entity/b", "B", ["doc1|文档一", "doc2|文档二"], chunk_refs=["aaa111", "bbb222"]))
    db_session.commit()

    llm = fake_llm_factory(["SUMMARY: 重写后\n仅剩 doc2 贡献的内容"])
    monkeypatch.setattr("pyrag.wiki.retract.get_llm_client", lambda: llm)

    await retract_document(db_session, "kb1", "doc1", "文档一")

    page = db_session.query(WikiPage).filter(WikiPage.slug == "entity/b").one()
    assert page.source_refs == ["doc2|文档二"]
    assert page.chunk_refs == ["bbb222"]
    assert "doc2" in page.content or "剩" in page.content
    assert page.deleted_at is None


async def test_retract_pages_subset_only(db_session, fake_llm_factory, monkeypatch):
    from pyrag.wiki.retract import retract_pages

    db_session.add(_page("entity/c", "C", ["doc1|文档一"]))
    db_session.add(_page("entity/d", "D", ["doc1|文档一"]))
    db_session.commit()
    pages = db_session.query(WikiPage).filter(WikiPage.slug == "entity/c").all()

    await retract_pages(db_session, "kb1", "doc1", "文档一", pages)

    c = db_session.query(WikiPage).filter(WikiPage.slug == "entity/c").one()
    d = db_session.query(WikiPage).filter(WikiPage.slug == "entity/d").one()
    assert c.deleted_at is not None and d.deleted_at is None
