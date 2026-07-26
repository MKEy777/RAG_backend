import pytest

from pyrag.models.wiki import WikiPage
from pyrag.wiki.dedup import dedup_candidates, prefilter_candidates, surface_score


def _page(slug, title, aliases=None, kind="entity", content="已有内容"):
    return WikiPage(
        page_id=slug.replace("/", "-"), tenant_id=1, knowledge_base_id="kb1",
        kind=kind, topic_key=f"kb1:{kind}:{slug}", title=title, slug=slug,
        content=content, aliases=aliases or [], status="published", version=1,
    )


def test_surface_score_name_variant_high():
    page = _page("entity/acme-corp", "Acme Corp", aliases=["Acme Corporation"])
    cand = {"name": "Acme Corporation", "aliases": []}
    assert surface_score(cand, page) > 0.5


def test_surface_score_unrelated_zero():
    page = _page("entity/chuantong-wenhua", "中华优秀传统文化")
    cand = {"name": "城镇登记失业人员", "aliases": []}
    assert surface_score(cand, page) < 0.08


def test_prefilter_drops_unrelated():
    pages = [_page("entity/acme", "Acme Corp"), _page("entity/other", "完全无关条目")]
    cand = {"name": "Acme Corporation", "aliases": []}
    kept = prefilter_candidates(cand, pages)
    assert [p.slug for p in kept] == ["entity/acme"]


async def test_dedup_returns_valid_merges_only(db_session, fake_llm_factory, monkeypatch):
    db_session.add(_page("entity/acme", "Acme Corp", aliases=["Acme"]))
    db_session.commit()
    # LLM 声称两个合并，但 entity/ghost 不存在 → 只接受 acme
    llm = fake_llm_factory(['{"merges": {"entity/acme-corporation": "entity/acme", "entity/x": "entity/ghost"}}'])
    monkeypatch.setattr("pyrag.wiki.dedup.get_llm_client", lambda: llm)
    merges = await dedup_candidates(
        db_session, "kb1",
        [{"slug": "entity/acme-corporation", "name": "Acme Corporation", "_type": "entity"},
         {"slug": "entity/x", "name": "X", "_type": "entity"}],
    )
    assert merges == {"entity/acme-corporation": "entity/acme"}


async def test_dedup_no_existing_pages_skips_llm(db_session, fake_llm_factory, monkeypatch):
    llm = fake_llm_factory([])
    monkeypatch.setattr("pyrag.wiki.dedup.get_llm_client", lambda: llm)
    merges = await dedup_candidates(db_session, "kb1", [{"slug": "entity/a", "name": "A"}])
    assert merges == {} and llm.calls == []


async def test_reduce_pages_redirects_merged_slug(db_session, fake_llm_factory, fake_storage, monkeypatch):
    from pyrag.wiki.ingest import MapResult
    from pyrag.wiki.merge import reduce_pages

    db_session.add(_page("entity/acme", "Acme Corp", aliases=["Acme"]))
    db_session.commit()

    dedup_llm = fake_llm_factory(['{"merges": {"entity/acme-corporation": "entity/acme"}}'])
    merge_llm = fake_llm_factory(["SUMMARY: merged\nAcme Corp 合并后内容"])
    monkeypatch.setattr("pyrag.wiki.dedup.get_llm_client", lambda: dedup_llm)
    monkeypatch.setattr("pyrag.wiki.merge.get_llm_client", lambda: merge_llm)
    monkeypatch.setattr("pyrag.wiki.merge.get_storage", lambda: fake_storage)

    mr = MapResult(candidates=[{
        "slug": "entity/acme-corporation", "name": "Acme Corporation",
        "_type": "entity", "description": "d", "details": "Acme Corporation 是一家公司",
    }])
    await reduce_pages(db_session, "kb1", 1, "doc1", "文档A", mr)

    pages = db_session.query(WikiPage).filter(WikiPage.knowledge_base_id == "kb1").all()
    slugs = {p.slug for p in pages}
    assert "entity/acme-corporation" not in slugs
    acme = next(p for p in pages if p.slug == "entity/acme")
    assert "Acme Corporation" in (acme.aliases or [])


async def test_overview_slug_uses_doc_id(db_session, fake_llm_factory, fake_storage, monkeypatch):
    from pyrag.wiki.ingest import MapResult
    from pyrag.wiki.merge import reduce_pages

    monkeypatch.setattr("pyrag.wiki.merge.get_storage", lambda: fake_storage)
    mr = MapResult(overview_content="正文", overview_summary="一句话")
    await reduce_pages(db_session, "kb1", 1, "doc42", "同名文档.pdf", mr)

    page = db_session.query(WikiPage).filter(WikiPage.knowledge_base_id == "kb1").one()
    assert page.slug == "overview/doc42"
    assert page.title == "同名文档.pdf"
