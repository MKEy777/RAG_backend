import pytest

from datetime import datetime, timedelta, timezone

from knowlink.models.chunk import Chunk
from knowlink.models.wiki import WikiPage
from knowlink.wiki import dedup as dedup_mod
from knowlink.wiki import finalize as finalize_mod
from knowlink.wiki.dedup import dedup_candidates, prefilter_candidates, surface_score


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
    monkeypatch.setattr("knowlink.wiki.dedup.get_llm_client", lambda: llm)
    merges = await dedup_candidates(
        db_session, "kb1",
        [{"slug": "entity/acme-corporation", "name": "Acme Corporation", "_type": "entity"},
         {"slug": "entity/x", "name": "X", "_type": "entity"}],
    )
    assert merges == {"entity/acme-corporation": "entity/acme"}


async def test_dedup_no_existing_pages_skips_llm(db_session, fake_llm_factory, monkeypatch):
    llm = fake_llm_factory([])
    monkeypatch.setattr("knowlink.wiki.dedup.get_llm_client", lambda: llm)
    merges = await dedup_candidates(db_session, "kb1", [{"slug": "entity/a", "name": "A"}])
    assert merges == {} and llm.calls == []


async def test_reduce_pages_redirects_merged_slug(db_session, fake_llm_factory, fake_storage, monkeypatch):
    from knowlink.wiki.ingest import MapResult
    from knowlink.wiki.merge import reduce_pages

    db_session.add(_page("entity/acme", "Acme Corp", aliases=["Acme"]))
    db_session.commit()

    dedup_llm = fake_llm_factory(['{"merges": {"entity/acme-corporation": "entity/acme"}}'])
    merge_llm = fake_llm_factory(["SUMMARY: merged\nAcme Corp 合并后内容"])
    monkeypatch.setattr("knowlink.wiki.dedup.get_llm_client", lambda: dedup_llm)
    monkeypatch.setattr("knowlink.wiki.merge.get_llm_client", lambda: merge_llm)
    monkeypatch.setattr("knowlink.wiki.merge.get_storage", lambda: fake_storage)

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
    from knowlink.wiki.ingest import MapResult
    from knowlink.wiki.merge import reduce_pages

    monkeypatch.setattr("knowlink.wiki.merge.get_storage", lambda: fake_storage)
    mr = MapResult(overview_content="正文", overview_summary="一句话")
    await reduce_pages(db_session, "kb1", 1, "doc42", "同名文档.pdf", mr)

    page = db_session.query(WikiPage).filter(WikiPage.knowledge_base_id == "kb1").one()
    assert page.slug == "overview/doc42"
    assert page.title == "同名文档.pdf"


async def test_dedup_exact_title_short_circuits_llm(db_session, monkeypatch):
    db_session.add(_page("entity/song-zikai", "宋子恺"))
    db_session.commit()

    def _fail():
        raise AssertionError("exact title match must not reach LLM")

    monkeypatch.setattr(dedup_mod, "get_llm_client", _fail)
    merges = await dedup_candidates(
        db_session, "kb1",
        [{"slug": "entity/songzikai", "name": "宋子恺", "aliases": []}],
    )
    assert merges == {"entity/songzikai": "entity/song-zikai"}


async def test_dedup_alias_match_short_circuits_llm(db_session, monkeypatch):
    db_session.add(_page("entity/blg", "北京理工大学", aliases=["北理工"]))
    db_session.commit()
    monkeypatch.setattr(dedup_mod, "get_llm_client", lambda: (_ for _ in ()).throw(AssertionError()))
    merges = await dedup_candidates(
        db_session, "kb1",
        [{"slug": "entity/beiligong", "name": "北理工", "aliases": []}],
    )
    assert merges == {"entity/beiligong": "entity/blg"}


async def test_finalize_merges_duplicate_titles(db_session, monkeypatch):
    t0 = datetime.now(timezone.utc)
    p1 = _page("entity/song-zikai", "宋子恺", content="内容A")
    p1.source_refs = ["d1|doc1"]
    p1.chunk_refs = ["c1"]
    p2 = _page("entity/songzikai", "宋子恺", aliases=["宋先生"], content="内容B")
    p2.source_refs = ["d2|doc2"]
    p2.chunk_refs = ["c2"]
    p1.created_at = t0
    p2.created_at = t0 + timedelta(seconds=1)
    db_session.add_all([p1, p2])
    db_session.add(Chunk(
        id="cw2", tenant_id=1, knowledge_base_id="kb1", knowledge_id="entity-songzikai",
        content="内容B", chunk_type="wiki_page",
    ))
    db_session.commit()

    deleted: list[tuple[str, list[str]]] = []

    class FakeStore:
        async def delete_chunks(self, kb_id, chunk_ids):
            deleted.append((kb_id, list(chunk_ids)))

    class Bundle:
        vector = FakeStore()
        keyword = FakeStore()

    monkeypatch.setattr(finalize_mod, "get_storage", lambda: Bundle())
    await finalize_mod._merge_duplicate_titles(db_session, "kb1")

    live = db_session.query(WikiPage).filter(WikiPage.deleted_at.is_(None)).all()
    assert [p.page_id for p in live] == ["entity-song-zikai"]
    survivor = live[0]
    assert set(survivor.source_refs) == {"d1|doc1", "d2|doc2"}
    assert set(survivor.chunk_refs) == {"c1", "c2"}
    assert "宋先生" in survivor.aliases
    assert "内容B" in survivor.content
    assert db_session.get(Chunk, "cw2").deleted_at is not None
    assert deleted == [("kb1", ["cw2"]), ("kb1", ["cw2"])]
