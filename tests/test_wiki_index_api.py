from pyrag.api.routers.wiki import build_index_payload
from pyrag.models.wiki import WikiFolder, WikiPage


def _page(slug, title, kind, folder_id=None, summary="s"):
    return WikiPage(
        page_id=slug.replace("/", "-"), tenant_id=1, knowledge_base_id="kb1",
        kind=kind, topic_key=f"kb1:{kind}:{slug}", title=title, slug=slug,
        summary=summary, content="c", folder_id=folder_id,
        status="published", version=1,
    )


def test_index_payload_groups_and_folders(db_session):
    f1 = WikiFolder(id="f1", knowledge_base_id="kb1", name="AI", parent_id=None)
    f2 = WikiFolder(id="f2", knowledge_base_id="kb1", name="RAG", parent_id="f1")
    db_session.add_all([f1, f2])
    db_session.add(_page("index/main", "Index", "index", summary="intro page"))
    db_session.add(_page("overview/doc1", "文档一", "overview"))
    db_session.add(_page("entity/acme", "Acme", "entity", folder_id="f2"))
    db_session.add(_page("concept/rag", "RAG", "concept", folder_id="f2"))
    db_session.commit()

    payload = build_index_payload(db_session, "kb1")

    assert payload["intro"] == "c"  # index 页 content
    types = {g["type"]: g for g in payload["groups"]}
    assert "index" not in types
    assert [i["slug"] for i in types["overview"]["items"]] == ["overview/doc1"]
    assert types["entity"]["total"] == 1

    folders = {f["id"]: f for f in payload["folders"]}
    assert folders["f2"]["parent_id"] == "f1"
    assert folders["f2"]["page_count"] == 2


def test_index_payload_empty_kb(db_session):
    payload = build_index_payload(db_session, "kb-none")
    assert payload == {"intro": "", "groups": [], "folders": []}
