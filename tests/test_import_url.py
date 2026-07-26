from __future__ import annotations

import httpx
import pytest
import respx

from knowlink.models.knowledge import Knowledge
from knowlink.models.knowledge_base import KnowledgeBase


@pytest.fixture
def kb(db_session):
    row = KnowledgeBase(id="kb1", tenant_id=1, name="kb1", type="document", embedding_model_id="m")
    db_session.add(row)
    db_session.commit()
    return row


@respx.mock
async def test_import_url_creates_doc(db_session, kb, monkeypatch):
    from knowlink.api.routers import documents as docs_mod

    respx.get("https://example.com/page").mock(
        return_value=httpx.Response(200, text="<html><body><h1>标题</h1><p>正文</p></body></html>",
                                    headers={"content-type": "text/html; charset=utf-8"})
    )
    dispatched = {}

    async def fake_dispatch(doc_id, kb_id, tenant_id, data, filename, original_key):
        dispatched.update(doc_id=doc_id, filename=filename)

    monkeypatch.setattr(docs_mod, "_save_and_dispatch", fake_dispatch)

    body = docs_mod.ImportUrlBody(kb_id="kb1", url="https://example.com/page")
    doc = await docs_mod.import_url_document(body, session=db_session)
    assert dispatched["filename"].endswith(".html")
    row = db_session.get(Knowledge, doc.id)
    assert row.source == "url"
    assert row.metadata_["source_url"] == "https://example.com/page"


async def test_import_url_rejects_bad_scheme(db_session, kb):
    from fastapi import HTTPException

    from knowlink.api.routers import documents as docs_mod

    with pytest.raises(HTTPException):
        await docs_mod.import_url_document(
            docs_mod.ImportUrlBody(kb_id="kb1", url="file:///etc/passwd"), session=db_session
        )
