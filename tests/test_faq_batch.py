from __future__ import annotations

import numpy as np
import pytest

from knowlink.models.chunk import Chunk
from knowlink.models.knowledge_base import KnowledgeBase


@pytest.fixture
def faq_env(db_session, monkeypatch):
    db_session.add(KnowledgeBase(id="kbf", tenant_id=1, name="f", type="faq", embedding_model_id="m"))
    db_session.commit()

    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    class V:
        async def upsert(self, kb_id, ids, vectors, metadata):
            pass

    class S:
        vector = V()

    monkeypatch.setattr("knowlink.api.routers.faq.embed_texts", fake_embed)
    monkeypatch.setattr("knowlink.api.routers.faq.get_storage", lambda: S())


async def test_batch_dry_run_reports_errors(db_session, faq_env):
    from knowlink.api.routers.faq import FaqBatchBody, FaqItemBody, import_faq_batch

    body = FaqBatchBody(items=[
        FaqItemBody(question="问题一", answer="答案一"),
        FaqItemBody(question="", answer="缺问题"),
    ], dry_run=True)
    out = await import_faq_batch("kbf", body, session=db_session)
    assert out["valid"] == 1 and out["imported"] == 0
    assert out["errors"][0]["index"] == 1


async def test_batch_import_creates_chunks(db_session, faq_env):
    from knowlink.api.routers.faq import FaqBatchBody, FaqItemBody, import_faq_batch

    body = FaqBatchBody(items=[
        FaqItemBody(question="问题一", answer="答案一", similar_questions=["变体一"]),
        FaqItemBody(question="问题二", answer="答案二"),
    ])
    out = await import_faq_batch("kbf", body, session=db_session)
    assert out["imported"] == 2
    n = db_session.query(Chunk).filter(Chunk.knowledge_base_id == "kbf", Chunk.chunk_type == "faq").count()
    assert n == 3  # 标准问2 + 相似问1
