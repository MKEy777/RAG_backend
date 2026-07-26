import numpy as np
import pytest

from pyrag.models.chunk import Chunk
from pyrag.models.knowledge_base import KnowledgeBase


class FakeVector:
    def __init__(self):
        self.upserts = []
        self.deleted = []

    async def upsert(self, kb_id, chunk_ids, vectors, metadata):
        self.upserts.append((kb_id, list(chunk_ids), metadata))

    async def delete_chunks(self, kb_id, chunk_ids):
        self.deleted.append((kb_id, list(chunk_ids)))


async def test_import_faq_item_embeds_each_question(db_session, monkeypatch):
    from pyrag.api.routers import faq as faq_router

    db_session.add(KnowledgeBase(id="kb1", tenant_id=1, name="kb", type="faq", embedding_model_id="m1"))
    db_session.commit()

    fake_vec = FakeVector()

    class FakeStorage:
        vector = fake_vec

    async def fake_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(faq_router, "get_storage", lambda: FakeStorage())
    monkeypatch.setattr(faq_router, "embed_texts", fake_embed)

    body = faq_router.FaqItemBody(
        question="如何重置密码？",
        similar_questions=["忘记密码怎么办"],
        answer="前往设置页点击重置。",
        negative_questions=["如何注销账号"],
        tags=["账号"],
    )
    result = await faq_router.import_faq_item("kb1", body, db_session)

    chunks = db_session.query(Chunk).filter(Chunk.chunk_type == "faq").all()
    assert len(chunks) == 2  # 标准问 + 相似问分别嵌入
    assert {c.metadata_["role"] for c in chunks} == {"standard", "similar"}
    assert all(c.metadata_["answer"] == "前往设置页点击重置。" for c in chunks)
    assert all(c.metadata_["faq_id"] == result["faq_id"] for c in chunks)
    _, ids, metas = fake_vec.upserts[0]
    assert len(ids) == 2 and all(m["chunk_type"] == "faq" for m in metas)


def test_faq_search_tag_priority_ordering():
    from pyrag.retrieval.faq import sort_faq_candidates
    from pyrag.retrieval.hybrid import Candidate

    c1 = Candidate(chunk_id="1", score=0.9, channel="faq", chunk_type="faq",
                   document_id="", metadata={"tags": ["其他"]})
    c2 = Candidate(chunk_id="2", score=0.5, channel="faq", chunk_type="faq",
                   document_id="", metadata={"tags": ["账号"]})
    ordered = sort_faq_candidates([c1, c2], tag_priority=["账号"])
    assert [c.chunk_id for c in ordered] == ["2", "1"]  # 标签优先于分数
