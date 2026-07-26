import numpy as np
import pytest

from pyrag.models.chunk import Chunk
from pyrag.models.knowledge_base import KnowledgeBase


def _kb(qg_config=None):
    return KnowledgeBase(
        id="kb1", tenant_id=1, name="kb", type="document",
        embedding_model_id="m1", question_generation_config=qg_config,
    )


def _chunk(cid, content):
    return Chunk(
        id=cid, tenant_id=1, knowledge_base_id="kb1", knowledge_id="doc1",
        content=content, chunk_type="text",
    )


class FakeVector:
    def __init__(self):
        self.upserts = []

    async def upsert(self, kb_id, chunk_ids, vectors, metadata):
        self.upserts.append((kb_id, list(chunk_ids), metadata))


async def test_question_gen_disabled_by_default(db_session, fake_llm_factory, monkeypatch):
    from pyrag.indexing.question_gen import generate_faq_chunks

    db_session.add(_kb(None))
    db_session.add(_chunk("c1", "RAG 结合检索与生成。"))
    db_session.commit()

    llm = fake_llm_factory([])
    monkeypatch.setattr("pyrag.indexing.question_gen.get_llm_client", lambda: llm)
    n = await generate_faq_chunks(db_session, "kb1", "doc1", 1, "文档一")
    assert n == 0 and llm.calls == []


async def test_question_gen_creates_faq_chunks(db_session, fake_llm_factory, monkeypatch):
    from pyrag.indexing import question_gen

    db_session.add(_kb({"enabled": True}))
    db_session.add(_chunk("c1", "RAG 结合检索与生成。"))
    db_session.commit()

    llm = fake_llm_factory(['["什么是 RAG？", "RAG 如何结合检索与生成？"]'])
    fake_vec = FakeVector()

    class FakeStorage:
        vector = fake_vec

    async def fake_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(question_gen, "get_llm_client", lambda: llm)
    monkeypatch.setattr(question_gen, "get_storage", lambda: FakeStorage())
    monkeypatch.setattr(question_gen, "embed_texts", fake_embed)

    n = await question_gen.generate_faq_chunks(db_session, "kb1", "doc1", 1, "文档一")

    assert n == 2
    faq_chunks = db_session.query(Chunk).filter(Chunk.chunk_type == "faq").all()
    assert {c.content for c in faq_chunks} == {"什么是 RAG？", "RAG 如何结合检索与生成？"}
    assert all(c.metadata_["source_chunk_id"] == "c1" for c in faq_chunks)
    kb_id, ids, metas = fake_vec.upserts[0]
    assert all(m["chunk_type"] == "faq" for m in metas)
