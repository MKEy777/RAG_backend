"""End-to-end integration test: upload → parse → chunk → embed → retrieve → generate.

Uses respx to mock all LLM endpoints so no real LLM server is needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pyrag.models.base import Base
from pyrag.models.knowledge import Knowledge
from pyrag.models.knowledge_base import KnowledgeBase
from pyrag.settings import settings

from tests.fixtures.llm_responses import (
    mock_chat_response,
    mock_embedding_response,
    mock_rerank_response,
)

LLM_BASE = settings.llm_base_url.rstrip("/")


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_kb(db_session):
    kb = KnowledgeBase(
        id="kb-test-001",
        tenant_id=1,
        name="Integration Test KB",
        type="document",
        embedding_model_id="mock-embed",
        chunking_config={"strategy": "auto"},
        indexing_strategy={"vector": True, "keyword": True},
    )
    db_session.add(kb)
    db_session.commit()
    return kb


SAMPLE_MD = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.

## How RAG Works

RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context for generating answers.

## Benefits of RAG

- Improved factual accuracy
- Reduced hallucination
- Source attribution and traceability
"""


@respx.mock
@pytest.mark.asyncio
async def test_full_pipeline(db_session, sample_kb, tmp_path):
    respx.post(f"{LLM_BASE}/embeddings").mock(
        side_effect=lambda request: httpx.Response(
            200, json=mock_embedding_response(json.loads(request.content)["input"])
        )
    )
    respx.post(f"{LLM_BASE}/chat/completions").mock(
        side_effect=lambda request: httpx.Response(
            200, json=mock_chat_response(json.loads(request.content)["messages"])
        )
    )

    from pyrag.indexing.pipeline import run_indexing_pipeline
    from pyrag.models.chunk import Chunk

    doc = Knowledge(
        id="doc-test-001",
        tenant_id=1,
        knowledge_base_id=sample_kb.id,
        type="file",
        title="sample.md",
        file_name="sample.md",
        file_type="text/markdown",
        file_size=len(SAMPLE_MD),
        parse_status="processing",
    )
    db_session.add(doc)
    db_session.commit()

    await run_indexing_pipeline(
        session=db_session,
        knowledge_id=doc.id,
        kb_id=sample_kb.id,
        tenant_id=1,
        markdown=SAMPLE_MD,
        doc_title="sample.md",
    )

    db_session.refresh(doc)
    assert doc.parse_status == "completed"

    chunks = (
        db_session.query(Chunk)
        .filter(Chunk.knowledge_id == doc.id, Chunk.deleted_at.is_(None))
        .all()
    )
    assert len(chunks) > 0

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    parent_chunks = [c for c in chunks if c.chunk_type == "parent_text"]
    assert len(text_chunks) > 0
    assert len(parent_chunks) > 0

    for child in text_chunks:
        assert child.parent_chunk_id is not None


@respx.mock
@pytest.mark.asyncio
async def test_retrieval_after_indexing(db_session, sample_kb, tmp_path):
    respx.post(f"{LLM_BASE}/embeddings").mock(
        side_effect=lambda request: httpx.Response(
            200, json=mock_embedding_response(json.loads(request.content)["input"])
        )
    )
    respx.post(f"{LLM_BASE}/chat/completions").mock(
        side_effect=lambda request: httpx.Response(
            200, json=mock_chat_response(json.loads(request.content)["messages"])
        )
    )
    respx.post(f"{settings.llm_rerank_base_url.rstrip('/')}/rerank").mock(
        side_effect=lambda request: httpx.Response(
            200, json=mock_rerank_response("test", ["doc"] * 10)
        )
    )

    from pyrag.indexing.embedder import embed_texts
    from pyrag.indexing.pipeline import run_indexing_pipeline
    from pyrag.retrieval.fusion import rrf_fusion
    from pyrag.retrieval.hybrid import hybrid_search

    doc = Knowledge(
        id="doc-test-002",
        tenant_id=1,
        knowledge_base_id=sample_kb.id,
        type="file",
        title="sample.md",
        file_name="sample.md",
        file_type="text/markdown",
        file_size=len(SAMPLE_MD),
        parse_status="processing",
    )
    db_session.add(doc)
    db_session.commit()

    await run_indexing_pipeline(
        session=db_session,
        knowledge_id=doc.id,
        kb_id=sample_kb.id,
        tenant_id=1,
        markdown=SAMPLE_MD,
        doc_title="sample.md",
    )

    vectors = await embed_texts(["What is RAG?"])
    query_vector = vectors[0]

    vec_results, kw_results = await hybrid_search(sample_kb.id, "What is RAG?", query_vector)
    fused = rrf_fusion(vec_results, kw_results)

    assert len(fused) > 0
    assert fused[0].score > 0


@respx.mock
@pytest.mark.asyncio
async def test_embed_failure_marks_doc_failed(db_session, sample_kb):
    respx.post(f"{LLM_BASE}/embeddings").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    from pyrag.indexing.pipeline import run_indexing_pipeline

    doc = Knowledge(
        id="doc-test-003",
        tenant_id=1,
        knowledge_base_id=sample_kb.id,
        type="file",
        title="fail.md",
        file_name="fail.md",
        file_type="text/markdown",
        file_size=100,
        parse_status="processing",
    )
    db_session.add(doc)
    db_session.commit()

    with pytest.raises(Exception):
        await run_indexing_pipeline(
            session=db_session,
            knowledge_id=doc.id,
            kb_id=sample_kb.id,
            tenant_id=1,
            markdown="# Test\n\nSome content here.",
            doc_title="fail.md",
        )

    db_session.refresh(doc)
    assert doc.parse_status == "failed"
    assert doc.error_message is not None
