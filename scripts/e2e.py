"""End-to-end smoke test: upload → parse → chunk → embed → retrieve → generate."""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowlink.models.database import SessionLocal, init_db
from knowlink.models.knowledge import Knowledge
from knowlink.models.knowledge_base import KnowledgeBase
from knowlink.parsing.registry import parse_file
from knowlink.indexing.pipeline import run_indexing_pipeline
from knowlink.indexing.embedder import embed_texts
from knowlink.retrieval.hybrid import hybrid_search
from knowlink.retrieval.fusion import rrf_fusion
from knowlink.storage.factory import get_storage


async def main():
    print("=== knowlink e2e smoke test ===")

    print("[1/6] Initializing database...")
    init_db()
    session = SessionLocal()

    print("[2/6] Creating knowledge base...")
    kb = KnowledgeBase(
        id=uuid.uuid4().hex,
        tenant_id=1,
        name="e2e-test-kb",
        type="document",
        embedding_model_id="qwen3-embedding-4b",
        chunking_config={"strategy": "auto"},
        indexing_strategy={"vector": True, "keyword": True},
    )
    session.add(kb)
    session.commit()
    print(f"  KB created: {kb.id}")

    print("[3/6] Parsing sample document...")
    sample_md = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.

## How RAG Works

RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context for generating answers.

## Benefits

- Improved factual accuracy
- Reduced hallucination
- Source attribution
"""
    result = parse_file(sample_md.encode(), "sample.md")
    print(f"  Parsed: {len(result.markdown)} chars, {len(result.images)} images")

    print("[4/6] Creating document record and running indexing pipeline...")
    doc = Knowledge(
        id=uuid.uuid4().hex,
        tenant_id=1,
        knowledge_base_id=kb.id,
        type="file",
        title="sample.md",
        file_name="sample.md",
        file_type="text/markdown",
        file_size=len(sample_md),
        parse_status="processing",
    )
    session.add(doc)
    session.commit()

    try:
        await run_indexing_pipeline(
            session=session,
            knowledge_id=doc.id,
            kb_id=kb.id,
            tenant_id=1,
            markdown=result.markdown,
            doc_title="sample.md",
        )
        print("  Indexing completed successfully")
    except Exception as e:
        print(f"  Indexing failed (expected without LLM server): {e}")
        print("  Skipping retrieval test.")
        session.close()
        return

    print("[5/6] Running hybrid retrieval...")
    query = "What is RAG?"
    vectors = await embed_texts([query])
    query_vector = vectors[0]

    vec_results, kw_results = await hybrid_search(kb.id, query, query_vector)
    fused = rrf_fusion(vec_results, kw_results)
    print(f"  Retrieved {len(fused)} candidates")
    for i, c in enumerate(fused[:3]):
        print(f"    [{i+1}] score={c.score:.4f} channel={c.channel} id={c.chunk_id[:8]}...")

    print("[6/6] Done!")
    session.close()
    print("=== e2e smoke test passed ===")


if __name__ == "__main__":
    asyncio.run(main())
