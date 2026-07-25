"""Quick e2e test with a real PDF file."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.parsing.registry import parse_file
from pyrag.chunking import chunk_document
from pyrag.chunking.parent_child import ParentChildPair
from pyrag.llm.client import get_llm_client
from pyrag.indexing.embedder import build_embed_content


async def main():
    pdf_path = Path(__file__).parent / "【论文精读 & 面试题】Transformer论文逐段精读.pdf"
    if not pdf_path.exists():
        # try glob
        candidates = list(Path(__file__).parent.glob("*.pdf"))
        if candidates:
            pdf_path = candidates[0]
        else:
            print("ERROR: No PDF found in tests/")
            return

    print(f"=== Testing with: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.0f} KB) ===\n")

    # Step 1: Parse
    print("[1/4] Parsing PDF...")
    t0 = time.perf_counter()
    data = pdf_path.read_bytes()
    result = parse_file(data, pdf_path.name)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.2f}s")
    print(f"  Markdown length: {len(result.markdown)} chars")
    print(f"  Images extracted: {len(result.images)}")
    print(f"  First 300 chars:\n  {result.markdown[:300]}\n")

    # Step 2: Chunk
    print("[2/4] Chunking...")
    t0 = time.perf_counter()
    chunks = chunk_document(result.markdown, strategy="auto", enable_parent_child=True)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.2f}s")
    print(f"  Parent chunks: {len(chunks)}")
    total_children = sum(len(p.children) for p in chunks if isinstance(p, ParentChildPair))
    print(f"  Child chunks: {total_children}")
    if chunks and isinstance(chunks[0], ParentChildPair):
        first = chunks[0]
        print(f"  First parent title_path: {first.parent_title_path}")
        print(f"  First child text: {first.children[0].text[:150]}...\n")

    # Step 3: Embed (first 5 children)
    print("[3/4] Embedding first 5 chunks via SiliconFlow...")
    client = get_llm_client()
    sample_texts = []
    for pair in chunks[:3]:
        if isinstance(pair, ParentChildPair):
            for child in pair.children[:2]:
                sample_texts.append(build_embed_content(pdf_path.stem, pair.parent_title_path, child.text))
    sample_texts = sample_texts[:5]

    t0 = time.perf_counter()
    try:
        vectors = await client.embed(sample_texts)
        t1 = time.perf_counter()
        print(f"  Done in {t1-t0:.2f}s")
        print(f"  Vectors: {len(vectors)}, dims: {len(vectors[0])}")
    except Exception as e:
        t1 = time.perf_counter()
        print(f"  FAILED in {t1-t0:.2f}s: {e}")
        vectors = None

    # Step 4: Query
    if vectors:
        print("\n[4/4] Query: 'Transformer的自注意力机制是什么?'")
        query_vecs = await client.embed(["Transformer的自注意力机制是什么?"])
        import numpy as np
        q = np.array(query_vecs[0], dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        mat = np.array(vectors, dtype=np.float32)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        scores = mat_norm @ q_norm

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        print(f"  Top results:")
        for idx, score in ranked[:3]:
            print(f"    [{idx}] score={score:.4f} | {sample_texts[idx][:80]}...")

    await client.close()
    print("\n=== E2E test complete ===")


if __name__ == "__main__":
    asyncio.run(main())
