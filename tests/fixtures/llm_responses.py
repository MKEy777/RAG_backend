"""Mock LLM responses for integration testing without a real LLM endpoint."""
from __future__ import annotations

import json

EMBEDDING_DIM = 2560


def mock_embedding_response(texts: list[str]) -> dict:
    import hashlib

    data = []
    for i, text in enumerate(texts):
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        vec = [(seed >> (j % 32) & 1) * 0.01 - 0.005 for j in range(EMBEDDING_DIM)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vec = [v / norm for v in vec]
        data.append({"embedding": vec, "index": i, "object": "embedding"})
    return {"data": data, "model": "mock-embed", "object": "list", "usage": {"prompt_tokens": 0, "total_tokens": 0}}


def mock_chat_response(messages: list[dict]) -> dict:
    last_msg = messages[-1]["content"] if messages else ""
    if isinstance(last_msg, list):
        last_msg = str(last_msg)

    if "entity" in last_msg.lower() or "extract" in last_msg.lower():
        content = json.dumps([
            {"title": "RAG", "type": "Concept", "description": "Retrieval-Augmented Generation"},
            {"title": "BM25", "type": "Concept", "description": "A keyword retrieval algorithm"},
        ])
    elif "SUMMARY" in last_msg or "wiki" in last_msg.lower():
        content = "SUMMARY: A document about RAG systems.\n\n## Overview\n\nThis document covers RAG."
    elif "question" in last_msg.lower() or "generate" in last_msg.lower():
        content = json.dumps(["What is RAG?", "How does BM25 work?"])
    else:
        content = "Based on the provided context, RAG combines retrieval with generation."

    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "index": 0, "finish_reason": "stop"}],
        "model": "mock-chat",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def mock_rerank_response(query: str, documents: list[str]) -> dict:
    results = []
    for i in range(len(documents)):
        score = max(0.1, 1.0 - i * 0.15)
        results.append({"index": i, "relevance_score": score})
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"results": results, "model": "mock-rerank"}
