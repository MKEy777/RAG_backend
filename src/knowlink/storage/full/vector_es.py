from __future__ import annotations

import numpy as np


class ESVectorStore:
    """Elasticsearch 8 dense_vector (cosine + HNSW) implementation. Placeholder."""

    async def upsert(
        self, kb_id: str, chunk_ids: list[str], vectors: np.ndarray, metadata: list[dict]
    ) -> None:
        raise NotImplementedError("Implement ESVectorStore.upsert")

    async def search(
        self, kb_id: str, query_vector: np.ndarray, top_k: int, threshold: float = 0.0
    ) -> list[dict]:
        raise NotImplementedError("Implement ESVectorStore.search")

    async def delete_chunks(self, kb_id: str, chunk_ids: list[str]) -> None:
        raise NotImplementedError("Implement ESVectorStore.delete_chunks")

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None:
        raise NotImplementedError("Implement ESVectorStore.delete_document")
