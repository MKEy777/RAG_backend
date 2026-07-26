from __future__ import annotations


class ESKeywordIndex:
    """Elasticsearch 8 BM25 (content field text mapping) implementation. Placeholder."""

    async def index_chunks(
        self, kb_id: str, chunk_ids: list[str], texts: list[str], metadata: list[dict]
    ) -> None:
        raise NotImplementedError("Implement ESKeywordIndex.index_chunks")

    async def search(
        self, kb_id: str, query: str, top_k: int, threshold: float = 0.0
    ) -> list[dict]:
        raise NotImplementedError("Implement ESKeywordIndex.search")

    async def delete_chunks(self, kb_id: str, chunk_ids: list[str]) -> None:
        raise NotImplementedError("Implement ESKeywordIndex.delete_chunks")

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None:
        raise NotImplementedError("Implement ESKeywordIndex.delete_document")

    async def rebuild(self, kb_id: str) -> None:
        raise NotImplementedError("Implement ESKeywordIndex.rebuild")
