from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(
        self, kb_id: str, chunk_ids: list[str], vectors: np.ndarray, metadata: list[dict]
    ) -> None: ...

    async def search(
        self, kb_id: str, query_vector: np.ndarray, top_k: int, threshold: float = 0.0
    ) -> list[dict]: ...

    async def delete_chunks(self, kb_id: str, chunk_ids: list[str]) -> None: ...

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None: ...


@runtime_checkable
class KeywordIndex(Protocol):
    async def index_chunks(
        self, kb_id: str, chunk_ids: list[str], texts: list[str], metadata: list[dict]
    ) -> None: ...

    async def search(
        self, kb_id: str, query: str, top_k: int, threshold: float = 0.0
    ) -> list[dict]: ...

    async def delete_chunks(self, kb_id: str, chunk_ids: list[str]) -> None: ...

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None: ...

    async def rebuild(self, kb_id: str) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    async def add_graph(self, kb_id: str, knowledge_id: str, graph: dict) -> None: ...

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None: ...

    async def search_entities(
        self, kb_id: str, entity_names: list[str], top_k: int
    ) -> list[dict]: ...

    async def get_relations(
        self, kb_id: str, entity_names: list[str], hops: int = 1
    ) -> list[dict]: ...

    async def snapshot(self, kb_id: str) -> bytes | None: ...

    async def load_snapshot(self, kb_id: str, data: bytes) -> None: ...


@runtime_checkable
class CacheLock(Protocol):
    async def acquire(self, key: str, ttl_s: int = 30) -> bool: ...

    async def release(self, key: str) -> None: ...

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None: ...


@runtime_checkable
class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes) -> str: ...

    async def get(self, key: str) -> bytes | None: ...

    async def delete(self, key: str) -> None: ...

    async def url(self, key: str) -> str: ...
