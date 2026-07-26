from __future__ import annotations


class Neo4jGraphStore:
    """Neo4j graph store implementation. Placeholder."""

    async def add_graph(self, kb_id: str, knowledge_id: str, graph: dict) -> None:
        raise NotImplementedError("Implement Neo4jGraphStore.add_graph")

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None:
        raise NotImplementedError("Implement Neo4jGraphStore.delete_document")

    async def search_entities(
        self, kb_id: str, entity_names: list[str], top_k: int
    ) -> list[dict]:
        raise NotImplementedError("Implement Neo4jGraphStore.search_entities")

    async def get_relations(
        self, kb_id: str, entity_names: list[str], hops: int = 1
    ) -> list[dict]:
        raise NotImplementedError("Implement Neo4jGraphStore.get_relations")

    async def snapshot(self, kb_id: str) -> bytes | None:
        raise NotImplementedError("Implement Neo4jGraphStore.snapshot")

    async def load_snapshot(self, kb_id: str, data: bytes) -> None:
        raise NotImplementedError("Implement Neo4jGraphStore.load_snapshot")
