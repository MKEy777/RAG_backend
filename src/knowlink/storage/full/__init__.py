from __future__ import annotations

from ..factory import StorageBundle


def create_full_storage() -> StorageBundle:
    raise NotImplementedError(
        "Full storage profile (MySQL + Elasticsearch + Neo4j + Redis) is not yet implemented. "
        "Implement vector_es.py, keyword_es.py, graph_neo4j.py, object_s3.py in storage/full/."
    )
