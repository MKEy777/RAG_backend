from __future__ import annotations

from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_array
from ..settings import settings
from ..storage.factory import get_storage
from .hybrid import Candidate


async def extract_query_entities(query: str) -> list[str]:
    client = get_llm_client()
    prompt = (
        "Extract key entities (people, organizations, products, concepts) from the query. "
        "Return a JSON array of entity name strings only. No explanation.\n\n"
        f"Query: {query}"
    )
    try:
        resp = await client.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=256
        )
        entities = parse_json_array(resp)
        return [str(e) for e in entities if e]
    except Exception:
        return []


async def graph_boost(
    kb_id: str,
    query: str,
    existing_ids: set[str],
    top_k: int = 10,
) -> list[Candidate]:
    entities = await extract_query_entities(query)
    if not entities:
        return []

    storage = get_storage()
    try:
        relations = await storage.vector.search(kb_id, query, top_k=0)
    except Exception:
        return []

    from ..graph.builder import get_graph_manager

    gm = get_graph_manager()
    hits = await gm.search_by_entities(kb_id, entities, top_k=top_k)

    candidates = []
    for hit in hits:
        chunk_id = hit.get("chunk_id", "")
        if chunk_id and chunk_id not in existing_ids:
            candidates.append(Candidate(
                chunk_id=chunk_id,
                score=hit.get("score", 0.0),
                channel="graph",
                chunk_type="text",
                document_id=hit.get("knowledge_id", ""),
                metadata=hit,
            ))
    return candidates
