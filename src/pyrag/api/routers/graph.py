from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from ...settings import settings

router = APIRouter()


class GraphNode(BaseModel):
    name: str
    type: str
    description: str
    frequency: int
    degree: int


class GraphEdge(BaseModel):
    source: str
    target: str
    description: str
    weight: float


class GraphStats(BaseModel):
    total_entities: int
    total_relations: int
    returned_nodes: int
    truncated: bool


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: GraphStats


@router.get("/{kb_id}")
def get_graph(kb_id: str, limit: int = 300):
    snapshot_path = settings.object_storage_root / "graphs" / f"{kb_id}.json"

    if not snapshot_path.exists():
        return GraphResponse(
            nodes=[],
            edges=[],
            stats=GraphStats(
                total_entities=0,
                total_relations=0,
                returned_nodes=0,
                truncated=False,
            ),
        )

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return GraphResponse(
            nodes=[],
            edges=[],
            stats=GraphStats(
                total_entities=0,
                total_relations=0,
                returned_nodes=0,
                truncated=False,
            ),
        )

    raw_entities = data.get("entities", {})
    raw_relations = data.get("relations", [])

    cap = min(max(limit, 1), 1000)
    sorted_entities = sorted(
        raw_entities.values(),
        key=lambda e: e.get("frequency", 0),
        reverse=True,
    )
    truncated = len(sorted_entities) > cap
    selected = sorted_entities[:cap]
    selected_names = {e["name"] for e in selected}

    nodes = [
        GraphNode(
            name=e["name"],
            type=e.get("type", ""),
            description=e.get("description", ""),
            frequency=e.get("frequency", 0),
            degree=e.get("in_degree", 0) + e.get("out_degree", 0),
        )
        for e in selected
    ]

    edges = [
        GraphEdge(
            source=r["source"],
            target=r["target"],
            description=r.get("description", ""),
            weight=r.get("weight", 1.0),
        )
        for r in raw_relations
        if r.get("source") in selected_names and r.get("target") in selected_names
    ]

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        stats=GraphStats(
            total_entities=len(raw_entities),
            total_relations=len(raw_relations),
            returned_nodes=len(nodes),
            truncated=truncated,
        ),
    )
