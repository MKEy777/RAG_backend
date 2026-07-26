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


@router.post("/{kb_id}/rebuild")
def rebuild_graph(kb_id: str):
    """清空并从 KB 全部已解析文档重建图谱，后台线程逐文档执行（graph_status 可追踪进度）。"""
    import threading

    from sqlalchemy import select

    from ...models.database import SessionLocal
    from ...models.knowledge import Knowledge

    probe = SessionLocal()
    try:
        doc_ids = [
            r[0]
            for r in probe.execute(
                select(Knowledge.id).where(
                    Knowledge.knowledge_base_id == kb_id,
                    Knowledge.deleted_at.is_(None),
                    Knowledge.parse_status == "completed",
                )
            )
        ]
    finally:
        probe.close()

    def _run():
        import asyncio

        from ...graph.builder import get_graph_manager
        from ...logging import get_logger
        from ...models.chunk import Chunk

        log = get_logger("graph_rebuild")
        session = SessionLocal()
        gm = get_graph_manager()
        try:
            asyncio.run(gm.reset(kb_id))
            for doc_id in doc_ids:
                session.execute(
                    Knowledge.__table__.update()
                    .where(Knowledge.id == doc_id)
                    .values(graph_status="processing")
                )
                session.commit()
                try:
                    chunks = session.execute(
                        select(Chunk).where(
                            Chunk.knowledge_id == doc_id,
                            Chunk.chunk_type == "text",
                            Chunk.deleted_at.is_(None),
                        ).order_by(Chunk.start_at)
                    ).scalars().all()
                    chunk_dicts = [{"id": c.id, "content": c.content} for c in chunks]
                    if chunk_dicts:
                        asyncio.run(gm.build_document(kb_id, doc_id, chunk_dicts))
                    status = "completed"
                except Exception as e:
                    status = "failed"
                    log.error("图谱重建_文档失败", kb_id=kb_id, doc_id=doc_id, error=str(e)[:200])
                session.execute(
                    Knowledge.__table__.update()
                    .where(Knowledge.id == doc_id)
                    .values(graph_status=status)
                )
                session.commit()
            log.info("图谱重建完成", kb_id=kb_id, doc_count=len(doc_ids))
        finally:
            session.close()

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "documents": len(doc_ids)}


@router.delete("/{kb_id}")
async def delete_graph(kb_id: str):
    from ...graph.builder import get_graph_manager

    await get_graph_manager().reset(kb_id)
    return {"status": "deleted"}


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
