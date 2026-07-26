from __future__ import annotations

from sqlalchemy.orm import Session

from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.knowledge import Knowledge
from ..models.knowledge_base import KnowledgeBase
from ..storage.factory import get_storage
from .embedder import build_embed_content

log = get_logger("reconcile")


def _live_index_entries(session: Session, kb_id: str) -> tuple[dict[str, str], dict[str, dict]]:
    doc_deleted = {
        k.id: k.deleted_at is not None
        for k in session.query(Knowledge).filter(Knowledge.knowledge_base_id == kb_id).all()
    }
    chunks = (
        session.query(Chunk)
        .filter(Chunk.knowledge_base_id == kb_id, Chunk.deleted_at.is_(None))
        .all()
    )

    live_texts: dict[str, str] = {}
    live_meta: dict[str, dict] = {}
    for c in chunks:
        # wiki_page/faq 的 knowledge_id 不是 Knowledge 行，缺省视为存活
        if doc_deleted.get(c.knowledge_id, False):
            continue
        if c.chunk_type == "parent_text":
            continue
        meta = c.metadata_ or {}
        doc_title = meta.get("doc_title", "")
        if c.chunk_type == "text":
            live_texts[c.id] = build_embed_content(doc_title, meta.get("title_path", []), c.content)
            live_meta[c.id] = {"knowledge_id": c.knowledge_id, "chunk_type": "text", "doc_title": doc_title}
        else:
            live_texts[c.id] = f"{doc_title}\n{c.content}" if doc_title else c.content
            live_meta[c.id] = {
                "knowledge_id": kb_id if c.chunk_type == "wiki_page" else c.knowledge_id,
                "chunk_type": c.chunk_type,
            }
    return live_texts, live_meta


def live_text_chunk_ids(session: Session, kb_id: str) -> set[str]:
    rows = (
        session.query(Chunk.id)
        .join(Knowledge, Knowledge.id == Chunk.knowledge_id)
        .filter(
            Chunk.knowledge_base_id == kb_id,
            Chunk.chunk_type == "text",
            Chunk.deleted_at.is_(None),
            Knowledge.deleted_at.is_(None),
        )
        .all()
    )
    return {r[0] for r in rows}


async def reconcile_kb_indexes(session: Session, kb_id: str) -> dict:
    storage = get_storage()
    # full profile（ES 自带按文档删除）无需 lite 对账
    if not hasattr(storage.vector, "reconcile") or not hasattr(storage.keyword, "reconcile"):
        return {}

    from ..graph.builder import get_graph_manager

    live_texts, live_meta = _live_index_entries(session, kb_id)
    vec_removed = await storage.vector.reconcile(kb_id, live_meta)
    kw_removed, kw_added = await storage.keyword.reconcile(kb_id, live_texts, live_meta)
    graph_removed = await get_graph_manager().reconcile(kb_id, set(live_texts.keys()))
    return {
        "kb_id": kb_id,
        "live": len(live_texts),
        "vector_removed": vec_removed,
        "keyword_removed": kw_removed,
        "keyword_added": kw_added,
        "graph_removed": graph_removed,
    }


async def reconcile_all_indexes(session: Session) -> list[dict]:
    kb_ids = [kb.id for kb in session.query(KnowledgeBase).all()]
    results = []
    for kb_id in kb_ids:
        try:
            result = await reconcile_kb_indexes(session, kb_id)
            if result:
                results.append(result)
        except Exception as e:
            log.warning("索引对账失败_跳过", kb_id=kb_id, error=str(e))
    log.info("索引对账完成", kb_count=len(results))
    return results
