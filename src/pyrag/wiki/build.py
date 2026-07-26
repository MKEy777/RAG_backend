from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.knowledge import Knowledge
from ..models.knowledge_base import KnowledgeBase
from ..models.wiki import WikiPendingOperation
from ..settings import settings
from .embed_pages import embed_wiki_pages
from .finalize import finalize
from .ingest import map_from_text
from .merge import reduce_pages
from .queue import enqueue_ingest, mark_completed, mark_failed, peek_batch

log = get_logger("wiki_build")


def run_wiki_build(session: Session, kb_id: str) -> int:
    kb = session.get(KnowledgeBase, kb_id)
    tenant_id = kb.tenant_id if kb else 1

    processed_op = exists().where(
        and_(
            WikiPendingOperation.document_id == Knowledge.id,
            WikiPendingOperation.knowledge_base_id == kb_id,
            WikiPendingOperation.completed_at.is_not(None),
        )
    )
    docs = (
        session.query(Knowledge)
        .filter(
            Knowledge.knowledge_base_id == kb_id,
            Knowledge.parse_status == "completed",
            Knowledge.deleted_at.is_(None),
            ~processed_op,
        )
        .all()
    )
    for doc in docs:
        enqueue_ingest(session, kb_id, doc.id)

    session.query(WikiPendingOperation).filter(
        WikiPendingOperation.knowledge_base_id == kb_id,
        WikiPendingOperation.completed_at.is_(None),
    ).update({"available_at": datetime.now(timezone.utc)})
    session.commit()

    attempted: set[str] = set()
    all_ops = []
    while True:
        ops = [op for op in peek_batch(session, kb_id) if op.operation_id not in attempted]
        if not ops:
            break
        attempted.update(op.operation_id for op in ops)
        all_ops.extend(ops)

    if not all_ops:
        asyncio.run(_finalize_and_embed(session, kb_id, tenant_id))
        return 0

    processed = asyncio.run(_wiki_build_all(session, all_ops, kb_id, tenant_id))
    log.info("Wiki构建完成", kb_id=kb_id, processed=processed)
    return processed


async def _finalize_and_embed(session: Session, kb_id: str, tenant_id: int) -> None:
    await finalize(session, kb_id, tenant_id)
    await embed_wiki_pages(session, kb_id, tenant_id)


async def _wiki_build_all(
    session: Session, ops: list, kb_id: str, tenant_id: int
) -> int:
    processed = await _wiki_build_async(session, ops, kb_id, tenant_id)
    await finalize(session, kb_id, tenant_id)
    await embed_wiki_pages(session, kb_id, tenant_id)
    return processed


async def _wiki_build_async(
    session: Session, ops: list, kb_id: str, tenant_id: int
) -> int:
    sem = asyncio.Semaphore(settings.wiki_map_concurrency)

    kb = session.get(KnowledgeBase, kb_id)
    granularity = ((kb.wiki_config or {}).get("extraction_granularity") or "Standard") if kb else "Standard"

    # 预加载切块（串行 DB 读，毫秒级）
    doc_data: list[tuple] = []
    for op in ops:
        doc = session.get(Knowledge, op.document_id)
        doc_title = (doc.title or doc.file_name or op.document_id) if doc else op.document_id
        chunks = session.execute(
            select(Chunk)
            .where(
                Chunk.knowledge_id == op.document_id,
                Chunk.chunk_type == "text",
                Chunk.deleted_at.is_(None),
            )
            .order_by(Chunk.start_at)
        ).scalars().all()
        if not chunks:
            mark_completed(session, op)
            continue
        source_text = "\n".join(c.content for c in chunks)
        if not source_text.strip():
            mark_completed(session, op)
            continue
        chunk_dicts = [{"id": c.id, "content": c.content[:500]} for c in chunks]
        doc_data.append((op, doc_title, source_text, chunk_dicts))

    # Phase 1: 跨文档并发 map（纯 LLM，无 DB 访问）
    async def _map_one(item: tuple):
        op, doc_title, source_text, chunk_dicts = item
        async with sem:
            try:
                result = await map_from_text(source_text, chunk_dicts, kb_id, op.document_id, granularity)
                return (op, doc_title, result, None)
            except Exception as e:
                return (op, doc_title, None, e)

    map_results = await asyncio.gather(*[_map_one(item) for item in doc_data])

    # Phase 2: 串行 reduce（写共享 wiki 页面）
    processed = 0
    for op, doc_title, result, error in map_results:
        if error:
            log.error("Wiki构建_文档处理失败", kb_id=kb_id, document_id=op.document_id, error=str(error)[:200])
            mark_failed(session, op, str(error))
            continue
        try:
            if result:
                await reduce_pages(session, kb_id, tenant_id, op.document_id, doc_title, result)
            mark_completed(session, op)
            processed += 1
        except Exception as e:
            log.error("Wiki构建_文档处理失败", kb_id=kb_id, document_id=op.document_id, error=str(e)[:200])
            mark_failed(session, op, str(e))

    return processed
