from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session

from ..chunking import chunk_document
from ..chunking.parent_child import ParentChildPair
from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.knowledge import Knowledge
from ..settings import settings
from ..storage.factory import get_storage
from .embedder import build_embed_content, content_hash, embed_texts

log = get_logger("pipeline")


async def run_indexing_pipeline(
    session: Session,
    knowledge_id: str,
    kb_id: str,
    tenant_id: int,
    markdown: str,
    doc_title: str,
    chunking_config: dict | None = None,
    mark_completed: bool = True,
) -> None:
    config = chunking_config or {}
    strategy = config.get("strategy", "auto")
    chunk_size = config.get("chunk_size", settings.chunk_size)
    overlap = config.get("chunk_overlap", settings.chunk_overlap)
    enable_pc = config.get("enable_parent_child", True)

    log.info("索引管线启动", knowledge_id=knowledge_id, kb_id=kb_id, strategy=strategy,
             chunk_size=chunk_size, overlap=overlap)

    storage = get_storage()
    old_chunk_ids = [
        r[0]
        for r in session.query(Chunk.id)
        .filter(Chunk.knowledge_id == knowledge_id, Chunk.deleted_at.is_(None))
        .all()
    ]

    result = chunk_document(
        markdown,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        enable_parent_child=enable_pc,
    )
    chunk_count = len(result) if result else 0
    log.info("分块完成", chunk_count=chunk_count, pc_mode=enable_pc)

    new_ids: list[str] = []
    try:
        if enable_pc and result and isinstance(result[0], ParentChildPair):
            child_total = sum(len(p.children) for p in result)
            log.info("使用父子块模式", parent_count=len(result), child_count=child_total)
            new_ids = await _index_parent_child(session, result, knowledge_id, kb_id, tenant_id, doc_title, storage)
        else:
            log.info("使用平铺块模式", count=chunk_count)
            new_ids = await _index_flat(session, result, knowledge_id, kb_id, tenant_id, doc_title, storage)

        if old_chunk_ids:
            session.execute(
                update(Chunk)
                .where(Chunk.id.in_(old_chunk_ids))
                .values(deleted_at=datetime.now(timezone.utc))
            )
        if mark_completed:
            session.execute(
                update(Knowledge)
                .where(Knowledge.id == knowledge_id)
                .values(parse_status="completed", processed_at=datetime.now(timezone.utc))
            )
        session.commit()
        if old_chunk_ids:
            await storage.vector.delete_chunks(kb_id, old_chunk_ids)
            await storage.keyword.delete_chunks(kb_id, old_chunk_ids)
        log.info("索引管线完成", knowledge_id=knowledge_id, kb_id=kb_id)
    except Exception as e:
        session.rollback()
        log.error("索引管线失败_回滚", knowledge_id=knowledge_id, kb_id=kb_id, error=str(e))
        if new_ids:
            await storage.vector.delete_chunks(kb_id, new_ids)
            await storage.keyword.delete_chunks(kb_id, new_ids)
        session.execute(
            update(Knowledge)
            .where(Knowledge.id == knowledge_id)
            .values(parse_status="failed", error_message=str(e)[:2000])
        )
        session.commit()
        raise


async def _index_parent_child(
    session, pairs: list[ParentChildPair], knowledge_id, kb_id, tenant_id, doc_title, storage
) -> list[str]:
    child_texts: list[str] = []
    child_ids: list[str] = []
    pre_id: str | None = None

    for pair in pairs:
        parent_id = uuid.uuid4().hex
        parent_chunk = Chunk(
            id=parent_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=knowledge_id,
            parent_chunk_id=None,
            content=pair.parent_text,
            chunk_type="parent_text",
            start_at=pair.parent_start,
            end_at=pair.parent_end,
            content_hash=content_hash(pair.parent_text),
            metadata_={"title_path": pair.parent_title_path, "doc_title": doc_title},
        )
        session.add(parent_chunk)

        for child_span in pair.children:
            child_id = uuid.uuid4().hex
            embed_content = build_embed_content(doc_title, pair.parent_title_path, child_span.text)
            child_texts.append(embed_content)
            child_ids.append(child_id)

            child_chunk = Chunk(
                id=child_id,
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                knowledge_id=knowledge_id,
                parent_chunk_id=parent_id,
                content=child_span.text,
                chunk_type="text",
                start_at=child_span.start,
                end_at=child_span.end,
                pre_chunk_id=pre_id,
                content_hash=content_hash(child_span.text),
                metadata_={"title_path": pair.parent_title_path, "doc_title": doc_title},
            )
            if pre_id:
                prev = session.get(Chunk, pre_id)
                if prev:
                    prev.next_chunk_id = child_id
            pre_id = child_id
            session.add(child_chunk)

    session.flush()

    if child_texts:
        vectors = await embed_texts(child_texts)
        metadata_list = [
            {"knowledge_id": knowledge_id, "chunk_type": "text", "doc_title": doc_title}
            for _ in child_ids
        ]
        await storage.vector.upsert(kb_id, child_ids, vectors, metadata_list)
        await storage.keyword.index_chunks(kb_id, child_ids, child_texts, metadata_list)
    return child_ids


async def _index_flat(
    session, spans, knowledge_id, kb_id, tenant_id, doc_title, storage
) -> list[str]:
    texts: list[str] = []
    ids: list[str] = []
    pre_id: str | None = None

    for span in spans:
        chunk_id = uuid.uuid4().hex
        embed_content = build_embed_content(doc_title, span.title_path, span.text)
        texts.append(embed_content)
        ids.append(chunk_id)

        chunk = Chunk(
            id=chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=knowledge_id,
            parent_chunk_id=None,
            content=span.text,
            chunk_type="text",
            start_at=span.start,
            end_at=span.end,
            pre_chunk_id=pre_id,
            content_hash=content_hash(span.text),
            metadata_={"title_path": span.title_path, "doc_title": doc_title},
        )
        if pre_id:
            prev = session.get(Chunk, pre_id)
            if prev:
                prev.next_chunk_id = chunk_id
        pre_id = chunk_id
        session.add(chunk)

    session.flush()

    if texts:
        vectors = await embed_texts(texts)
        metadata_list = [
            {"knowledge_id": knowledge_id, "chunk_type": "text", "doc_title": doc_title}
            for _ in ids
        ]
        await storage.vector.upsert(kb_id, ids, vectors, metadata_list)
        await storage.keyword.index_chunks(kb_id, ids, texts, metadata_list)
    return ids
