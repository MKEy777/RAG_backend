from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_array
from ..llm.retry import wiki_llm_retry
from ..logging import get_logger
from ..models.chunk import Chunk
from ..prompts.loader import get_prompt
from ..settings import settings
from ..storage.factory import get_storage
from .embedder import content_hash, embed_texts

log = get_logger("question_gen")


async def generate_faq_chunks(
    session: Session, kb_id: str, knowledge_id: str, tenant_id: int, doc_title: str
) -> int:
    # 幂等：重建时先清掉本文档旧的自动生成问句
    stale = [
        c
        for c in session.execute(
            select(Chunk).where(
                Chunk.knowledge_id == knowledge_id,
                Chunk.chunk_type == "faq",
                Chunk.deleted_at.is_(None),
            )
        ).scalars()
        if (c.metadata_ or {}).get("generated")
    ]
    if stale:
        now = datetime.now(timezone.utc)
        for c in stale:
            c.deleted_at = now
        session.commit()
        await get_storage().vector.delete_chunks(kb_id, [c.id for c in stale])

    chunks = session.execute(
        select(Chunk).where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.chunk_type == "text",
            Chunk.deleted_at.is_(None),
        )
    ).scalars().all()
    if not chunks:
        return 0

    client = get_llm_client()
    sem = asyncio.Semaphore(settings.faq_question_concurrency)
    system = get_prompt("faq", "faq_question_generation", {"DocTitle": doc_title})

    async def _gen(chunk: Chunk) -> list[tuple[str, str]]:
        async with sem:
            try:
                @wiki_llm_retry()
                async def _call():
                    return await client.chat(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": chunk.content},
                        ],
                        temperature=0.3,
                        max_tokens=1024,
                        purpose="faq:question_gen",
                    )

                questions = parse_json_array(await _call())
                return [(chunk.id, q) for q in questions if isinstance(q, str) and q.strip()]
            except Exception:
                return []

    results = await asyncio.gather(*[_gen(c) for c in chunks])

    texts: list[str] = []
    ids: list[str] = []
    for pairs in results:
        for source_chunk_id, question in pairs:
            cid = uuid.uuid4().hex
            session.add(Chunk(
                id=cid, tenant_id=tenant_id, knowledge_base_id=kb_id,
                knowledge_id=knowledge_id, content=question, chunk_type="faq",
                content_hash=content_hash(question),
                metadata_={"source_chunk_id": source_chunk_id, "generated": True},
            ))
            texts.append(question)
            ids.append(cid)
    session.commit()

    if texts:
        storage = get_storage()
        vectors = await embed_texts(texts)
        metadata = [{"knowledge_id": knowledge_id, "chunk_type": "faq"} for _ in ids]
        await storage.vector.upsert(kb_id, ids, vectors, metadata)
        log.info("FAQ问句生成完成", kb_id=kb_id, knowledge_id=knowledge_id, count=len(ids))
    return len(ids)
