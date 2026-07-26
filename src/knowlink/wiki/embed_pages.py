from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from ..indexing.embedder import content_hash, embed_texts
from ..models.chunk import Chunk
from ..models.wiki import WikiPage
from ..storage.factory import get_storage


async def embed_wiki_pages(session: Session, kb_id: str, tenant_id: int) -> None:
    pages = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.deleted_at.is_(None),
            WikiPage.status == "published",
        )
        .all()
    )
    if not pages:
        return

    storage = get_storage()
    texts: list[str] = []
    chunk_ids: list[str] = []

    for page in pages:
        if not page.content:
            continue
        existing = (
            session.query(Chunk)
            .filter(
                Chunk.knowledge_base_id == kb_id,
                Chunk.chunk_type == "wiki_page",
                Chunk.content_hash == content_hash(page.content),
                Chunk.deleted_at.is_(None),
            )
            .first()
        )
        if existing:
            continue

        chunk_id = uuid.uuid4().hex
        chunk = Chunk(
            id=chunk_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=page.page_id,
            parent_chunk_id=None,
            content=page.content,
            chunk_type="wiki_page",
            content_hash=content_hash(page.content),
            metadata_={"title_path": [page.title], "doc_title": page.title, "slug": page.slug},
        )
        session.add(chunk)
        texts.append(f"{page.title}\n{page.content}")
        chunk_ids.append(chunk_id)

    session.commit()

    if texts:
        vectors = await embed_texts(texts)
        metadata_list = [
            {"knowledge_id": kb_id, "chunk_type": "wiki_page"}
            for _ in chunk_ids
        ]
        await storage.vector.upsert(kb_id, chunk_ids, vectors, metadata_list)
        await storage.keyword.index_chunks(kb_id, chunk_ids, texts, metadata_list)
