from __future__ import annotations

from sqlalchemy import BigInteger, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, SoftDeleteMixin, TimestampMixin, _uuid


class Chunk(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    knowledge_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pre_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    next_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    relation_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    indirect_relation_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    image_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
