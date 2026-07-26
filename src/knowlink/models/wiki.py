from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, _uuid


# === 调度层 ===


class WikiPendingOperation(Base):
    __tablename__ = "wiki_pending_operations"

    operation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(20), nullable=False)
    dedup_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claim_batch: Mapped[str | None] = mapped_column(String(36), nullable=True)
    claim_fencing: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WikiDeadLetter(Base):
    __tablename__ = "wiki_dead_letters"

    operation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# === 内容层（写入即发布，无 release 机制，detail.md §步骤5） ===


class WikiPage(Base, TimestampMixin):
    __tablename__ = "wiki_pages"

    page_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    topic_key: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    slug: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_refs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    chunk_refs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    aliases: Mapped[list | None] = mapped_column(JSON, nullable=True)
    out_links: Mapped[list | None] = mapped_column(JSON, nullable=True)
    in_links: Mapped[list | None] = mapped_column(JSON, nullable=True)
    folder_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="published")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WikiFolder(Base, TimestampMixin):
    __tablename__ = "wiki_folders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
