from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, _uuid


# === 调度层 ===


class WikiBuildRequest(Base):
    __tablename__ = "wiki_build_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    build_input_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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


class WikiLock(Base):
    __tablename__ = "wiki_locks"

    lock_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# === 内容层 ===


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


class WikiChunk(Base):
    __tablename__ = "wiki_chunks"

    wiki_chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    page_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    paragraph_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class WikiSourceRef(Base):
    __tablename__ = "wiki_source_refs"

    ref_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    revision_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    wiki_chunk_id: Mapped[str] = mapped_column(String(36), nullable=False)
    raw_document_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    raw_document_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    raw_child_id: Mapped[str] = mapped_column(String(36), nullable=False)
    raw_parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    raw_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class WikiPageRevision(Base):
    __tablename__ = "wiki_page_revisions"

    revision_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    page_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# === 发布层 ===


class WikiRelease(Base):
    __tablename__ = "wiki_releases"

    release_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    fencing_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WikiReleasePage(Base):
    __tablename__ = "wiki_release_pages"

    release_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision_id: Mapped[str] = mapped_column(String(36), nullable=False)


class WikiActiveRelease(Base):
    __tablename__ = "wiki_active_releases"

    tenant_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    release_id: Mapped[str] = mapped_column(String(36), nullable=False)


class WikiPageLink(Base):
    __tablename__ = "wiki_page_links"

    revision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target_page_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    relation: Mapped[str | None] = mapped_column(String(50), nullable=True)


class WikiPublishOutbox(Base):
    __tablename__ = "wiki_publish_outbox"

    outbox_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    release_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WikiFolder(Base, TimestampMixin):
    __tablename__ = "wiki_folders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
