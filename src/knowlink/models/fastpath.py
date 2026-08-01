from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, SoftDeleteMixin, TimestampMixin, _uuid


class FastPathEntry(Base, TimestampMixin, SoftDeleteMixin):
    """快路径条目：问法集 + 目标定位 + 答案模板（含 {var} 变量占位）。"""

    __tablename__ = "fast_path_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    kb_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    canonical_query: Mapped[str] = mapped_column(String(512), nullable=False)
    example_queries: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    answer_template: Mapped[str] = mapped_column(Text, nullable=False, default="")
    variables: Mapped[list | None] = mapped_column(JSON, nullable=True)
    target_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    target_document_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="manual")
    source_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SemanticCacheEntry(Base, TimestampMixin):
    """语义缓存条目：历史问答对，向量比对命中后校验源 chunk 未变则复用。"""

    __tablename__ = "semantic_cache_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    kb_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    references: Mapped[list | None] = mapped_column(JSON, nullable=True)
    source_chunk_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    source_chunk_hashes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class FastPathHit(Base):
    """快路径命中日志：命中率看板 + 离线挖掘数据源。"""

    __tablename__ = "fast_path_hits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    kb_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    layer: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
