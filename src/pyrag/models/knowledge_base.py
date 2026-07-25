from __future__ import annotations

from sqlalchemy import JSON, BigInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, _uuid


class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_base"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False, default="document")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunking_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    embedding_model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    image_processing_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    vlm_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    faq_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    wiki_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    question_generation_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    graph_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rerank_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    indexing_strategy: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    vector_store_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    storage_backend_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    storage_provider_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
