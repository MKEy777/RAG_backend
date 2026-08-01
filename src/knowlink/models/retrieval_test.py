from __future__ import annotations

from sqlalchemy import BigInteger, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, _uuid


class RetrievalTest(Base, TimestampMixin):
    __tablename__ = "retrieval_tests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    kb_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    params_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
