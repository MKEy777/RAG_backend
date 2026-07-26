"""drop unused wiki tables

Revision ID: 4cc06c3b8b09
Revises: 0001
Create Date: 2026-07-26 20:16:46.510982
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = '4cc06c3b8b09'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in (
        "wiki_build_requests", "wiki_locks", "wiki_chunks", "wiki_source_refs",
        "wiki_page_revisions", "wiki_releases", "wiki_release_pages",
        "wiki_active_releases", "wiki_page_links", "wiki_publish_outbox",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")


def downgrade() -> None:
    pass  # 数据本就未使用，不提供恢复
