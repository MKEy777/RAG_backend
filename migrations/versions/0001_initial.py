"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-07-25
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from knowlink.models.base import Base
    from knowlink.models import (  # noqa: F401
        KnowledgeBase, Knowledge, Artifact, Chunk,
    )
    from knowlink.models.wiki import (  # noqa: F401
        WikiBuildRequest, WikiPendingOperation, WikiDeadLetter, WikiLock,
        WikiPage, WikiChunk, WikiSourceRef, WikiPageRevision,
        WikiRelease, WikiReleasePage, WikiActiveRelease,
        WikiPageLink, WikiPublishOutbox, WikiFolder,
    )

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from knowlink.models.base import Base
    from knowlink.models import (  # noqa: F401
        KnowledgeBase, Knowledge, Artifact, Chunk,
    )
    from knowlink.models.wiki import (  # noqa: F401
        WikiBuildRequest, WikiPendingOperation, WikiDeadLetter, WikiLock,
        WikiPage, WikiChunk, WikiSourceRef, WikiPageRevision,
        WikiRelease, WikiReleasePage, WikiActiveRelease,
        WikiPageLink, WikiPublishOutbox, WikiFolder,
    )

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
