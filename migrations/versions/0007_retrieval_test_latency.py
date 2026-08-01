from alembic import op
import sqlalchemy as sa

revision = "0007_retrieval_test_latency"
down_revision = "0006_fastpath"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("retrieval_tests", sa.Column("latency_p50_ms", sa.Float(), nullable=True))
    op.add_column("retrieval_tests", sa.Column("latency_p99_ms", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("retrieval_tests", "latency_p99_ms")
    op.drop_column("retrieval_tests", "latency_p50_ms")
