from alembic import op
import sqlalchemy as sa

revision = "0005_retrieval_config_chunk_enabled"
down_revision = "0004_kb_doc_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_base") as batch:
        batch.add_column(sa.Column("retrieval_config", sa.JSON(), nullable=True))

    with op.batch_alter_table("chunks") as batch:
        batch.add_column(sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"))

    op.create_table(
        "retrieval_tests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("kb_ids", sa.JSON(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("params_snapshot", sa.JSON(), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("top_score", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("retrieval_tests")
    with op.batch_alter_table("chunks") as batch:
        batch.drop_column("enabled")
    with op.batch_alter_table("knowledge_base") as batch:
        batch.drop_column("retrieval_config")
