from alembic import op
import sqlalchemy as sa

revision = "0006_fastpath"
down_revision = "0005_retrieval_config_chunk_enabled"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fast_path_entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("kb_id", sa.String(36), nullable=False),
        sa.Column("canonical_query", sa.String(512), nullable=False),
        sa.Column("example_queries", sa.JSON(), nullable=False),
        sa.Column("answer_template", sa.Text(), nullable=False, server_default=""),
        sa.Column("variables", sa.JSON(), nullable=True),
        sa.Column("target_chunk_id", sa.String(36), nullable=True),
        sa.Column("target_document_id", sa.String(36), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("source", sa.String(20), nullable=False, server_default="manual"),
        sa.Column("source_chunk_id", sa.String(36), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_fast_path_entries_kb_id", "fast_path_entries", ["kb_id"])
    op.create_index("ix_fast_path_entries_target_chunk_id", "fast_path_entries", ["target_chunk_id"])
    op.create_index("ix_fast_path_entries_target_document_id", "fast_path_entries", ["target_document_id"])

    op.create_table(
        "semantic_cache_entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("kb_key", sa.String(64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("references", sa.JSON(), nullable=True),
        sa.Column("source_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("source_chunk_hashes", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_semantic_cache_entries_kb_key", "semantic_cache_entries", ["kb_key"])

    op.create_table(
        "fast_path_hits",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("kb_ids", sa.JSON(), nullable=False),
        sa.Column("layer", sa.String(20), nullable=False),
        sa.Column("entry_id", sa.String(36), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_fast_path_hits_layer", "fast_path_hits", ["layer"])


def downgrade() -> None:
    op.drop_table("fast_path_hits")
    op.drop_table("semantic_cache_entries")
    op.drop_table("fast_path_entries")
