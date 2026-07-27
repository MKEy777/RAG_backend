from alembic import op
import sqlalchemy as sa

revision = "0004_kb_doc_metadata"
down_revision = "0003_chat_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_base") as batch:
        batch.add_column(sa.Column("group_name", sa.String(100), nullable=True))
        batch.add_column(sa.Column("icon", sa.String(100), nullable=True))
    op.create_index("ix_knowledge_base_group_name", "knowledge_base", ["group_name"])

    with op.batch_alter_table("knowledge") as batch:
        batch.add_column(sa.Column("tags", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("uploaded_by", sa.String(100), nullable=True))
        batch.add_column(sa.Column("authority", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_index("ix_knowledge_base_group_name", "knowledge_base")
    with op.batch_alter_table("knowledge_base") as batch:
        batch.drop_column("group_name")
        batch.drop_column("icon")
    with op.batch_alter_table("knowledge") as batch:
        batch.drop_column("tags")
        batch.drop_column("uploaded_by")
        batch.drop_column("authority")
