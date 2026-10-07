"""Initial schema: tasks, runs, events, agent runs, knowledge base (pgvector), evaluations.

Row-level security is enabled *and forced* on every table: the application connects with a
non-superuser role that owns the tables, so FORCE makes the policies apply to it as well.
Each transaction declares its tenant with ``set_config('app.tenant_id', ..., true)``; when the
setting is missing, ``current_setting(..., true)`` is NULL and no row is visible (fail closed).

Revision ID: 0001
Revises:
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Snapshot of the values at the time of this migration (never import application enums here).
TASK_STATUSES = ("PENDING", "PLANNING", "RUNNING", "REVIEWING", "COMPLETED", "FAILED", "CANCELLED")
RUN_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")
AGENT_RUN_STATUSES = ("COMPLETED", "FAILED")
EVALUATION_STATUSES = ("RUNNING", "COMPLETED", "FAILED")
EMBEDDING_DIMENSIONS = 512

TENANT_TABLES = (
    "tasks",
    "runs",
    "run_events",
    "agent_runs",
    "documents",
    "document_chunks",
    "evaluations",
)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def _jsonb(name: str, *, nullable: bool = True, default: str | None = None) -> sa.Column:
    server_default = sa.text(default) if default else None
    return sa.Column(name, postgresql.JSONB(), nullable=nullable, server_default=server_default)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column("user_request", sa.Text(), nullable=False),
        _jsonb("optimized_prompt"),
        sa.Column("status", sa.String(20), nullable=False),
        _jsonb("result"),
        sa.Column("error", sa.Text()),
        _jsonb("metadata", nullable=False, default="'{}'::jsonb"),
        *_timestamps(),
        sa.CheckConstraint(_in("status", TASK_STATUSES), name="ck_tasks_status"),
    )
    op.create_index("ix_tasks_tenant_created", "tasks", ["tenant_id", sa.text("created_at DESC")])
    op.create_index("ix_tasks_tenant_status", "tasks", ["tenant_id", "status"])

    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column(
            "task_id", sa.Uuid(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("status", sa.String(20), nullable=False),
        _jsonb("config", nullable=False, default="'{}'::jsonb"),
        _jsonb("plan"),
        _jsonb("review"),
        _jsonb("final_result"),
        sa.Column("error", sa.Text()),
        sa.Column("step_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("agent_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("trace_id", sa.String(64)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.CheckConstraint(_in("status", RUN_STATUSES), name="ck_runs_status"),
    )
    op.create_index("ix_runs_tenant_task", "runs", ["tenant_id", "task_id"])
    op.create_index("ix_runs_tenant_created", "runs", ["tenant_id", sa.text("created_at DESC")])

    op.create_table(
        "run_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column("node", sa.String(40)),
        sa.Column("agent", sa.String(40)),
        sa.Column("step_id", sa.String(64)),
        _jsonb("payload", nullable=False, default="'{}'::jsonb"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
    )
    op.create_index("ix_run_events_tenant_type", "run_events", ["tenant_id", "event_type"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column("agent", sa.String(40), nullable=False),
        sa.Column("step_id", sa.String(64)),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(20), nullable=False),
        _jsonb("input"),
        _jsonb("output"),
        sa.Column("error", sa.Text()),
        sa.Column("model", sa.String(100)),
        sa.Column("prompt_version", sa.String(40)),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(_in("status", AGENT_RUN_STATUSES), name="ck_agent_runs_status"),
    )
    op.create_index("ix_agent_runs_run", "agent_runs", ["run_id"])
    op.create_index("ix_agent_runs_tenant_agent", "agent_runs", ["tenant_id", "agent"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("source", sa.String(500)),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        _jsonb("metadata", nullable=False, default="'{}'::jsonb"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        *_timestamps(),
        # Idempotent ingestion: the same content is stored once per tenant.
        sa.UniqueConstraint("tenant_id", "content_sha256", name="uq_documents_tenant_sha"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple', content)", persisted=True),
        ),
        _jsonb("metadata", nullable=False, default="'{}'::jsonb"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )
    op.create_index("ix_chunks_tenant_document", "document_chunks", ["tenant_id", "document_id"])
    # Approximate nearest-neighbour index for cosine distance (pgvector HNSW).
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    # Full-text side of the hybrid search.
    op.create_index(
        "ix_chunks_content_tsv", "document_chunks", ["content_tsv"], postgresql_using="gin"
    )
    # Metadata filters use JSONB containment (@>).
    op.create_index(
        "ix_chunks_metadata",
        "document_chunks",
        ["metadata"],
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )

    op.create_table(
        "evaluations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.String(63), nullable=False),
        sa.Column("dataset", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        _jsonb("config", nullable=False, default="'{}'::jsonb"),
        _jsonb("summary"),
        _jsonb("results"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(_in("status", EVALUATION_STATUSES), name="ck_evaluations_status"),
    )
    op.create_index(
        "ix_evaluations_tenant_created", "evaluations", ["tenant_id", sa.text("created_at DESC")]
    )

    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            "USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )


def downgrade() -> None:
    for table in reversed(TENANT_TABLES):
        op.drop_table(table)
