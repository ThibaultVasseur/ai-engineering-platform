"""SQLAlchemy ORM models.

Conventions:
* every table carries ``tenant_id``: repositories filter on it explicitly *and* PostgreSQL
  row-level security enforces it again (see migration 0001) — defence in depth;
* free-form structures produced by agents (plans, reviews, results) are JSONB documents
  validated by Pydantic at the application boundary;
* the migration is the source of truth for DDL; these models mirror it for querying.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import EMBEDDING_DIMENSIONS

JSONDict = dict[str, Any]


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {JSONDict: JSONB, list[JSONDict]: JSONB}


def _tenant_column() -> Mapped[str]:
    return mapped_column(String(63), nullable=False)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _updated_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class TaskModel(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = _tenant_column()
    user_request: Mapped[str] = mapped_column(Text, nullable=False)
    optimized_prompt: Mapped[JSONDict | None]
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    result: Mapped[JSONDict | None]
    error: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[JSONDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class RunModel(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = _tenant_column()
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[JSONDict] = mapped_column(JSONB, nullable=False, default=dict)
    plan: Mapped[JSONDict | None]
    review: Mapped[JSONDict | None]
    final_result: Mapped[JSONDict | None]
    error: Mapped[str | None] = mapped_column(Text)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    agent_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(
        Numeric(12, 6, asdecimal=False), nullable=False, default=0.0
    )
    trace_id: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class RunEventModel(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = _tenant_column()
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    node: Mapped[str | None] = mapped_column(String(40))
    agent: Mapped[str | None] = mapped_column(String(40))
    step_id: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[JSONDict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created_at()


class AgentRunModel(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = _tenant_column()
    agent: Mapped[str] = mapped_column(String(40), nullable=False)
    step_id: Mapped[str | None] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    input: Mapped[JSONDict | None]
    output: Mapped[JSONDict | None]
    error: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _created_at()


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = _tenant_column()
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source: Mapped[str | None] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_: Mapped[JSONDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = _tenant_column()
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple', content)", persisted=True)
    )
    metadata_: Mapped[JSONDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _created_at()


class EvaluationModel(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = _tenant_column()
    dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[JSONDict] = mapped_column(JSONB, nullable=False, default=dict)
    summary: Mapped[JSONDict | None]
    results: Mapped[list[JSONDict] | None]
    created_at: Mapped[datetime] = _created_at()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


TENANT_SCOPED_TABLES: tuple[str, ...] = (
    "tasks",
    "runs",
    "run_events",
    "agent_runs",
    "documents",
    "document_chunks",
    "evaluations",
)
