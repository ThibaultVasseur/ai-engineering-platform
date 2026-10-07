"""Run: one execution of the multi-agent graph for a task."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import RequestModel
from app.schemas.knowledge import SourceRef
from app.schemas.review import FinalDraft, Review
from app.schemas.task import MAX_REQUEST_CHARS, clean_request


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_RUN_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


class FinalStatus(StrEnum):
    COMPLETED = "completed"  # the critic accepted the answer
    FAILED = "failed"  # limits reached, a step failed, or the critic kept rejecting
    REJECTED = "rejected"  # the request was refused by the input guardrail / optimizer


class StepSummary(BaseModel):
    step_id: str
    title: str
    agent: str | None
    status: str
    attempts: int
    summary: str = ""


class RunMetrics(BaseModel):
    step_count: int = 0
    agent_calls: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0


class FinalResult(BaseModel):
    status: FinalStatus
    accepted: bool
    answer: FinalDraft | None = None
    review: Review | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    steps: list[StepSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    abort_reason: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
class RunOptions(RequestModel):
    allow_tool_writes: bool = Field(
        default=False,
        description="Explicit opt-in for WRITE tools (database_write). Off by default.",
    )
    supervisor_strategy: Literal["rules", "llm"] | None = Field(
        default=None, description="Override the configured supervisor strategy for this run."
    )


class RunCreate(RequestModel):
    task_id: UUID | None = Field(default=None, description="Execute an existing task...")
    request: str | None = Field(
        default=None,
        min_length=10,
        max_length=MAX_REQUEST_CHARS,
        description="...or create a task from this request and execute it.",
    )
    wait: bool = Field(
        default=False,
        description="Block until the run finishes (bounded by MAX_RUN_SECONDS). "
        "By default the run executes in the background: poll GET /runs/{id}.",
    )
    options: RunOptions = Field(default_factory=RunOptions)

    @field_validator("request")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        return clean_request(value) if value is not None else None

    @model_validator(mode="after")
    def _exactly_one_target(self) -> RunCreate:
        if (self.task_id is None) == (self.request is None):
            raise ValueError("provide exactly one of 'task_id' or 'request'")
        return self


class RunMetricsRead(BaseModel):
    step_count: int
    agent_calls: int
    retry_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class RunRead(BaseModel):
    id: UUID
    task_id: UUID
    status: RunStatus
    config: dict[str, Any]
    plan: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None
    error: str | None = None
    metrics: RunMetricsRead
    trace_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunEventRead(BaseModel):
    sequence: int
    event_type: str
    node: str | None
    agent: str | None
    step_id: str | None
    payload: dict[str, Any]
    created_at: datetime


class RunEventPage(BaseModel):
    items: list[RunEventRead]
    next_after: int = Field(description="Pass as ?after= to fetch the following events.")
    run_status: RunStatus
