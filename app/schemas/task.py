"""Task: the business request a user submits. A task is executed by one or more runs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from app.core.text import normalize_text
from app.schemas.common import Metadata, RequestModel, ResponseModel

MAX_REQUEST_CHARS = 8_000
MIN_REQUEST_CHARS = 10


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    REVIEWING = "REVIEWING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})


def clean_request(value: str) -> str:
    cleaned = normalize_text(value)
    if len(cleaned) < MIN_REQUEST_CHARS:
        raise ValueError(f"request must contain at least {MIN_REQUEST_CHARS} visible characters")
    return cleaned


class TaskCreate(RequestModel):
    request: str = Field(
        min_length=MIN_REQUEST_CHARS,
        max_length=MAX_REQUEST_CHARS,
        description="The user request, in natural language.",
        examples=[
            "Analyse les documents disponibles et propose une architecture pour ajouter "
            "un système de devis à une application SaaS."
        ],
    )
    metadata: Metadata = Field(default_factory=dict)

    _clean = field_validator("request")(clean_request)


class TaskRead(ResponseModel):
    id: UUID
    status: TaskStatus
    user_request: str
    optimized_prompt: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    # The ORM attribute is ``metadata_`` (``metadata`` is reserved by SQLAlchemy).
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime
