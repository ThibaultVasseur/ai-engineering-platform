from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import RequestModel


class EvaluationRequest(RequestModel):
    dataset: str = Field(default="default", max_length=100)
    case_ids: list[str] = Field(
        default_factory=list, max_length=50, description="Subset of cases (all when empty)."
    )


class EvaluationRead(BaseModel):
    id: UUID
    dataset: str
    status: str
    config: dict[str, Any]
    summary: dict[str, Any] | None = None
    results: list[dict[str, Any]] | None = None
    created_at: datetime
    finished_at: datetime | None = None
