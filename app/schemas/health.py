from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    database: Literal["ok", "unavailable"]
    llm_provider: str
    llm_model: str
    embedding_provider: str
    langfuse_enabled: bool
