"""Tool call records (what was called, by whom, with which outcome)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolPermission(StrEnum):
    READ = "read"
    WRITE = "write"  # requires an explicit per-run opt-in (allow_tool_writes)


class ToolCallStatus(StrEnum):
    OK = "ok"
    ERROR = "error"  # invalid arguments, timeout, handler failure
    DENIED = "denied"  # unknown tool, not allowed for the agent, write not authorised


class ToolCallRecord(BaseModel):
    tool: str
    agent: str
    step_id: str | None = None
    status: ToolCallStatus
    arguments: dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""
    error: str | None = None
    latency_ms: float = 0.0


class ToolInfo(BaseModel):
    name: str
    description: str
    permission: ToolPermission
    timeout_seconds: float
    input_schema: dict[str, Any]
