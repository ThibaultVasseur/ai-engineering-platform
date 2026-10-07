"""Run tracing: one typed event stream, several sinks.

Agents, tools and the graph only know the ``Tracer`` protocol. Where events go is decided by
the composition root: the ``run_events`` table (always), structured logs, Langfuse (optional),
or memory (tests, CLI). A failing sink is logged and ignored — observability must never break
a run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.core.logging import log_event

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    NODE_COMPLETED = "node_completed"
    PROMPT_OPTIMIZED = "prompt_optimized"
    PLAN_CREATED = "plan_created"
    SUPERVISOR_DECISION = "supervisor_decision"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    LLM_CALL = "llm_call"
    STRUCTURED_OUTPUT_REPAIRED = "structured_output_repaired"
    TOOL_CALL = "tool_call"
    TOOL_DENIED = "tool_denied"
    RETRIEVAL = "retrieval"
    CRITIC_VERDICT = "critic_verdict"
    REWORK_REQUESTED = "rework_requested"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    LIMIT_REACHED = "limit_reached"


def utcnow() -> datetime:
    return datetime.now(UTC)


class TraceEvent(BaseModel):
    type: EventType
    timestamp: datetime = Field(default_factory=utcnow)
    node: str | None = None
    agent: str | None = None
    step_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Tracer(Protocol):
    async def emit(self, event: TraceEvent) -> None: ...


class NullTracer:
    async def emit(self, event: TraceEvent) -> None:
        return None


class InMemoryTracer:
    """Keeps events in order; used by tests, the evaluation runner and the CLI."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: EventType) -> list[TraceEvent]:
        return [event for event in self.events if event.type is event_type]


class LoggingTracer:
    """Mirrors every event as a structured log line (JSON in containers)."""

    def __init__(self, level: int = logging.INFO) -> None:
        self._level = level

    async def emit(self, event: TraceEvent) -> None:
        log_event(
            logger,
            event.type.value,
            level=self._level,
            node=event.node,
            agent=event.agent,
            step_id=event.step_id,
            **{f"data_{key}": value for key, value in event.payload.items()},
        )


class CompositeTracer:
    def __init__(self, tracers: Sequence[Tracer]) -> None:
        self._tracers = list(tracers)

    async def emit(self, event: TraceEvent) -> None:
        for tracer in self._tracers:
            try:
                await tracer.emit(event)
            except Exception as exc:  # a broken sink must not break the run
                log_event(
                    logger,
                    "tracer_failed",
                    level=logging.WARNING,
                    tracer=type(tracer).__name__,
                    error_type=type(exc).__name__,
                )
