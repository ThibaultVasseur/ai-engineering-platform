"""Persists the event stream of a run (``run_events``) and one row per agent invocation
(``agent_runs``). This is the local, always-on observability backend: it powers
``GET /api/v1/runs/{id}/events`` and works without any third-party service."""

from __future__ import annotations

import uuid
from typing import Any

from app.core.logging import redact
from app.db.database import Database
from app.db.repositories.runs import AgentRunRepository, RunEventRepository
from app.observability.tracing import EventType, TraceEvent

_AGENT_RUN_EVENTS = {EventType.AGENT_COMPLETED: "COMPLETED", EventType.AGENT_FAILED: "FAILED"}
MAX_PAYLOAD_CHARS = 20_000


def _bounded(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets; replace oversized values (e.g. full outputs) by a preview."""
    safe: dict[str, Any] = redact(payload)
    for key, value in list(safe.items()):
        if len(repr(value)) > MAX_PAYLOAD_CHARS:
            safe[key] = {"truncated": True, "preview": repr(value)[:2_000]}
    return safe


class DatabaseTracer:
    def __init__(self, database: Database, *, tenant_id: str, run_id: uuid.UUID) -> None:
        self._db = database
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._sequence = 0

    async def emit(self, event: TraceEvent) -> None:
        self._sequence += 1
        payload = _bounded(event.payload)
        async with self._db.session(self._tenant_id) as session:
            await RunEventRepository(session, self._tenant_id).add(
                self._run_id,
                sequence=self._sequence,
                event_type=event.type.value,
                node=event.node,
                agent=event.agent,
                step_id=event.step_id,
                payload=payload,
            )
            status = _AGENT_RUN_EVENTS.get(event.type)
            if status is not None and event.agent:
                await AgentRunRepository(session, self._tenant_id).add(
                    self._run_id,
                    agent=event.agent,
                    step_id=event.step_id,
                    attempt=int(payload.get("attempt", 1)),
                    status=status,
                    output=payload.get("output")
                    if isinstance(payload.get("output"), dict)
                    else None,
                    error=payload.get("error"),
                    model=payload.get("model"),
                    prompt_version=payload.get("prompt_version"),
                    input_tokens=int(payload.get("input_tokens") or 0),
                    output_tokens=int(payload.get("output_tokens") or 0),
                    latency_ms=int(payload.get("latency_ms") or 0),
                )
