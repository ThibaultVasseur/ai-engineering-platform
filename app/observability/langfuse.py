"""Optional Langfuse exporter.

Maps the run's event stream onto Langfuse observations — one trace per run:

    run (chain)
    ├── agent:<name> (agent)          one per agent invocation
    │   ├── llm:<prompt> (generation) model, tokens, cost, prompt version
    │   ├── tool:<name> (tool)        arguments, output preview, denials as warnings
    │   ├── retrieval (retriever)     query, sources, chunks filtered by the guardrail
    │   └── <guardrail> (guardrail)
    ├── critic (evaluator) + trace score "critic_score"
    └── events: prompt_optimized, plan_created, supervisor_decision, rework_requested...

Langfuse is never required: without keys (or without the ``langfuse`` package) no factory is
built and runs are traced locally only. Every payload goes through the same redaction as logs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from app import __version__
from app.core.config import Settings
from app.core.logging import log_event, redact
from app.observability.tracing import EventType, TraceEvent

logger = logging.getLogger(__name__)

_LEVELS = {"error": "ERROR", "denied": "WARNING"}
_EVENT_ONLY = {
    EventType.PROMPT_OPTIMIZED,
    EventType.PLAN_CREATED,
    EventType.SUPERVISOR_DECISION,
    EventType.REWORK_REQUESTED,
    EventType.LIMIT_REACHED,
    EventType.STRUCTURED_OUTPUT_REPAIRED,
}
_RUN_END = {EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_CANCELLED}


def mask(*, data: Any, **_: Any) -> Any:
    """Langfuse ``mask`` hook: the log redaction applied to everything exported."""
    return redact(data)


class LangfuseTracer:
    def __init__(self, client: Any, *, run_id: uuid.UUID, tenant_id: str) -> None:
        self._client = client
        self._run_id = run_id
        self._tenant_id = tenant_id
        self.trace_id: str = client.create_trace_id(seed=str(run_id))
        self._root: Any = None
        self._agents: dict[tuple[str | None, str | None], Any] = {}

    def _parent(self, event: TraceEvent) -> Any:
        return self._agents.get((event.agent, event.step_id)) or self._ensure_root()

    def _ensure_root(self) -> Any:
        if self._root is None:
            self._root = self._client.start_observation(
                trace_context={"trace_id": self.trace_id},
                name="run",
                as_type="chain",
                metadata={"run_id": str(self._run_id), "tenant_id": self._tenant_id},
            )
        return self._root

    async def emit(self, event: TraceEvent) -> None:
        payload = event.payload
        kind = event.type
        if kind is EventType.RUN_STARTED:
            root = self._ensure_root()
            root.update(input={"request_chars": payload.get("request_chars")}, metadata=payload)
        elif kind is EventType.AGENT_STARTED:
            self._agents[(event.agent, event.step_id)] = self._ensure_root().start_observation(
                name=f"agent:{event.agent}",
                as_type="agent",
                metadata={"step_id": event.step_id, "attempt": payload.get("attempt")},
            )
        elif kind in (EventType.AGENT_COMPLETED, EventType.AGENT_FAILED):
            span = self._agents.pop((event.agent, event.step_id), None)
            if span is not None:
                failed = kind is EventType.AGENT_FAILED
                span.update(
                    output=payload.get("summary") or payload.get("error"),
                    metadata={k: v for k, v in payload.items() if k != "output"},
                    level="ERROR" if failed else "DEFAULT",
                    status_message=payload.get("error"),
                )
                span.end()
        elif kind is EventType.LLM_CALL:
            self._parent(event).start_observation(
                name=f"llm:{payload.get('prompt')}",
                as_type="generation",
                model=payload.get("model"),
                version=payload.get("prompt_version"),
                usage_details={
                    "input": int(payload.get("input_tokens") or 0),
                    "output": int(payload.get("output_tokens") or 0),
                    "cache_read_input_tokens": int(payload.get("cache_read_tokens") or 0),
                },
                cost_details={"total": float(payload.get("cost_usd") or 0.0)},
                metadata=payload,
                level=_LEVELS.get(str(payload.get("status")), "DEFAULT"),
            ).end()
        elif kind in (EventType.TOOL_CALL, EventType.TOOL_DENIED):
            self._parent(event).start_observation(
                name=f"tool:{payload.get('tool')}",
                as_type="tool",
                input=payload.get("arguments"),
                output=payload.get("output_preview"),
                metadata={"latency_ms": payload.get("latency_ms"), "status": payload.get("status")},
                level=_LEVELS.get(str(payload.get("status")), "DEFAULT"),
                status_message=payload.get("error"),
            ).end()
        elif kind is EventType.RETRIEVAL:
            self._parent(event).start_observation(
                name="retrieval",
                as_type="retriever",
                input=payload.get("query"),
                output=payload.get("results"),
                metadata={"filtered_out": payload.get("filtered_out")},
            ).end()
        elif kind is EventType.GUARDRAIL_TRIGGERED:
            self._parent(event).start_observation(
                name=str(payload.get("guardrail")),
                as_type="guardrail",
                output=payload,
                level="WARNING",
            ).end()
        elif kind is EventType.CRITIC_VERDICT:
            root = self._ensure_root()
            root.start_observation(name="critic", as_type="evaluator", output=payload).end()
            root.score_trace(
                name="critic_score",
                value=float(payload.get("score") or 0),
                comment=f"round {payload.get('round')}: {payload.get('status')}",
            )
        elif kind in _EVENT_ONLY:
            self._ensure_root().create_event(name=kind.value, metadata=payload)
        elif kind in _RUN_END:
            await self._finish(event)

    async def _finish(self, event: TraceEvent) -> None:
        root = self._ensure_root()
        for span in self._agents.values():  # agents interrupted by a failure or cancellation
            span.end()
        self._agents.clear()
        root.update(
            output={
                k: event.payload.get(k) for k in ("status", "accepted", "abort_reason", "metrics")
            },
            level="DEFAULT" if event.type is EventType.RUN_COMPLETED else "ERROR",
        )
        root.end()
        await asyncio.to_thread(self._client.flush)


def build_langfuse_tracer_factory(settings: Settings) -> Any:
    """Returns ``(tenant_id, run_id) -> LangfuseTracer`` or None when Langfuse is not configured."""
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log_event(
            logger,
            "langfuse_unavailable",
            level=logging.WARNING,
            detail="LANGFUSE_* keys are set but the package is missing: "
            "install the 'observability' extra",
        )
        return None
    secret = settings.langfuse_secret_key
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=secret.get_secret_value() if secret else None,
        host=settings.langfuse_host,
        environment=settings.app_env,
        release=__version__,
        mask=mask,
    )

    def factory(tenant_id: str, run_id: uuid.UUID) -> LangfuseTracer:
        return LangfuseTracer(client, run_id=run_id, tenant_id=tenant_id)

    return factory
