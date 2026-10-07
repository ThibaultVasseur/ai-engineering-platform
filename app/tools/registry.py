"""Tool registry and the per-run executor that enforces permissions.

Every call goes through ``BoundToolExecutor.execute`` which applies, in order:
1. the tool exists;
2. the tool is in the calling agent's allow-list (``AgentProfile.allowed_tools``);
3. WRITE tools require the run-level opt-in ``allow_writes``;
4. arguments validate against the tool's Pydantic model;
5. the handler runs under a timeout;
6. the output is serialised and truncated, the call is recorded and traced (arguments are
   redacted in records).
Denials and errors are returned to the model as ``is_error`` tool results so it can adapt;
they are never silently dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agents.registry import AgentProfile
from app.core.logging import log_event, redact
from app.core.text import truncate
from app.llm.structured import describe_validation_error
from app.llm.types import ToolCall, ToolResult, ToolSpec
from app.observability.tracing import EventType, TraceEvent, Tracer
from app.orchestration.journal import RunJournal
from app.schemas.tool import ToolCallRecord, ToolCallStatus, ToolInfo, ToolPermission
from app.tools.base import KnowledgeRetriever, PlatformData, Tool, ToolContext, ToolError

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool],
        *,
        default_timeout_seconds: float = 10.0,
        max_output_chars: int = 8_000,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool
        self.default_timeout_seconds = default_timeout_seconds
        self.max_output_chars = max_output_chars

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def describe(self) -> list[ToolInfo]:
        return [tool.info(self.default_timeout_seconds) for tool in self._tools.values()]

    def bind(
        self,
        *,
        tenant_id: str,
        run_id: str,
        tracer: Tracer,
        journal: RunJournal,
        allow_writes: bool = False,
        knowledge: KnowledgeRetriever | None = None,
        data: PlatformData | None = None,
    ) -> BoundToolExecutor:
        return BoundToolExecutor(
            self,
            tenant_id=tenant_id,
            run_id=run_id,
            tracer=tracer,
            journal=journal,
            allow_writes=allow_writes,
            knowledge=knowledge,
            data=data,
        )


class BoundToolExecutor:
    """The registry bound to one run: tenant, tracer, journal and write policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tenant_id: str,
        run_id: str,
        tracer: Tracer,
        journal: RunJournal,
        allow_writes: bool,
        knowledge: KnowledgeRetriever | None,
        data: PlatformData | None,
    ) -> None:
        self._registry = registry
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._tracer = tracer
        self._journal = journal
        self._allow_writes = allow_writes
        self._knowledge = knowledge
        self._data = data

    def _visible(self, tool: Tool, agent: AgentProfile) -> bool:
        if tool.name not in agent.allowed_tools:
            return False
        return tool.permission is ToolPermission.READ or self._allow_writes

    def specs_for(self, agent: AgentProfile) -> list[ToolSpec]:
        """Only the tools the agent may actually call are shown to the model."""
        return [tool.spec() for tool in self._registry.all() if self._visible(tool, agent)]

    async def execute(
        self, call: ToolCall, agent: AgentProfile, *, step_id: str | None
    ) -> ToolResult:
        started = time.perf_counter()
        tool = self._registry.get(call.name)
        if tool is None:
            return await self._deny(call, agent, step_id, f"unknown tool '{call.name}'")
        if tool.name not in agent.allowed_tools:
            return await self._deny(
                call, agent, step_id, f"agent '{agent.name}' is not allowed to use '{tool.name}'"
            )
        if tool.permission is ToolPermission.WRITE and not self._allow_writes:
            return await self._deny(
                call, agent, step_id, f"'{tool.name}' writes data; writes are disabled for this run"
            )

        try:
            arguments = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return await self._finish(
                call,
                agent,
                step_id,
                started,
                error=f"invalid arguments:\n{describe_validation_error(exc)}",
            )

        context = ToolContext(
            tenant_id=self._tenant_id,
            run_id=self._run_id,
            agent=agent.name.value,
            step_id=step_id,
            sources=self._journal.sources,
            knowledge=self._knowledge,
            data=self._data,
            tracer=self._tracer,
        )
        timeout = tool.timeout_seconds or self._registry.default_timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                output = await tool.handler(arguments, context)
        except TimeoutError:
            return await self._finish(
                call, agent, step_id, started, error=f"timed out after {timeout}s"
            )
        except ToolError as exc:
            return await self._finish(call, agent, step_id, started, error=str(exc))
        except Exception as exc:  # never leak internals (stack traces, SQL) to the model
            log_event(
                logger,
                "tool_handler_crashed",
                level=logging.ERROR,
                exc_info=exc,
                tool=tool.name,
                run_id=self._run_id,
            )
            return await self._finish(
                call, agent, step_id, started, error=f"internal tool error ({type(exc).__name__})"
            )
        return await self._finish(call, agent, step_id, started, output=output)

    # ----------------------------------------------------------------------------------
    def _serialise(self, output: Any) -> str:
        if isinstance(output, BaseModel):
            output = output.model_dump(mode="json")
        text = json.dumps(output, ensure_ascii=False, default=str)
        return truncate(text, self._registry.max_output_chars)

    async def _finish(
        self,
        call: ToolCall,
        agent: AgentProfile,
        step_id: str | None,
        started: float,
        *,
        output: Any = None,
        error: str | None = None,
    ) -> ToolResult:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        content = f"Tool error: {error}" if error else self._serialise(output)
        record = ToolCallRecord(
            tool=call.name,
            agent=agent.name.value,
            step_id=step_id,
            status=ToolCallStatus.ERROR if error else ToolCallStatus.OK,
            arguments=redact(call.arguments),
            output_preview=truncate(content, 500),
            error=error,
            latency_ms=latency_ms,
        )
        await self._record(record, EventType.TOOL_CALL)
        return ToolResult(tool_call_id=call.id, content=content, is_error=error is not None)

    async def _deny(
        self, call: ToolCall, agent: AgentProfile, step_id: str | None, reason: str
    ) -> ToolResult:
        record = ToolCallRecord(
            tool=call.name,
            agent=agent.name.value,
            step_id=step_id,
            status=ToolCallStatus.DENIED,
            arguments=redact(call.arguments),
            error=reason,
        )
        await self._record(record, EventType.TOOL_DENIED)
        log_event(
            logger,
            "tool_denied",
            level=logging.WARNING,
            tool=call.name,
            agent=agent.name.value,
            reason=reason,
        )
        return ToolResult(
            tool_call_id=call.id, content=f"Permission denied: {reason}", is_error=True
        )

    async def _record(self, record: ToolCallRecord, event_type: EventType) -> None:
        self._journal.record_tool_call(record)
        await self._tracer.emit(
            TraceEvent(
                type=event_type,
                agent=record.agent,
                step_id=record.step_id,
                payload={
                    "tool": record.tool,
                    "status": record.status.value,
                    "arguments": record.arguments,
                    "error": record.error,
                    "latency_ms": record.latency_ms,
                    "output_preview": record.output_preview[:300],
                },
            )
        )
