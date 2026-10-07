"""Agent infrastructure shared by every agent.

``BaseAgent.run`` is the single entry point. It wraps the agent-specific ``execute`` with:
* budget accounting (MAX_AGENT_CALLS, runtime, cost) before anything happens;
* per-invocation usage metering (tokens, LLM calls, model);
* trace events ``agent_started`` / ``agent_completed`` / ``agent_failed``.

Two ways for an agent to talk to the LLM:
* ``_generate``: one structured answer (validated, repaired if needed);
* ``_tool_loop``: the agent may call its allowed tools, bounded by MAX_TOOL_CALLS_PER_AGENT,
  then must give a structured final answer.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ValidationError

from app.agents.registry import AgentProfile
from app.llm.prompts import PromptTemplate
from app.llm.schema import strict_json_schema
from app.llm.structured import (
    RepairHook,
    describe_validation_error,
    generate_structured,
    parse_structured,
    repair_message,
)
from app.llm.types import (
    CallMetadata,
    LLMClient,
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    StructuredOutputError,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from app.observability.tracing import EventType, TraceEvent, Tracer
from app.orchestration.journal import RunJournal
from app.orchestration.policies import RunBudget, RunLimits


class AgentError(Exception):
    """The agent could not complete its task (as opposed to a provider error)."""


class ToolExecutor(Protocol):
    """Tool access granted to the agents of one run (permissions enforced by the registry)."""

    def specs_for(self, agent: AgentProfile) -> list[ToolSpec]: ...

    async def execute(
        self, call: ToolCall, agent: AgentProfile, *, step_id: str | None
    ) -> ToolResult: ...


@dataclass(frozen=True)
class AgentContext:
    """Run-scoped dependencies handed to every agent of a run."""

    run_id: str
    tenant_id: str
    llm: LLMClient
    tracer: Tracer
    budget: RunBudget
    limits: RunLimits
    tools: ToolExecutor | None = None
    retriever: Any = None  # KnowledgeRetriever (see app.rag.retrieval), used by the RAG agent
    journal: RunJournal = field(default_factory=RunJournal)
    structured_max_attempts: int = 3

    async def emit(
        self,
        event_type: EventType,
        *,
        agent: str | None = None,
        step_id: str | None = None,
        node: str | None = None,
        **payload: Any,
    ) -> None:
        await self.tracer.emit(
            TraceEvent(type=event_type, agent=agent, step_id=step_id, node=node, payload=payload)
        )


@dataclass(frozen=True)
class AgentOutcome[T]:
    value: T
    usage: Usage
    llm_calls: int
    model: str | None
    latency_ms: float


class _UsageCounter:
    """LLM wrapper that measures what one agent invocation consumed."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.usage = Usage()
        self.calls = 0
        self.last_model: str | None = None

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await self._inner.complete(request)
        self.calls += 1
        self.usage = self.usage + response.usage
        self.last_model = response.model
        return response


class BaseAgent[InT, OutT: BaseModel](ABC):
    profile: ClassVar[AgentProfile]
    prompt: ClassVar[PromptTemplate]

    async def run(
        self, ctx: AgentContext, data: InT, *, step_id: str | None = None, attempt: int = 1
    ) -> AgentOutcome[OutT]:
        ctx.budget.check_agent_call()
        ctx.budget.record_agent_call()
        counter = _UsageCounter(ctx.llm)
        name = self.profile.name.value
        await ctx.emit(EventType.AGENT_STARTED, agent=name, step_id=step_id, attempt=attempt)
        started = time.perf_counter()
        try:
            value = await self.execute(replace(ctx, llm=counter), data, step_id=step_id)
        except Exception as exc:
            await ctx.emit(
                EventType.AGENT_FAILED,
                agent=name,
                step_id=step_id,
                attempt=attempt,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                llm_calls=counter.calls,
                input_tokens=counter.usage.input_tokens,
                output_tokens=counter.usage.output_tokens,
                prompt_version=self.prompt.version,
                model=counter.last_model,
            )
            raise
        outcome = AgentOutcome(
            value=value,
            usage=counter.usage,
            llm_calls=counter.calls,
            model=counter.last_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        await ctx.emit(
            EventType.AGENT_COMPLETED,
            agent=name,
            step_id=step_id,
            attempt=attempt,
            summary=self.summarize(value),
            latency_ms=outcome.latency_ms,
            llm_calls=outcome.llm_calls,
            input_tokens=outcome.usage.input_tokens,
            output_tokens=outcome.usage.output_tokens,
            prompt_version=self.prompt.version,
            model=outcome.model,
            output=value.model_dump(mode="json"),
        )
        return outcome

    @abstractmethod
    async def execute(self, ctx: AgentContext, data: InT, *, step_id: str | None) -> OutT: ...

    def summarize(self, value: OutT) -> str:
        return value.model_dump_json()[:240]

    # ----------------------------------------------------------------------------------
    # LLM helpers
    # ----------------------------------------------------------------------------------
    def _request(
        self,
        payload: dict[str, Any],
        *,
        step_id: str | None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMRequest:
        return LLMRequest(
            system=self.prompt.system,
            messages=[Message.user(self.prompt.render(payload))],
            tools=tools or [],
            metadata=CallMetadata(
                agent=self.profile.name.value,
                prompt_name=self.prompt.name,
                prompt_version=self.prompt.version,
                step_id=step_id,
            ),
        )

    def _repair_hook(self, ctx: AgentContext, step_id: str | None) -> RepairHook:
        async def hook(attempt: int, errors: str) -> None:
            await ctx.emit(
                EventType.STRUCTURED_OUTPUT_REPAIRED,
                agent=self.profile.name.value,
                step_id=step_id,
                attempt=attempt,
                errors=errors[:500],
            )

        return hook

    async def _generate[M: BaseModel](
        self,
        ctx: AgentContext,
        payload: dict[str, Any],
        output_model: type[M],
        *,
        step_id: str | None,
        validate: Callable[[M], None] | None = None,
    ) -> M:
        result = await generate_structured(
            ctx.llm,
            self._request(payload, step_id=step_id),
            output_model,
            max_attempts=ctx.structured_max_attempts,
            on_repair=self._repair_hook(ctx, step_id),
            validate=validate,
        )
        return result.value

    async def _tool_loop[M: BaseModel](
        self,
        ctx: AgentContext,
        payload: dict[str, Any],
        output_model: type[M],
        *,
        step_id: str | None,
        validate: Callable[[M], None] | None = None,
    ) -> M:
        if ctx.tools is None:
            raise AgentError(f"{self.profile.name}: no tool executor configured for this run")
        request = self._request(payload, step_id=step_id, tools=ctx.tools.specs_for(self.profile))
        request = request.model_copy(
            update={
                "response_schema": strict_json_schema(output_model),
                "response_schema_name": output_model.__name__,
            }
        )
        messages = list(request.messages)
        tool_budget = ctx.limits.max_tool_calls_per_agent
        tools_used = 0
        repairs = 0
        max_turns = tool_budget + ctx.structured_max_attempts + 1
        repair = self._repair_hook(ctx, step_id)

        for _ in range(max_turns):
            response = await ctx.llm.complete(request.model_copy(update={"messages": messages}))
            if response.stop_reason is StopReason.REFUSAL:
                raise LLMRefusalError(
                    f"{self.profile.name}: the model declined the step",
                    category=response.refusal_category,
                )
            if response.tool_calls:
                results: list[ToolResult] = []
                for call in response.tool_calls:
                    if tools_used >= tool_budget:
                        results.append(await self._deny_over_budget(ctx, call, step_id))
                        continue
                    tools_used += 1
                    results.append(await ctx.tools.execute(call, self.profile, step_id=step_id))
                messages = [*messages, Message.from_response(response), Message.results(results)]
                continue
            try:
                value = parse_structured(response.text, output_model)
                if validate is not None:
                    validate(value)
            except (ValidationError, ValueError) as exc:
                repairs += 1
                if repairs >= ctx.structured_max_attempts:
                    raise StructuredOutputError(
                        output_model.__name__, repairs, describe_validation_error(exc)
                    ) from exc
                await repair(repairs, describe_validation_error(exc))
                messages = [
                    *messages,
                    Message.from_response(response),
                    repair_message(response, exc),
                ]
                continue
            return value
        raise AgentError(f"{self.profile.name}: no final answer after {max_turns} turns")

    async def _deny_over_budget(
        self, ctx: AgentContext, call: ToolCall, step_id: str | None
    ) -> ToolResult:
        await ctx.emit(
            EventType.TOOL_DENIED,
            agent=self.profile.name.value,
            step_id=step_id,
            tool=call.name,
            reason="tool_budget_exhausted",
        )
        return ToolResult(
            tool_call_id=call.id,
            content="Tool budget exhausted for this step: answer now with what you already have.",
            is_error=True,
        )
