"""Test doubles for the LLM layer."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from app.llm.types import LLMRequest, LLMResponse, StopReason, ToolCall, Usage

ScriptItem = LLMResponse | str | BaseModel | dict[str, Any] | Callable[[LLMRequest], LLMResponse]

_ids = itertools.count(1)


def text_response(text: str, *, stop: StopReason = StopReason.END_TURN) -> LLMResponse:
    return LLMResponse(
        text=text,
        stop_reason=stop,
        usage=Usage(input_tokens=100, output_tokens=20),
        model="scripted-model",
        provider="scripted",
    )


def tool_response(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)
            for name, arguments in calls
        ],
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(input_tokens=100, output_tokens=10),
        model="scripted-model",
        provider="scripted",
    )


class StaticRetriever:
    """KnowledgeRetriever returning fixed chunks (whatever the query)."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    async def search(self, tenant_id: str, query: str, *, top_k: int, filters: Any = None) -> Any:
        from app.schemas.knowledge import RetrievalResult

        self.queries.append(query)
        return RetrievalResult(query=query, chunks=self.chunks[:top_k])


def chunk(index: int, content: str, title: str = "Technical guide") -> Any:
    from app.schemas.knowledge import RetrievedChunk

    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        title=title,
        source="docs/demo/technical_guide.txt",
        chunk_index=index,
        content=content,
        score=0.9 - index * 0.1,
    )


def make_context(
    llm: Any,
    *,
    limits: Any = None,
    tools: Any = None,
    retriever: Any = None,
) -> tuple[Any, Any]:
    """An ``AgentContext`` wired to in-memory collaborators, plus its tracer."""
    from app.agents.base import AgentContext
    from app.observability.tracing import InMemoryTracer
    from app.orchestration.policies import RunBudget, RunLimits

    run_limits = limits or RunLimits()
    tracer = InMemoryTracer()
    context = AgentContext(
        run_id="run-test",
        tenant_id="tenant-test",
        llm=llm,
        tracer=tracer,
        budget=RunBudget(run_limits),
        limits=run_limits,
        tools=tools,
        retriever=retriever,
    )
    return context, tracer


class ScriptedLLM:
    """Replays a fixed script of responses and records every request it receives."""

    def __init__(self, script: Iterable[ScriptItem]) -> None:
        self._script = list(script)
        self.requests: list[LLMRequest] = []

    @property
    def provider(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "scripted-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError(f"ScriptedLLM exhausted (prompt={request.metadata.prompt_name})")
        item = self._script.pop(0)
        if callable(item):
            return item(request)
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, BaseModel):
            return text_response(item.model_dump_json())
        if isinstance(item, dict):
            return text_response(json.dumps(item))
        return text_response(item)
