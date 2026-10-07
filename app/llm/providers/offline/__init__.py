"""Offline provider: a deterministic, rule-based stand-in for an LLM.

It exists so that the *whole* platform — graph, tool calls, RAG, critic loop, persistence,
tracing, evaluation — runs in CI and on a fresh clone without any API key. Each prompt name has
a handler that reads the JSON context the agents send and answers with the same structured
format a real model must produce. It is a simulator for the plumbing, not an intelligence:
answers are template-based and documented as such.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from app.llm.types import LLMError, LLMRequest, LLMResponse, StopReason, ToolCall, Usage

OFFLINE_MODEL = "offline-simulator-v1"


@dataclass
class OfflineReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


OfflineHandler = Callable[[LLMRequest], OfflineReply]


def estimate_tokens(text: str) -> int:
    """~4 characters per token: good enough for usage reports of a simulator."""
    return max(1, len(text) // 4) if text else 0


def _request_text(request: LLMRequest) -> str:
    parts = [request.system]
    for message in request.messages:
        parts.append(message.text)
        parts.extend(result.content for result in message.tool_results)
    return "\n".join(parts)


class OfflineLLM:
    def __init__(self, handlers: Mapping[str, OfflineHandler] | None = None) -> None:
        if handlers is None:
            from app.llm.providers.offline.handlers import default_handlers

            handlers = default_handlers()
        self._handlers = dict(handlers)

    @property
    def provider(self) -> str:
        return "offline"

    @property
    def model(self) -> str:
        return OFFLINE_MODEL

    async def complete(self, request: LLMRequest) -> LLMResponse:
        handler = self._handlers.get(request.metadata.prompt_name)
        if handler is None:
            raise LLMError(
                f"offline provider has no handler for prompt '{request.metadata.prompt_name}'"
            )
        started = time.perf_counter()
        try:
            reply = handler(request)
        except Exception as exc:
            # Same contract as a real provider: a failed generation is an LLMError, which the
            # graph handles (step retry, then a clean stop) instead of crashing the run.
            raise LLMError(
                f"offline handler '{request.metadata.prompt_name}' failed: {type(exc).__name__}"
            ) from exc
        return LLMResponse(
            text=reply.text,
            tool_calls=reply.tool_calls,
            stop_reason=StopReason.TOOL_USE if reply.tool_calls else StopReason.END_TURN,
            usage=Usage(
                input_tokens=estimate_tokens(_request_text(request)),
                output_tokens=estimate_tokens(reply.text)
                + sum(estimate_tokens(str(call.arguments)) for call in reply.tool_calls),
            ),
            model=OFFLINE_MODEL,
            provider="offline",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
