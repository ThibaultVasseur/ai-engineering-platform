"""Per-run LLM wrapper: enforces the run budget *before* each call and traces every call.

Enforcing limits at this layer (and not only in the supervisor) means that no code path —
a tool loop, a repair loop, a misbehaving agent — can spend past the run's budget.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.core.logging import log_event
from app.llm.pricing import PriceBook
from app.llm.types import LLMClient, LLMError, LLMRequest, LLMResponse, Usage
from app.observability.tracing import EventType, TraceEvent, Tracer

logger = logging.getLogger(__name__)


class LLMBudget(Protocol):
    def check_llm_call(self) -> None:
        """Raise ``BudgetExceededError`` if another call is not allowed."""

    def record_llm_call(self, usage: Usage, cost_usd: float) -> None: ...


class MeteredLLMClient:
    def __init__(
        self, inner: LLMClient, *, budget: LLMBudget, tracer: Tracer, prices: PriceBook
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._tracer = tracer
        self._prices = prices

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._budget.check_llm_call()
        meta = request.metadata
        try:
            response = await self._inner.complete(request)
        except LLMError as exc:
            log_event(logger, "llm_call_failed", level=logging.WARNING, agent=meta.agent)
            await self._tracer.emit(
                TraceEvent(
                    type=EventType.LLM_CALL,
                    agent=meta.agent,
                    step_id=meta.step_id,
                    payload={
                        "status": "error",
                        "prompt": meta.prompt_name,
                        "prompt_version": meta.prompt_version,
                        "model": self._inner.model,
                        "error": str(exc)[:300],
                    },
                )
            )
            raise

        # The served model can differ from the configured one (dated snapshot, server-side
        # fallback model): without a price of its own, it is costed as the configured model.
        priced_as = response.model if self._prices.knows(response.model) else self._inner.model
        cost = self._prices.cost(priced_as, response.usage)
        self._budget.record_llm_call(response.usage, cost)
        await self._tracer.emit(
            TraceEvent(
                type=EventType.LLM_CALL,
                agent=meta.agent,
                step_id=meta.step_id,
                payload={
                    "status": "ok",
                    "provider": response.provider,
                    "model": response.model,
                    "prompt": meta.prompt_name,
                    "prompt_version": meta.prompt_version,
                    "stop_reason": response.stop_reason.value,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": response.usage.cache_read_tokens,
                    "cache_write_tokens": response.usage.cache_write_tokens,
                    "latency_ms": round(response.latency_ms, 1),
                    "cost_usd": cost,
                    "tool_calls": [call.name for call in response.tool_calls],
                    "request_id": response.request_id,
                },
            )
        )
        return response
