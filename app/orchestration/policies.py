"""Run limits and the per-run budget that enforces them.

Limits are enforced at two levels:
* gracefully by the supervisor, which checks ``RunBudget.violation()`` before dispatching and
  routes to the finalizer with an explicit abort reason;
* strictly at the LLM boundary (``check_llm_call``) and agent boundary (``check_agent_call``),
  so no loop — tool loop, repair loop, rework loop — can exceed them.
LangGraph's ``recursion_limit`` is a last-resort backstop on top of MAX_AGENT_STEPS.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.llm.types import BudgetExceededError, Usage


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 40
    max_agent_calls: int = 16
    max_retries: int = 2
    max_step_attempts: int = 2
    max_tool_calls_per_agent: int = 6
    max_plan_steps: int = 6
    max_runtime_seconds: float = 300.0
    max_cost_usd: float = 2.0
    critic_pass_threshold: int = 70

    @classmethod
    def from_settings(cls, settings: Settings) -> RunLimits:
        return cls(
            max_steps=settings.max_agent_steps,
            max_agent_calls=settings.max_agent_calls,
            max_retries=settings.max_agent_retries,
            max_step_attempts=settings.max_step_attempts,
            max_tool_calls_per_agent=settings.max_tool_calls_per_agent,
            max_plan_steps=settings.max_plan_steps,
            max_runtime_seconds=settings.max_run_seconds,
            max_cost_usd=settings.max_run_cost_usd,
            critic_pass_threshold=settings.critic_pass_threshold,
        )

    @property
    def recursion_limit(self) -> int:
        """Backstop for LangGraph: a little above the graceful MAX_AGENT_STEPS check."""
        return self.max_steps + 10


@dataclass(frozen=True)
class LimitViolation:
    limit: str
    detail: str


class RunBudget:
    """Mutable counters shared by the agents of one run (single event loop: no locking)."""

    def __init__(self, limits: RunLimits, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits
        self._clock = clock
        self._started = clock()
        self.agent_calls = 0
        self.llm_calls = 0
        self.usage = Usage()
        self.cost_usd = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started

    def violation(self) -> LimitViolation | None:
        """Limits that forbid starting new work (graceful check used by the supervisor)."""
        if self.elapsed_seconds >= self.limits.max_runtime_seconds:
            return LimitViolation(
                "max_runtime", f"{self.elapsed_seconds:.1f}s >= {self.limits.max_runtime_seconds}s"
            )
        if self.cost_usd >= self.limits.max_cost_usd:
            return LimitViolation(
                "max_cost", f"${self.cost_usd:.4f} >= ${self.limits.max_cost_usd}"
            )
        if self.agent_calls >= self.limits.max_agent_calls:
            return LimitViolation(
                "max_agent_calls", f"{self.agent_calls} >= {self.limits.max_agent_calls}"
            )
        return None

    def _raise_for_time_or_cost(self) -> None:
        if self.elapsed_seconds >= self.limits.max_runtime_seconds:
            raise BudgetExceededError(
                "max_runtime", f"run exceeded {self.limits.max_runtime_seconds}s"
            )
        if self.cost_usd >= self.limits.max_cost_usd:
            raise BudgetExceededError("max_cost", f"run cost reached ${self.cost_usd:.4f}")

    def check_agent_call(self) -> None:
        if self.agent_calls >= self.limits.max_agent_calls:
            raise BudgetExceededError(
                "max_agent_calls", f"{self.limits.max_agent_calls} agent calls already made"
            )
        self._raise_for_time_or_cost()

    def record_agent_call(self) -> None:
        self.agent_calls += 1

    def check_llm_call(self) -> None:
        self._raise_for_time_or_cost()

    def record_llm_call(self, usage: Usage, cost_usd: float) -> None:
        self.llm_calls += 1
        self.usage = self.usage + usage
        self.cost_usd = round(self.cost_usd + cost_usd, 6)

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_calls": self.agent_calls,
            "llm_calls": self.llm_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cost_usd": self.cost_usd,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
