"""Executes one run of the graph and reports its lifecycle.

Used by the API (through the run service, which adds persistence), by the CLI and by the
evaluation runner — so the three exercise exactly the same orchestration code.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from app.observability.tracing import EventType
from app.orchestration.context import RunContext
from app.orchestration.graph import build_graph
from app.orchestration.state import GraphState, initial_state
from app.schemas.run import FinalResult

# Hard timeout on top of the graceful MAX_RUN_SECONDS check done by the supervisor.
RUNTIME_GRACE_SECONDS = 30.0

NodeCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class RunAbortedError(Exception):
    """The run could not finish normally (hard timeout, recursion limit, unexpected error)."""


class GraphRunner:
    def __init__(self, graph: Any = None) -> None:
        self.graph = graph if graph is not None else build_graph()

    async def run(
        self,
        run_ctx: RunContext,
        *,
        user_request: str,
        task_id: str | None = None,
        on_update: NodeCallback | None = None,
    ) -> GraphState:
        ctx = run_ctx.agent_ctx
        limits = ctx.limits
        state = initial_state(
            run_id=ctx.run_id,
            tenant_id=ctx.tenant_id,
            user_request=user_request,
            max_retries=limits.max_retries,
            task_id=task_id,
        )
        await ctx.emit(
            EventType.RUN_STARTED, request_chars=len(user_request), limits=asdict(limits)
        )
        final_state: GraphState = state
        try:
            async with asyncio.timeout(limits.max_runtime_seconds + RUNTIME_GRACE_SECONDS):
                async for mode, chunk in self.graph.astream(
                    state,
                    config={"recursion_limit": limits.recursion_limit},
                    context=run_ctx,
                    stream_mode=["updates", "values"],
                ):
                    if mode == "values":
                        final_state = chunk
                    elif on_update is not None:
                        for node, update in chunk.items():
                            await on_update(node, update or {})
        except asyncio.CancelledError:
            await ctx.emit(EventType.RUN_CANCELLED, metrics=ctx.budget.snapshot())
            raise
        except Exception as exc:
            await ctx.emit(
                EventType.RUN_FAILED,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                metrics=ctx.budget.snapshot(),
            )
            raise RunAbortedError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc

        result = final_state.get("final_result")
        await ctx.emit(
            EventType.RUN_COMPLETED if result and result.accepted else EventType.RUN_FAILED,
            status=result.status.value if result else "unknown",
            accepted=bool(result and result.accepted),
            abort_reason=result.abort_reason if result else None,
            metrics=result.metrics.model_dump() if result else ctx.budget.snapshot(),
        )
        return final_state


def final_result_of(state: GraphState) -> FinalResult:
    result = state.get("final_result")
    if result is None:
        raise RunAbortedError("the graph ended without a final result")
    return result
