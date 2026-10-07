"""Conditional edges. Pure functions of the state: trivial to read and to unit-test."""

from __future__ import annotations

from typing import Final, Literal

from app.orchestration.state import GraphState
from app.schemas.agent import SupervisorAction
from app.schemas.review import ReviewStatus
from app.schemas.spec import Feasibility

FINALIZER: Final = "finalizer"


def route_after_optimizer(state: GraphState) -> Literal["planner", "finalizer"]:
    spec = state.get("optimized_prompt")
    if state.get("abort_reason") or spec is None or spec.feasibility is Feasibility.REJECTED:
        return FINALIZER
    return "planner"


def route_after_planner(state: GraphState) -> Literal["supervisor", "finalizer"]:
    if state.get("abort_reason") or state.get("plan") is None:
        return FINALIZER
    return "supervisor"


def route_from_supervisor(state: GraphState) -> str:
    decision = state.get("next_action")
    if decision is None or decision.action is SupervisorAction.FINALIZE:
        return FINALIZER
    if decision.action is SupervisorAction.SYNTHESIZE:
        return "synthesizer"
    assert decision.agent is not None  # noqa: S101 - guaranteed by the supervisor
    return decision.agent


def route_after_synthesizer(state: GraphState) -> Literal["critic", "finalizer"]:
    if state.get("abort_reason") or state.get("draft") is None:
        return FINALIZER
    return "critic"


def route_after_critic(state: GraphState) -> Literal["supervisor", "finalizer"]:
    """FAIL -> rework through the supervisor while retries remain; PASS -> finalizer."""
    review = state.get("review")
    if state.get("abort_reason") or review is None or review.status is ReviewStatus.PASS:
        return FINALIZER
    if state.get("retry_count", 0) < state.get("max_retries", 0):
        return "supervisor"
    return FINALIZER
