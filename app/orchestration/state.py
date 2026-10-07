"""Typed LangGraph state.

LangGraph merges the partial updates returned by nodes into this state. Fields annotated with
a reducer are *accumulated* (append-only histories, per-step map, counters); the others are
overwritten by the last writer. Every value is a typed Pydantic model — no free-form dicts
travel between nodes.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from app.schemas.agent import AgentResult, SupervisorDecision
from app.schemas.knowledge import SourceRef
from app.schemas.plan import Plan
from app.schemas.review import FinalDraft, Review
from app.schemas.run import FinalResult
from app.schemas.spec import PromptSpec
from app.schemas.tool import ToolCallRecord


class StepStatus(StrEnum):
    PENDING = "pending"
    REWORK = "rework"
    COMPLETED = "completed"
    FAILED = "failed"


RUNNABLE_STATUSES = frozenset({StepStatus.PENDING, StepStatus.REWORK})


class StepState(BaseModel):
    status: StepStatus = StepStatus.PENDING
    agent: str | None = None
    attempts: int = 0  # executions started (initial run + retries + reworks)
    failures: int = 0  # consecutive failures in the current attempt cycle
    feedback: list[str] = Field(default_factory=list)  # critic issues to address
    result: AgentResult | None = None


class AgentMessage(BaseModel):
    """Inter-agent communication log (who handed what to whom)."""

    sender: str
    recipient: str
    content: str
    step_id: str | None = None


class RunError(BaseModel):
    node: str
    step_id: str | None = None
    error_type: str
    message: str


def merge_steps(left: dict[str, StepState], right: dict[str, StepState]) -> dict[str, StepState]:
    return {**left, **right}


def merge_sources(left: list[SourceRef], right: list[SourceRef]) -> list[SourceRef]:
    known = {source.label for source in left}
    return left + [source for source in right if source.label not in known]


class GraphState(TypedDict, total=False):
    # identity and input
    run_id: str
    task_id: str | None
    tenant_id: str
    user_request: str
    max_retries: int
    # artefacts produced along the pipeline
    optimized_prompt: PromptSpec | None
    plan: Plan | None
    steps: Annotated[dict[str, StepState], merge_steps]
    current_step: str | None
    next_action: SupervisorDecision | None
    draft: FinalDraft | None
    review: Review | None
    rework_pending: bool
    abort_reason: str | None
    final_result: FinalResult | None
    # append-only histories
    messages: Annotated[list[AgentMessage], operator.add]
    agent_results: Annotated[list[AgentResult], operator.add]
    tool_results: Annotated[list[ToolCallRecord], operator.add]
    retrieved_documents: Annotated[list[SourceRef], merge_sources]
    reviews: Annotated[list[Review], operator.add]
    errors: Annotated[list[RunError], operator.add]
    # counters
    retry_count: int
    step_count: Annotated[int, operator.add]


def initial_state(
    *,
    run_id: str,
    tenant_id: str,
    user_request: str,
    max_retries: int,
    task_id: str | None = None,
) -> GraphState:
    return GraphState(
        run_id=run_id,
        task_id=task_id,
        tenant_id=tenant_id,
        user_request=user_request,
        max_retries=max_retries,
        optimized_prompt=None,
        plan=None,
        steps={},
        current_step=None,
        next_action=None,
        draft=None,
        review=None,
        rework_pending=False,
        abort_reason=None,
        final_result=None,
        messages=[],
        agent_results=[],
        tool_results=[],
        retrieved_documents=[],
        reviews=[],
        errors=[],
        retry_count=0,
        step_count=0,
    )
