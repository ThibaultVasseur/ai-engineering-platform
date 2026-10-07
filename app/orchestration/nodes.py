"""Graph nodes: read the state, call one agent (or the supervisor policy), return an update.

Nodes never raise for *expected* failures (provider errors, invalid outputs, exhausted
budgets): they record a ``RunError`` and let the routers/supervisor decide what happens next.
Unexpected exceptions (bugs) propagate and fail the run loudly.
"""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.runtime import Runtime

from app.agents.base import AgentContext, AgentError
from app.llm.types import BudgetExceededError, LLMError
from app.observability.tracing import EventType
from app.orchestration.context import RunContext
from app.orchestration.policies import RunBudget
from app.orchestration.state import AgentMessage, GraphState, RunError, StepState, StepStatus
from app.schemas.agent import AgentResult, DependencyResult, SupervisorAction, WorkerInput
from app.schemas.plan import Plan, PlanStep
from app.schemas.review import (
    ReviewInput,
    ReviewStatus,
    SourceDigest,
    StepDigest,
    SynthesisInput,
)
from app.schemas.run import FinalResult, FinalStatus, RunMetrics, StepSummary
from app.schemas.spec import Feasibility, PromptSpec

RECOVERABLE_ERRORS = (LLMError, AgentError, BudgetExceededError)


class Node(Protocol):
    """Shape LangGraph expects for a node that receives the run context."""

    async def __call__(
        self, state: GraphState, *, runtime: Runtime[RunContext]
    ) -> dict[str, Any]: ...


def _error(node: str, exc: BaseException, step_id: str | None = None) -> RunError:
    return RunError(
        node=node, step_id=step_id, error_type=type(exc).__name__, message=str(exc)[:500]
    )


def _abort(node: str, exc: BaseException) -> dict[str, Any]:
    return {
        "errors": [_error(node, exc)],
        "abort_reason": f"{node} failed: {type(exc).__name__}: {str(exc)[:300]}",
        "step_count": 1,
    }


# --------------------------------------------------------------------------------------
# Control nodes
# --------------------------------------------------------------------------------------
async def prompt_optimizer_node(
    state: GraphState, *, runtime: Runtime[RunContext]
) -> dict[str, Any]:
    ctx = runtime.context.agent_ctx
    try:
        outcome = await runtime.context.agents.optimizer.run(ctx, state["user_request"])
    except RECOVERABLE_ERRORS as exc:
        return _abort("prompt_optimizer", exc)
    spec = outcome.value
    await ctx.emit(
        EventType.PROMPT_OPTIMIZED,
        node="prompt_optimizer",
        feasibility=spec.feasibility.value,
        objective=spec.objective,
        requirements=spec.requirements,
        acceptance_criteria=spec.acceptance_criteria,
        needs_knowledge_base=spec.needs_knowledge_base,
    )
    return {
        "optimized_prompt": spec,
        "step_count": 1,
        "messages": [
            AgentMessage(
                sender="prompt_optimizer",
                recipient="planner",
                content=f"[{spec.feasibility.value}] {spec.objective}"[:500],
            )
        ],
    }


async def planner_node(state: GraphState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context.agent_ctx
    spec = _spec(state)
    try:
        outcome = await runtime.context.agents.planner.run(ctx, spec)
    except RECOVERABLE_ERRORS as exc:
        return _abort("planner", exc)
    plan = outcome.value
    await ctx.emit(
        EventType.PLAN_CREATED,
        node="planner",
        rationale=plan.rationale,
        order=plan.topological_order(),
        steps=[
            {
                "id": step.id,
                "type": step.type.value,
                "title": step.title,
                "agent": step.recommended_agent,
                "dependencies": step.dependencies,
                "priority": step.priority,
            }
            for step in plan.steps
        ],
    )
    return {
        "plan": plan,
        "steps": {step.id: StepState() for step in plan.steps},
        "step_count": 1,
        "messages": [
            AgentMessage(
                sender="planner",
                recipient="supervisor",
                content=f"Plan: {', '.join(plan.topological_order())}",
            )
        ],
    }


async def supervisor_node(state: GraphState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context.agent_ctx
    supervisor = runtime.context.agents.supervisor
    outcome = await supervisor.decide(state, ctx)
    decision = outcome.decision
    payload: dict[str, Any] = {
        "action": decision.action.value,
        "target_agent": decision.agent,
        "reason": decision.reason,
        "strategy": supervisor.strategy,
    }
    if outcome.llm_rejected:
        payload["llm_proposal_rejected"] = outcome.llm_rejected
    await ctx.emit(
        EventType.SUPERVISOR_DECISION,
        node="supervisor",
        agent="supervisor",
        step_id=decision.step_id,
        **payload,
    )
    update: dict[str, Any] = {**outcome.updates, "next_action": decision, "step_count": 1}
    if outcome.abort:
        update["abort_reason"] = decision.reason
        if decision.reason.startswith("max_"):
            await ctx.emit(EventType.LIMIT_REACHED, node="supervisor", reason=decision.reason)
    if decision.action is SupervisorAction.DISPATCH and decision.agent:
        update["messages"] = [
            AgentMessage(
                sender="supervisor",
                recipient=decision.agent,
                content=f"Execute step '{decision.step_id}'",
                step_id=decision.step_id,
            )
        ]
    return update


# --------------------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------------------
def build_worker_input(
    state: GraphState,
    step: PlanStep,
    current: StepState,
    *,
    instructions: str | None,
    attempt: int,
) -> WorkerInput:
    """Only what the step needs: objective, its criteria, its dependencies' results."""
    spec = _spec(state)
    steps = state.get("steps", {})
    dependencies = []
    for dep_id in step.dependencies:
        result = steps[dep_id].result if dep_id in steps else None
        if result is not None:
            dependencies.append(
                DependencyResult(
                    step_id=dep_id,
                    agent=result.agent,
                    summary=result.summary,
                    key_points=result.key_points[:12],
                    source_ids=result.source_ids,
                )
            )
    return WorkerInput(
        objective=spec.objective,
        step=step,
        constraints=spec.constraints,
        acceptance_criteria=spec.acceptance_criteria,
        dependency_results=dependencies,
        feedback=current.feedback,
        instructions=instructions,
        attempt=attempt,
    )


def make_worker_node(worker: str) -> Node:
    async def worker_node(state: GraphState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
        ctx = runtime.context.agent_ctx
        decision = state.get("next_action")
        assert decision is not None  # noqa: S101 - set by the supervisor before routing here
        assert decision.step_id is not None  # noqa: S101
        step_id = decision.step_id
        step = _plan(state).step(step_id)
        current = state.get("steps", {})[step_id]
        attempt = current.attempts + 1
        worker_input = build_worker_input(
            state, step, current, instructions=decision.instructions, attempt=attempt
        )
        update: dict[str, Any] = {"step_count": 1, "current_step": step_id}
        try:
            outcome = await runtime.context.agents.workers[worker].run(
                ctx, worker_input, step_id=step_id, attempt=attempt
            )
        except RECOVERABLE_ERRORS as exc:
            update.update(_worker_failure(ctx, worker, step_id, current, attempt, exc))
        else:
            value = outcome.value
            result = AgentResult(
                step_id=step_id,
                agent=worker,
                attempt=attempt,
                status="completed",
                summary=value.summary,
                key_points=value.key_points(),
                source_ids=value.cited_sources(),
                output=value.model_dump(mode="json"),
                usage=outcome.usage,
                latency_ms=outcome.latency_ms,
            )
            update["steps"] = {
                step_id: current.model_copy(
                    update={
                        "status": StepStatus.COMPLETED,
                        "agent": worker,
                        "attempts": attempt,
                        "failures": 0,
                        "feedback": [],
                        "result": result,
                    }
                )
            }
            update["agent_results"] = [result]
            update["messages"] = [
                AgentMessage(
                    sender=worker,
                    recipient="supervisor",
                    content=value.summary[:500],
                    step_id=step_id,
                )
            ]
        update["tool_results"] = ctx.journal.take_new_tool_calls()
        update["retrieved_documents"] = ctx.journal.sources.all()
        return update

    worker_node.__name__ = f"{worker}_node"
    return worker_node


def _worker_failure(
    ctx: AgentContext,
    worker: str,
    step_id: str,
    current: StepState,
    attempt: int,
    exc: BaseException,
) -> dict[str, Any]:
    if isinstance(exc, BudgetExceededError):
        # Not the step's fault: leave it as is, the supervisor stops the run on the violation.
        new_state = current
    else:
        failures = current.failures + 1
        exhausted = failures >= ctx.limits.max_step_attempts
        new_state = current.model_copy(
            update={
                "agent": worker,
                "attempts": attempt,
                "failures": failures,
                "status": StepStatus.FAILED if exhausted else current.status,
            }
        )
    result = AgentResult(
        step_id=step_id,
        agent=worker,
        attempt=attempt,
        status="failed",
        error=f"{type(exc).__name__}: {str(exc)[:300]}",
    )
    return {
        "steps": {step_id: new_state},
        "agent_results": [result],
        "errors": [_error(worker, exc, step_id)],
        "messages": [
            AgentMessage(
                sender=worker,
                recipient="supervisor",
                content=f"Step failed: {type(exc).__name__}",
                step_id=step_id,
            )
        ],
    }


# --------------------------------------------------------------------------------------
# Synthesis, review, finalisation
# --------------------------------------------------------------------------------------
def step_digests(state: GraphState, *, include_output: bool) -> list[StepDigest]:
    plan = _plan(state)
    steps = state.get("steps", {})
    digests = []
    for step_id in plan.topological_order():
        step = plan.step(step_id)
        result = steps[step_id].result if step_id in steps else None
        digests.append(
            StepDigest(
                step_id=step_id,
                title=step.title,
                agent=result.agent if result else step.recommended_agent,
                summary=result.summary if result else "(not executed)",
                key_points=result.key_points if result else [],
                source_ids=result.source_ids if result else [],
                output=result.output if (result and include_output) else None,
            )
        )
    return digests


def build_synthesis_input(state: GraphState) -> SynthesisInput:
    spec = _spec(state)
    review = state.get("review")
    feedback: list[str] = []
    if review is not None and review.status is ReviewStatus.FAIL:
        feedback = [issue.description for issue in review.issues] + list(review.suggestions)
    return SynthesisInput(
        user_request=state.get("user_request", ""),
        objective=spec.objective,
        deliverables=spec.deliverables,
        acceptance_criteria=spec.acceptance_criteria,
        assumptions=spec.assumptions,
        clarifying_questions=spec.clarifying_questions,
        steps=step_digests(state, include_output=True),
        sources=[
            SourceDigest(label=source.label, title=source.title, excerpt=source.excerpt)
            for source in state.get("retrieved_documents", [])
        ],
        review_feedback=feedback,
    )


async def synthesizer_node(state: GraphState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context.agent_ctx
    try:
        outcome = await runtime.context.agents.synthesizer.run(ctx, build_synthesis_input(state))
    except RECOVERABLE_ERRORS as exc:
        return _abort("synthesizer", exc)
    draft = outcome.value
    return {
        "draft": draft,
        "step_count": 1,
        "messages": [
            AgentMessage(sender="synthesizer", recipient="critic", content=draft.title[:500])
        ],
    }


async def critic_node(state: GraphState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context.agent_ctx
    spec = _spec(state)
    draft = state.get("draft")
    assert draft is not None  # noqa: S101 - guaranteed by route_after_synthesizer
    review_input = ReviewInput(
        objective=spec.objective,
        acceptance_criteria=spec.acceptance_criteria,
        plan_steps=step_digests(state, include_output=False),
        draft=draft,
        available_sources=[source.label for source in state.get("retrieved_documents", [])],
        pass_threshold=ctx.limits.critic_pass_threshold,
        round=len(state.get("reviews", [])) + 1,
    )
    try:
        outcome = await runtime.context.agents.critic.run(ctx, review_input)
    except RECOVERABLE_ERRORS as exc:
        return _abort("critic", exc)
    review = outcome.value
    await ctx.emit(
        EventType.CRITIC_VERDICT,
        node="critic",
        agent="critic",
        status=review.status.value,
        score=review.score,
        round=review_input.round,
        issues=[issue.description for issue in review.issues][:10],
        rework_steps=review.rework_steps,
    )
    retries_left = state.get("retry_count", 0) < state.get("max_retries", 0)
    return {
        "review": review,
        "reviews": [review],
        "rework_pending": review.status is ReviewStatus.FAIL and retries_left,
        "step_count": 1,
        "messages": [
            AgentMessage(
                sender="critic",
                recipient="supervisor",
                content=f"{review.status.value} (score {review.score})",
            )
        ],
    }


async def finalizer_node(state: GraphState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
    return {
        "final_result": build_final_result(state, runtime.context.agent_ctx.budget),
        "step_count": 1,
    }


def build_final_result(state: GraphState, budget: RunBudget) -> FinalResult:
    spec = state.get("optimized_prompt")
    review = state.get("review")
    abort_reason = state.get("abort_reason")
    retry_count = state.get("retry_count", 0)
    warnings: list[str] = []

    if spec is not None and spec.feasibility is Feasibility.REJECTED:
        status, accepted = FinalStatus.REJECTED, False
        warnings.append(spec.rejection_reason or "Request rejected")
    elif abort_reason:
        status, accepted = FinalStatus.FAILED, False
        warnings.append(f"Run stopped: {abort_reason}")
    elif review is not None and review.status is ReviewStatus.PASS:
        status, accepted = FinalStatus.COMPLETED, True
    elif review is not None:
        status, accepted = FinalStatus.FAILED, False
        warnings.append(
            f"The critic rejected the answer after {retry_count} rework cycle(s) "
            f"(last score {review.score})."
        )
    else:
        status, accepted = FinalStatus.FAILED, False
        warnings.append("No review was produced.")
    if spec is not None and spec.feasibility is Feasibility.NEEDS_CLARIFICATION:
        warnings.append("The request was ambiguous: the answer relies on explicit assumptions.")

    plan = state.get("plan")
    steps = state.get("steps", {})
    summaries = []
    if plan is not None:
        for step in plan.steps:
            step_state = steps.get(step.id, StepState())
            summaries.append(
                StepSummary(
                    step_id=step.id,
                    title=step.title,
                    agent=step_state.agent,
                    status=step_state.status.value,
                    attempts=step_state.attempts,
                    summary=step_state.result.summary if step_state.result else "",
                )
            )
    return FinalResult(
        status=status,
        accepted=accepted,
        answer=state.get("draft"),
        review=review,
        sources=state.get("retrieved_documents", []),
        steps=summaries,
        warnings=warnings,
        abort_reason=abort_reason,
        metrics=RunMetrics(
            step_count=state.get("step_count", 0) + 1,
            agent_calls=budget.agent_calls,
            llm_calls=budget.llm_calls,
            tool_calls=len(state.get("tool_results", [])),
            retry_count=retry_count,
            input_tokens=budget.usage.input_tokens,
            output_tokens=budget.usage.output_tokens,
            cache_read_tokens=budget.usage.cache_read_tokens,
            cost_usd=budget.cost_usd,
            duration_ms=round(budget.elapsed_seconds * 1000, 1),
        ),
    )


def _spec(state: GraphState) -> PromptSpec:
    spec = state.get("optimized_prompt")
    assert spec is not None  # noqa: S101 - guaranteed by the routers
    return spec


def _plan(state: GraphState) -> Plan:
    plan = state.get("plan")
    assert plan is not None  # noqa: S101 - guaranteed by the routers
    return plan
