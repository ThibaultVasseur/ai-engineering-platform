"""Supervisor: the coordinator of the run.

The supervisor is a *policy*, not a free-running LLM loop:

1. hard limits first (MAX_AGENT_STEPS, runtime, cost, agent calls) -> stop with a reason;
2. a failed review is turned into targeted rework (flagged steps + their dependents), bounded
   by MAX_AGENT_RETRIES;
3. a step that failed MAX_STEP_ATTEMPTS times stops the run;
4. otherwise the next *ready* step (dependencies completed) is dispatched to a capable worker
   with only the context it needs; when everything is done, the draft is synthesised.

With ``strategy="llm"`` the model may choose among several ready steps and brief the worker,
but its proposal is validated against the same policy and discarded if invalid
("the LLM proposes, the policy disposes"). With a single option no LLM call is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.agents.base import AgentContext
from app.agents.registry import AgentName
from app.llm.prompts import SUPERVISOR
from app.llm.structured import generate_structured
from app.llm.types import BudgetExceededError, CallMetadata, LLMError, LLMRequest, Message
from app.observability.tracing import EventType
from app.orchestration.state import RUNNABLE_STATUSES, GraphState, StepState, StepStatus
from app.schemas.agent import SupervisorAction, SupervisorChoice, SupervisorDecision
from app.schemas.plan import CAPABLE_AGENTS, Plan, PlanStep, WorkerName


@dataclass
class SupervisorOutcome:
    decision: SupervisorDecision
    updates: dict[str, Any] = field(default_factory=dict)
    abort: bool = False
    llm_rejected: str | None = None


def select_agent(step: PlanStep) -> tuple[WorkerName, str | None]:
    """The planner's recommendation if capable, otherwise the default worker for the type."""
    capable = CAPABLE_AGENTS[step.type]
    if step.recommended_agent in capable:
        return step.recommended_agent, None
    return capable[0], (
        f"'{step.recommended_agent}' cannot execute a {step.type.value} step; using '{capable[0]}'"
    )


def ready_steps(plan: Plan, steps: dict[str, StepState]) -> list[PlanStep]:
    done = {step_id for step_id, state in steps.items() if state.status is StepStatus.COMPLETED}
    order = {step_id: index for index, step_id in enumerate(plan.topological_order())}
    ready = [
        step
        for step in plan.steps
        if steps[step.id].status in RUNNABLE_STATUSES and set(step.dependencies) <= done
    ]
    return sorted(ready, key=lambda step: order[step.id])


def dependents_of(plan: Plan, step_ids: set[str]) -> set[str]:
    """All steps that transitively depend on ``step_ids``."""
    found: set[str] = set()
    frontier = set(step_ids)
    while frontier:
        children = {s.id for s in plan.steps if set(s.dependencies) & frontier} - found
        found |= children
        frontier = children
    return found


class Supervisor:
    def __init__(self, strategy: Literal["rules", "llm"] = "rules") -> None:
        self.strategy = strategy

    async def decide(self, state: GraphState, ctx: AgentContext) -> SupervisorOutcome:
        plan = state.get("plan")
        if plan is None:
            return _abort("no plan available")
        if state.get("step_count", 0) >= ctx.limits.max_steps:
            return _abort(f"max_steps: {ctx.limits.max_steps} graph steps reached")
        violation = ctx.budget.violation()
        if violation is not None:
            return _abort(f"{violation.limit}: {violation.detail}")

        steps = dict(state.get("steps", {}))
        updates: dict[str, Any] = {}
        if state.get("rework_pending"):
            rework = await self._apply_rework(state, plan, steps, ctx)
            updates.update(rework)
            steps.update(rework.get("steps", {}))
            if not rework.get("steps"):
                return SupervisorOutcome(
                    decision=SupervisorDecision(
                        action=SupervisorAction.SYNTHESIZE,
                        reason="review failed without step-level issues: re-synthesise "
                        "the answer with the reviewer feedback",
                    ),
                    updates=updates,
                )

        failed = sorted(step_id for step_id, s in steps.items() if s.status is StepStatus.FAILED)
        if failed:
            outcome = _abort(
                f"step(s) {', '.join(failed)} failed after "
                f"{ctx.limits.max_step_attempts} attempt(s)"
            )
            outcome.updates = updates
            return outcome

        ready = ready_steps(plan, steps)
        if ready:
            outcome = await self._dispatch(ready, steps, state, ctx)
            outcome.updates = {**updates, **outcome.updates}
            return outcome
        if all(s.status is StepStatus.COMPLETED for s in steps.values()):
            return SupervisorOutcome(
                decision=SupervisorDecision(
                    action=SupervisorAction.SYNTHESIZE,
                    reason="all plan steps are completed",
                ),
                updates=updates,
            )
        outcome = _abort("no runnable step: remaining steps are blocked by their dependencies")
        outcome.updates = updates
        return outcome

    # ----------------------------------------------------------------------------------
    async def _apply_rework(
        self, state: GraphState, plan: Plan, steps: dict[str, StepState], ctx: AgentContext
    ) -> dict[str, Any]:
        review = state.get("review")
        retry_count = state.get("retry_count", 0) + 1
        updates: dict[str, Any] = {"rework_pending": False, "retry_count": retry_count}
        if review is None:
            return updates
        flagged = {step_id for step_id in review.rework_steps if step_id in steps}
        cascade = dependents_of(plan, flagged)
        changed: dict[str, StepState] = {}
        for step_id in sorted(flagged | cascade):
            if step_id in flagged:
                feedback = [i.description for i in review.issues if i.step_id == step_id]
                feedback = feedback or [i.description for i in review.issues][:5]
            else:
                feedback = ["An upstream step was reworked: update this step with its new output."]
            changed[step_id] = steps[step_id].model_copy(
                update={"status": StepStatus.REWORK, "feedback": feedback, "failures": 0}
            )
        if changed:
            updates["steps"] = changed
        await ctx.emit(
            EventType.REWORK_REQUESTED,
            node="supervisor",
            retry=retry_count,
            flagged_steps=sorted(flagged),
            cascaded_steps=sorted(cascade - flagged),
            score=review.score,
        )
        return updates

    async def _dispatch(
        self,
        ready: list[PlanStep],
        steps: dict[str, StepState],
        state: GraphState,
        ctx: AgentContext,
    ) -> SupervisorOutcome:
        chosen = ready[0]
        agent, override = select_agent(chosen)
        reason = f"next ready step by dependency order and priority ({len(ready)} ready)" + (
            f"; {override}" if override else ""
        )
        instructions: str | None = None
        llm_rejected: str | None = None

        if self.strategy == "llm" and len(ready) > 1:
            try:
                choice = await self._ask_llm(ready, steps, state, ctx)
            except (LLMError, BudgetExceededError) as exc:
                llm_rejected = f"LLM unavailable ({type(exc).__name__}): rules applied"
            else:
                by_id = {step.id: step for step in ready}
                candidate = by_id.get(choice.step_id)
                if candidate is None:
                    llm_rejected = f"'{choice.step_id}' is not a ready step"
                elif choice.agent not in CAPABLE_AGENTS[candidate.type]:
                    llm_rejected = f"'{choice.agent}' cannot execute step '{candidate.id}'"
                else:
                    chosen, agent = candidate, choice.agent
                    reason, instructions = f"LLM choice: {choice.reason}", choice.instructions

        return SupervisorOutcome(
            decision=SupervisorDecision(
                action=SupervisorAction.DISPATCH,
                step_id=chosen.id,
                agent=agent,
                reason=reason[:500],
                instructions=instructions,
            ),
            updates={"current_step": chosen.id},
            llm_rejected=llm_rejected,
        )

    async def _ask_llm(
        self,
        ready: list[PlanStep],
        steps: dict[str, StepState],
        state: GraphState,
        ctx: AgentContext,
    ) -> SupervisorChoice:
        spec = state.get("optimized_prompt")
        payload = {
            "objective": spec.objective if spec else state.get("user_request", ""),
            "ready_steps": [
                {
                    "id": step.id,
                    "type": step.type.value,
                    "title": step.title,
                    "priority": step.priority,
                    "recommended_agent": step.recommended_agent,
                    "capable_agents": list(CAPABLE_AGENTS[step.type]),
                    "feedback": steps[step.id].feedback,
                }
                for step in ready
            ],
            "completed_steps": [
                {"id": step_id, "summary": s.result.summary if s.result else ""}
                for step_id, s in steps.items()
                if s.status is StepStatus.COMPLETED
            ],
        }
        request = LLMRequest(
            system=SUPERVISOR.system,
            messages=[Message.user(SUPERVISOR.render(payload))],
            metadata=CallMetadata(
                agent=AgentName.SUPERVISOR.value,
                prompt_name=SUPERVISOR.name,
                prompt_version=SUPERVISOR.version,
            ),
        )
        result = await generate_structured(
            ctx.llm, request, SupervisorChoice, max_attempts=ctx.structured_max_attempts
        )
        return result.value


def _abort(reason: str) -> SupervisorOutcome:
    return SupervisorOutcome(
        decision=SupervisorDecision(action=SupervisorAction.FINALIZE, reason=reason[:500]),
        abort=True,
    )
