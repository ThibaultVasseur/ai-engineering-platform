"""Planner: ``PromptSpec`` -> ``Plan`` (a validated DAG of worker steps)."""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.llm.prompts import PLANNER
from app.schemas.plan import Plan, StepType
from app.schemas.spec import PromptSpec


class PlannerAgent(BaseAgent[PromptSpec, Plan]):
    profile = AGENT_PROFILES[AgentName.PLANNER]
    prompt = PLANNER

    async def execute(self, ctx: AgentContext, data: PromptSpec, *, step_id: str | None) -> Plan:
        max_steps = ctx.limits.max_plan_steps

        def consistent_with_spec(plan: Plan) -> None:
            if len(plan.steps) > max_steps:
                raise ValueError(
                    f"the plan has {len(plan.steps)} steps but at most {max_steps} are allowed; "
                    "merge related steps"
                )
            grounded = {StepType.KNOWLEDGE, StepType.RESEARCH}
            if data.needs_knowledge_base and not any(s.type in grounded for s in plan.steps):
                # Seen live: a question about a vendor's proposal answered by a coding step
                # that never searched the documents.
                raise ValueError(
                    "the specification needs the internal documents (needs_knowledge_base) but "
                    "no step searches them: add a 'knowledge' or 'research' step"
                )

        payload = {"spec": data.model_dump(mode="json"), "max_steps": max_steps}
        return await self._generate(
            ctx, payload, Plan, step_id=step_id, validate=consistent_with_spec
        )

    def summarize(self, value: Plan) -> str:
        steps = ", ".join(f"{step.id}->{step.recommended_agent}" for step in value.steps)
        return f"{len(value.steps)} step(s): {steps}"[:240]
