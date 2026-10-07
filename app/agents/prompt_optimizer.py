"""Prompt Optimizer: raw request -> validated ``PromptSpec`` (first node of the graph).

A deterministic injection scan runs first. A HIGH-risk request is rejected *without* any LLM
call: the attack never reaches a model or a tool. Lower risks are passed to the model as a
signal so it can decide (and are traced as guardrail events).
"""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.core.guardrails import InjectionRisk, InjectionScan, scan_for_injection
from app.llm.prompts import PROMPT_OPTIMIZER
from app.observability.tracing import EventType
from app.schemas.spec import Feasibility, PromptSpec


def rejected_spec(scan: InjectionScan) -> PromptSpec:
    return PromptSpec(
        objective="Request rejected by the input guardrail before any processing.",
        context="",
        needs_knowledge_base=False,
        feasibility=Feasibility.REJECTED,
        rejection_reason=(
            "Potential prompt injection detected ("
            + ", ".join(scan.signals)
            + "): the request tries to override instructions, extract secrets or trigger "
            "privileged tools. It was not forwarded to any agent or tool."
        ),
    )


class PromptOptimizerAgent(BaseAgent[str, PromptSpec]):
    profile = AGENT_PROFILES[AgentName.PROMPT_OPTIMIZER]
    prompt = PROMPT_OPTIMIZER

    async def execute(self, ctx: AgentContext, data: str, *, step_id: str | None) -> PromptSpec:
        scan = scan_for_injection(data)
        if scan.risk is not InjectionRisk.NONE:
            await ctx.emit(
                EventType.GUARDRAIL_TRIGGERED,
                agent=self.profile.name.value,
                guardrail="prompt_injection",
                target="user_request",
                risk=scan.risk.value,
                signals=scan.signals,
                action="blocked" if scan.is_high else "flagged",
            )
        if scan.is_high:
            return rejected_spec(scan)
        payload = {
            "user_request": data,
            "guardrail": {"risk": scan.risk.value, "signals": scan.signals},
        }
        return await self._generate(ctx, payload, PromptSpec, step_id=step_id)

    def summarize(self, value: PromptSpec) -> str:
        return f"[{value.feasibility.value}] {value.objective}"[:240]
