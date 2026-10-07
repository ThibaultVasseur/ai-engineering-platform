"""Research agent: gathers evidence with tools and reports sourced findings."""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.agents.worker_support import known_sources_only, worker_payload
from app.llm.prompts import RESEARCH
from app.schemas.agent import ResearchReport, WorkerInput


class ResearchAgent(BaseAgent[WorkerInput, ResearchReport]):
    profile = AGENT_PROFILES[AgentName.RESEARCH]
    prompt = RESEARCH

    async def execute(
        self, ctx: AgentContext, data: WorkerInput, *, step_id: str | None
    ) -> ResearchReport:
        return await self._tool_loop(
            ctx,
            worker_payload(data),
            ResearchReport,
            step_id=step_id,
            validate=known_sources_only(ctx),
        )

    def summarize(self, value: ResearchReport) -> str:
        return value.summary[:240]
