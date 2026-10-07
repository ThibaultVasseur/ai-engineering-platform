"""Coding agent: produces a technical design proposal.

It has no shell, no file system and no code execution — only read access to the knowledge base.
Its output is a structured proposal (components, API, data model, short skeletons, tests,
risks), never an action on the system.
"""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.agents.worker_support import known_sources_only, worker_payload
from app.llm.prompts import CODING
from app.schemas.agent import CodeProposal, WorkerInput


class CodingAgent(BaseAgent[WorkerInput, CodeProposal]):
    profile = AGENT_PROFILES[AgentName.CODING]
    prompt = CODING

    async def execute(
        self, ctx: AgentContext, data: WorkerInput, *, step_id: str | None
    ) -> CodeProposal:
        return await self._tool_loop(
            ctx,
            worker_payload(data),
            CodeProposal,
            step_id=step_id,
            validate=known_sources_only(ctx),
        )

    def summarize(self, value: CodeProposal) -> str:
        return value.summary[:240]
