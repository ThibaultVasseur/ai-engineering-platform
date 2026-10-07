"""Runtime context injected into the graph for one run (LangGraph ``context_schema``).

The graph is compiled once at start-up; everything that is specific to a run — tenant, budget,
tracer, tool permissions, the agents to use — travels in this context instead of being captured
in closures or globals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from app.agents.base import AgentContext, AgentOutcome
from app.agents.supervisor import Supervisor
from app.schemas.agent import WorkerInput, WorkerOutput
from app.schemas.plan import Plan
from app.schemas.review import FinalDraft, Review, ReviewInput, SynthesisInput
from app.schemas.spec import PromptSpec


class AgentRunner[InT, OutT](Protocol):
    async def run(
        self, ctx: AgentContext, data: InT, *, step_id: str | None = None, attempt: int = 1
    ) -> AgentOutcome[OutT]: ...


@dataclass(frozen=True)
class AgentSuite:
    optimizer: AgentRunner[str, PromptSpec]
    planner: AgentRunner[PromptSpec, Plan]
    supervisor: Supervisor
    workers: Mapping[str, AgentRunner[WorkerInput, WorkerOutput]]
    synthesizer: AgentRunner[SynthesisInput, FinalDraft]
    critic: AgentRunner[ReviewInput, Review]


@dataclass(frozen=True)
class RunContext:
    agent_ctx: AgentContext
    agents: AgentSuite
