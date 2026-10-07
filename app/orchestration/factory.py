"""Assembly of the agents and of the per-run context.

``RunDependencies`` holds what is shared by every run (LLM provider, tools, retriever...).
``build_run_context`` creates what is specific to ONE run: its budget, journal and tracer, a
metered LLM client enforcing that budget, and the tool executor bound to the run's tenant.
The API, the CLI and the evaluation runner all go through this function.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from app.agents.base import AgentContext
from app.agents.coding import CodingAgent
from app.agents.critic import CriticAgent
from app.agents.data import DataAgent
from app.agents.planner import PlannerAgent
from app.agents.prompt_optimizer import PromptOptimizerAgent
from app.agents.rag import RagAgent
from app.agents.research import ResearchAgent
from app.agents.supervisor import Supervisor
from app.agents.synthesizer import SynthesizerAgent
from app.llm.metering import MeteredLLMClient
from app.llm.pricing import PriceBook
from app.llm.types import LLMClient
from app.observability.tracing import Tracer
from app.orchestration.context import AgentSuite, RunContext
from app.orchestration.journal import RunJournal
from app.orchestration.policies import RunBudget, RunLimits
from app.tools.base import KnowledgeRetriever, PlatformData
from app.tools.registry import ToolRegistry

SupervisorStrategy = Literal["rules", "llm"]


def build_agent_suite(
    *, supervisor_strategy: SupervisorStrategy = "rules", rag_top_k: int = 5
) -> AgentSuite:
    return AgentSuite(
        optimizer=PromptOptimizerAgent(),
        planner=PlannerAgent(),
        supervisor=Supervisor(supervisor_strategy),
        workers={
            "research": ResearchAgent(),
            "data": DataAgent(),
            "coding": CodingAgent(),
            "rag": RagAgent(top_k=rag_top_k),
        },
        synthesizer=SynthesizerAgent(),
        critic=CriticAgent(),
    )


@dataclass(frozen=True)
class RunDependencies:
    llm: LLMClient
    prices: PriceBook
    tools: ToolRegistry
    agents: AgentSuite
    limits: RunLimits
    retriever: KnowledgeRetriever | None = None
    platform_data: PlatformData | None = None
    structured_max_attempts: int = 3


def build_run_context(
    deps: RunDependencies,
    *,
    run_id: str,
    tenant_id: str,
    tracer: Tracer,
    allow_tool_writes: bool = False,
    supervisor_strategy: SupervisorStrategy | None = None,
) -> RunContext:
    budget = RunBudget(deps.limits)
    journal = RunJournal()
    llm = MeteredLLMClient(deps.llm, budget=budget, tracer=tracer, prices=deps.prices)
    tools = deps.tools.bind(
        tenant_id=tenant_id,
        run_id=run_id,
        tracer=tracer,
        journal=journal,
        allow_writes=allow_tool_writes,
        knowledge=deps.retriever,
        data=deps.platform_data,
    )
    agent_ctx = AgentContext(
        run_id=run_id,
        tenant_id=tenant_id,
        llm=llm,
        tracer=tracer,
        budget=budget,
        limits=deps.limits,
        tools=tools,
        retriever=deps.retriever,
        journal=journal,
        structured_max_attempts=deps.structured_max_attempts,
    )
    agents = deps.agents
    if supervisor_strategy is not None and supervisor_strategy != agents.supervisor.strategy:
        agents = replace(agents, supervisor=Supervisor(supervisor_strategy))
    return RunContext(agent_ctx=agent_ctx, agents=agents)
