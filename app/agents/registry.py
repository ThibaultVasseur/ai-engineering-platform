"""Agent catalogue: mission, role and — crucially — the tools each agent may use.

Tool permissions are declared here, as data, and enforced by the tool registry at execution
time. An agent never sees (nor can call) a tool that is not in its allow-list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class AgentName(StrEnum):
    PROMPT_OPTIMIZER = "prompt_optimizer"
    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    DATA = "data"
    CODING = "coding"
    RAG = "rag"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"


@dataclass(frozen=True)
class AgentProfile:
    name: AgentName
    role: Literal["control", "worker", "review"]
    mission: str
    allowed_tools: frozenset[str]
    output_schema: str
    uses_llm: bool = True


AGENT_PROFILES: dict[AgentName, AgentProfile] = {
    profile.name: profile
    for profile in (
        AgentProfile(
            name=AgentName.PROMPT_OPTIMIZER,
            role="control",
            mission="Turn a raw request into a structured, verifiable specification; "
            "reject unsafe requests before any other agent sees them.",
            allowed_tools=frozenset(),
            output_schema="PromptSpec",
        ),
        AgentProfile(
            name=AgentName.PLANNER,
            role="control",
            mission="Decompose the specification into a small DAG of steps assigned to workers.",
            allowed_tools=frozenset(),
            output_schema="Plan",
        ),
        AgentProfile(
            name=AgentName.SUPERVISOR,
            role="control",
            mission="Schedule ready steps, pick and brief the worker, enforce limits, trigger "
            "review and rework (deterministic policy, optional LLM advice).",
            allowed_tools=frozenset(),
            output_schema="SupervisorDecision",
            uses_llm=False,
        ),
        AgentProfile(
            name=AgentName.RESEARCH,
            role="worker",
            mission="Gather facts with tools (knowledge base, past tasks, calculator) and "
            "report findings with their sources.",
            allowed_tools=frozenset({"knowledge_search", "task_lookup", "calculator"}),
            output_schema="ResearchReport",
        ),
        AgentProfile(
            name=AgentName.DATA,
            role="worker",
            mission="Analyse platform data through safe, parameterised read tools and compute "
            "metrics.",
            allowed_tools=frozenset({"database_read", "calculator", "database_write"}),
            output_schema="DataAnalysis",
        ),
        AgentProfile(
            name=AgentName.CODING,
            role="worker",
            mission="Design a technical solution (components, API, data model, code skeletons, "
            "tests). No shell, no file system: proposals only.",
            allowed_tools=frozenset({"knowledge_search"}),
            output_schema="CodeProposal",
        ),
        AgentProfile(
            name=AgentName.RAG,
            role="worker",
            mission="Answer a precise question from internal documents only, with verified "
            "citations; say so when the documents do not contain the answer.",
            allowed_tools=frozenset(),
            output_schema="RagAnswer",
        ),
        AgentProfile(
            name=AgentName.SYNTHESIZER,
            role="control",
            mission="Merge the step results into one coherent answer that addresses every "
            "acceptance criterion.",
            allowed_tools=frozenset(),
            output_schema="FinalDraft",
        ),
        AgentProfile(
            name=AgentName.CRITIC,
            role="review",
            mission="Check the draft against the acceptance criteria, flag errors and risks, "
            "and request targeted rework.",
            allowed_tools=frozenset(),
            output_schema="Review",
        ),
    )
}

WORKER_AGENTS: tuple[AgentName, ...] = (
    AgentName.RESEARCH,
    AgentName.DATA,
    AgentName.CODING,
    AgentName.RAG,
)
