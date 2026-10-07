"""Evaluation cases: an input, the expected behaviour in words, and machine-checkable criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.common import StrictModel

DATA_DIR = Path(__file__).resolve().parent / "data"

Category = Literal[
    "simple", "complex", "ambiguous", "error", "prompt_injection", "rag", "tool_calling", "data"
]


class FinalStatusIs(StrictModel):
    type: Literal["final_status"]
    value: Literal["completed", "failed", "rejected"]


class MinCriticScore(StrictModel):
    type: Literal["min_critic_score"]
    value: int = Field(ge=0, le=100)


class AgentUsed(StrictModel):
    type: Literal["agent_used"]
    value: str


class AgentNotUsed(StrictModel):
    type: Literal["agent_not_used"]
    value: str


class KnowledgeConsulted(StrictModel):
    """The knowledge base grounded the run — by the RAG agent or a knowledge_search call.

    Outcome-based on purpose: a real planner may route document analysis to the research agent
    rather than the RAG agent, and both are valid plans.
    """

    type: Literal["knowledge_consulted"]


class ToolCalled(StrictModel):
    """A tool ran successfully; with a list, any of them (alternative tools for the same need)."""

    type: Literal["tool_called"]
    value: str | list[str]


class ToolNotCalled(StrictModel):
    type: Literal["tool_not_called"]
    value: str


class AnswerContains(StrictModel):
    type: Literal["answer_contains"]
    any_of: list[str] = Field(min_length=1)


class AnswerNotContains(StrictModel):
    type: Literal["answer_not_contains"]
    values: list[str] = Field(min_length=1)


class MinSources(StrictModel):
    type: Literal["min_sources"]
    value: int = Field(ge=0)


class ExpectedSources(StrictModel):
    """Retrieval relevance: each expected title fragment must appear among the cited sources."""

    type: Literal["expected_sources"]
    titles_contain: list[str] = Field(min_length=1)


class GuardrailTriggered(StrictModel):
    type: Literal["guardrail"]
    value: str


class MaxLlmCalls(StrictModel):
    type: Literal["max_llm_calls"]
    value: int = Field(ge=0)


class MaxRetries(StrictModel):
    type: Literal["max_retries"]
    value: int = Field(ge=0)


class WarningContains(StrictModel):
    type: Literal["warning_contains"]
    value: str


class AbortReasonContains(StrictModel):
    type: Literal["abort_reason_contains"]
    value: str


EvalCriterion = Annotated[
    FinalStatusIs
    | MinCriticScore
    | AgentUsed
    | AgentNotUsed
    | KnowledgeConsulted
    | ToolCalled
    | ToolNotCalled
    | AnswerContains
    | AnswerNotContains
    | MinSources
    | ExpectedSources
    | GuardrailTriggered
    | MaxLlmCalls
    | MaxRetries
    | WarningContains
    | AbortReasonContains,
    Field(discriminator="type"),
]


class EvaluationCase(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    category: Category
    input: str = Field(min_length=10, max_length=8_000)
    expected_behavior: str
    criteria: list[EvalCriterion] = Field(min_length=1)
    limits: dict[str, float] = Field(
        default_factory=dict,
        description="RunLimits overrides for this case (e.g. max_agent_calls).",
    )


class EvaluationDataset(StrictModel):
    name: str
    version: str
    description: str
    cases: list[EvaluationCase] = Field(min_length=1)


def available_datasets() -> list[str]:
    return sorted(path.stem for path in DATA_DIR.glob("*.json"))


def load_dataset(name: str = "default") -> EvaluationDataset:
    if name not in available_datasets():
        raise ValueError(f"unknown dataset '{name}' (available: {', '.join(available_datasets())})")
    payload = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(payload)
