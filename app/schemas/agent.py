"""Agent I/O contracts: worker inputs, typed worker outputs, supervisor decisions."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator

from app.llm.types import Usage
from app.schemas.common import StrictModel
from app.schemas.plan import PlanStep, WorkerName


def _canonical_label(value: object) -> object:
    """'[S1]' or ' s1 ' -> 'S1': models often copy the inline citation format into id fields."""
    return value.strip().strip("[]").strip().upper() if isinstance(value, str) else value


Confidence = Literal["low", "medium", "high"]
ShortText = Annotated[str, Field(min_length=2, max_length=300)]
SourceLabel = Annotated[str, BeforeValidator(_canonical_label), Field(pattern=r"^S\d{1,3}$")]


# --------------------------------------------------------------------------------------
# What a worker receives: only the context it needs
# --------------------------------------------------------------------------------------
class DependencyResult(BaseModel):
    step_id: str
    agent: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class WorkerInput(BaseModel):
    objective: str
    step: PlanStep
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependency_results: list[DependencyResult] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)  # critic issues when reworking
    instructions: str | None = None  # supervisor briefing (LLM strategy)
    attempt: int = 1


# --------------------------------------------------------------------------------------
# What workers produce
# --------------------------------------------------------------------------------------
class WorkerOutput(StrictModel):
    summary: str = Field(min_length=10, max_length=1500)
    confidence: Confidence

    def key_points(self) -> list[str]:
        return []

    def cited_sources(self) -> list[str]:
        return []


class Finding(StrictModel):
    claim: str = Field(min_length=5, max_length=400)
    evidence: str = Field(max_length=800)
    source_ids: list[SourceLabel] = Field(default_factory=list, max_length=5)


class ResearchReport(WorkerOutput):
    findings: list[Finding] = Field(min_length=1, max_length=10)
    gaps: list[ShortText] = Field(default_factory=list, max_length=5)

    def key_points(self) -> list[str]:
        return [finding.claim for finding in self.findings]

    def cited_sources(self) -> list[str]:
        return sorted({label for finding in self.findings for label in finding.source_ids})


class Metric(StrictModel):
    name: str = Field(min_length=2, max_length=100)
    value: float
    unit: str = Field(max_length=20)
    method: str = Field(max_length=300, description="How the value was obtained.")


class DataAnalysis(WorkerOutput):
    metrics: list[Metric] = Field(default_factory=list, max_length=15)
    insights: list[ShortText] = Field(default_factory=list, max_length=8)
    limitations: list[ShortText] = Field(default_factory=list, max_length=5)

    def key_points(self) -> list[str]:
        values = [
            f"{metric.name}: {metric.value:g} {metric.unit}".strip() for metric in self.metrics
        ]
        return values + list(self.insights)


class Component(StrictModel):
    name: str = Field(min_length=2, max_length=80)
    responsibility: str = Field(min_length=5, max_length=400)


class Endpoint(StrictModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(
        pattern=r"^/[\w/{}.:-]*$",
        max_length=200,
        description="URL path template without query string, e.g. /api/v1/quotes/{quote_id}.",
    )
    purpose: str = Field(max_length=300)

    @field_validator("path", mode="before")
    @classmethod
    def _path_only(cls, value: object) -> object:
        """'GET /quotes?status=sent' -> '/quotes': the method or a query string is formatting."""
        if not isinstance(value, str) or not value.split():
            return value
        tokens = value.split()
        path = next((token for token in tokens if token.startswith("/")), tokens[0])
        path = re.split(r"[?#]", path, maxsplit=1)[0]
        return path if path.startswith("/") else f"/{path}"


class Entity(StrictModel):
    name: str = Field(min_length=2, max_length=80)
    fields: list[Annotated[str, Field(max_length=120)]] = Field(max_length=20)


class CodeFile(StrictModel):
    # Proposals are never written to disk; the path is still kept relative and traversal-free.
    path: str = Field(pattern=r"^[\w-][\w./-]{0,199}$")
    language: str = Field(max_length=30)
    purpose: str = Field(max_length=300)
    content: str = Field(max_length=4000)

    @field_validator("path", mode="before")
    @classmethod
    def _relative(cls, value: object) -> object:
        """'./src/x.py', '/src/x.py' or 'src\\x.py' -> 'src/x.py' ('..' stays rejected below)."""
        if not isinstance(value, str):
            return value
        path = value.strip().replace("\\", "/")
        while path.startswith(("./", "/")):
            path = path.removeprefix("./").removeprefix("/")
        return path

    @field_validator("path")
    @classmethod
    def _no_traversal(cls, value: str) -> str:
        if ".." in value.split("/"):
            raise ValueError("file paths must not contain '..' segments")
        return value


class CodeProposal(WorkerOutput):
    components: list[Component] = Field(min_length=1, max_length=12)
    api_endpoints: list[Endpoint] = Field(default_factory=list, max_length=15)
    data_model: list[Entity] = Field(default_factory=list, max_length=10)
    files: list[CodeFile] = Field(default_factory=list, max_length=6)
    tests: list[ShortText] = Field(default_factory=list, max_length=10)
    risks: list[ShortText] = Field(default_factory=list, max_length=8)
    source_ids: list[SourceLabel] = Field(default_factory=list, max_length=10)

    def key_points(self) -> list[str]:
        points = [f"{c.name}: {c.responsibility}" for c in self.components]
        points += [f"{e.method} {e.path}" for e in self.api_endpoints]
        points += [f"entity {e.name}({', '.join(e.fields[:6])})" for e in self.data_model]
        return points

    def cited_sources(self) -> list[str]:
        return sorted(set(self.source_ids))


MAX_QUOTE_CHARS = 300


class Citation(StrictModel):
    source_id: SourceLabel = Field(description="The source_id of the excerpt (e.g. S1).")
    quote: str = Field(
        max_length=MAX_QUOTE_CHARS,
        description="A short excerpt copied word for word from that source's content, in the "
        "document's own language: one sentence or part of one. Never translate, paraphrase or "
        "merge passages; mark words left out inside the excerpt with '…'.",
    )

    @field_validator("quote", mode="before")
    @classmethod
    def _shorten(cls, value: object) -> object:
        """Cut an over-long quote at a word boundary, marked '…', instead of refusing it.

        Live models keep quoting whole passages whatever the stated limit; the beginning of a
        verbatim excerpt is still verbatim, and the quote is verified against its source after.
        """
        if not isinstance(value, str) or len(value.strip()) <= MAX_QUOTE_CHARS:
            return value
        cut = value.strip()[: MAX_QUOTE_CHARS - 1]
        if " " in cut:
            cut = cut[: cut.rfind(" ")]
        return cut.rstrip(" ,;:") + "…"


class RagAnswer(WorkerOutput):
    answer: str = Field(min_length=2, max_length=3000)
    citations: list[Citation] = Field(default_factory=list, max_length=10)
    missing_information: list[ShortText] = Field(default_factory=list, max_length=5)

    def key_points(self) -> list[str]:
        return [self.answer[:400]]

    def cited_sources(self) -> list[str]:
        return sorted({citation.source_id for citation in self.citations})


WORKER_OUTPUTS: dict[str, type[WorkerOutput]] = {
    "research": ResearchReport,
    "data": DataAnalysis,
    "coding": CodeProposal,
    "rag": RagAnswer,
}


# --------------------------------------------------------------------------------------
# Results as stored in the graph state
# --------------------------------------------------------------------------------------
class AgentResult(BaseModel):
    step_id: str
    agent: str
    attempt: int
    status: Literal["completed", "failed"]
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    output: dict[str, Any] | None = None
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0


# --------------------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------------------
class SupervisorAction(StrEnum):
    DISPATCH = "dispatch"
    SYNTHESIZE = "synthesize"
    FINALIZE = "finalize"


class SupervisorDecision(StrictModel):
    action: SupervisorAction
    step_id: str | None = None
    agent: WorkerName | None = None
    reason: str = Field(max_length=500)
    instructions: str | None = Field(default=None, max_length=1000)


class SupervisorChoice(StrictModel):
    """What the optional LLM strategy may decide: which ready step to run next and how."""

    step_id: str
    agent: WorkerName
    reason: str = Field(max_length=500)
    instructions: str = Field(max_length=1000)


class AgentInfo(BaseModel):
    name: str
    role: str
    mission: str
    tools: list[str]
    output_schema: str
    uses_llm: bool
    prompt_version: str | None
    model: str | None
