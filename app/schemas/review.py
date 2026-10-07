"""Synthesised draft and critic review."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from app.schemas.common import StrictModel

ShortText = Annotated[str, Field(min_length=2, max_length=300)]


class Section(StrictModel):
    heading: str = Field(min_length=2, max_length=120)
    content: str = Field(min_length=2, max_length=6000, description="Markdown.")


class CriterionCoverage(StrictModel):
    criterion: str = Field(max_length=300)
    addressed: bool
    where: str = Field(max_length=200, description="Section heading that addresses it.")


class FinalDraft(StrictModel):
    title: str = Field(min_length=3, max_length=150)
    executive_summary: str = Field(min_length=10, max_length=1500)
    sections: list[Section] = Field(min_length=1, max_length=10)
    recommendations: list[ShortText] = Field(default_factory=list, max_length=10)
    open_questions: list[ShortText] = Field(default_factory=list, max_length=8)
    criteria_coverage: list[CriterionCoverage] = Field(default_factory=list, max_length=8)

    def full_text(self) -> str:
        return f"{self.title}\n{self.body_text()}"

    def body_text(self) -> str:
        """Everything but the title (which often restates the request verbatim)."""
        parts = [self.executive_summary]
        for section in self.sections:
            parts.extend([section.heading, section.content])
        parts.extend(self.recommendations)
        return "\n".join(parts)


class IssueSeverity(StrEnum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class ReviewIssue(StrictModel):
    severity: IssueSeverity
    description: str = Field(min_length=5, max_length=500)
    step_id: str | None = Field(default=None, description="Plan step to rework, if any.")
    criterion: str | None = Field(default=None, max_length=300)


class CriterionResult(StrictModel):
    criterion: str = Field(max_length=300)
    satisfied: bool
    evidence: str = Field(max_length=300)


class ReviewStatus(StrEnum):
    PASS = "PASS"  # noqa: S105 - a verdict, not a password
    FAIL = "FAIL"


class Review(StrictModel):
    status: ReviewStatus
    score: int = Field(ge=0, le=100)
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=15)
    suggestions: list[ShortText] = Field(default_factory=list, max_length=10)
    criteria_results: list[CriterionResult] = Field(default_factory=list, max_length=10)
    rework_steps: list[str] = Field(default_factory=list, max_length=10)


# --------------------------------------------------------------------------------------
# Inputs of the synthesizer and the critic (built by the graph from the state)
# --------------------------------------------------------------------------------------
class StepDigest(BaseModel):
    step_id: str
    title: str
    agent: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    output: dict[str, Any] | None = None


class SourceDigest(BaseModel):
    label: str
    title: str
    excerpt: str


class SynthesisInput(BaseModel):
    user_request: str = ""  # the answer is written in the user's language
    objective: str
    deliverables: list[str]
    acceptance_criteria: list[str]
    assumptions: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    steps: list[StepDigest]
    sources: list[SourceDigest] = Field(default_factory=list)
    review_feedback: list[str] = Field(default_factory=list)


class ReviewInput(BaseModel):
    objective: str
    acceptance_criteria: list[str]
    plan_steps: list[StepDigest]
    draft: FinalDraft
    available_sources: list[str] = Field(default_factory=list)
    pass_threshold: int = 70
    round: int = 1
