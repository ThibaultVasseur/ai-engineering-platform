"""Output of the Prompt Optimizer: a vague request turned into a structured specification."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.schemas.common import StrictModel

ShortText = Annotated[str, Field(min_length=2, max_length=300)]


class Feasibility(StrEnum):
    ACTIONABLE = "actionable"
    NEEDS_CLARIFICATION = "needs_clarification"  # proceed on explicit assumptions
    REJECTED = "rejected"  # unsafe or out of scope: the run stops here


class PromptSpec(StrictModel):
    objective: str = Field(min_length=10, max_length=600, description="One-sentence goal.")
    context: str = Field(max_length=1500, description="Relevant background for the agents.")
    constraints: list[ShortText] = Field(default_factory=list, max_length=10)
    requirements: list[ShortText] = Field(default_factory=list, max_length=12)
    deliverables: list[ShortText] = Field(default_factory=list, max_length=8)
    risks: list[ShortText] = Field(default_factory=list, max_length=10)
    acceptance_criteria: list[ShortText] = Field(default_factory=list, max_length=8)
    assumptions: list[ShortText] = Field(default_factory=list, max_length=8)
    clarifying_questions: list[ShortText] = Field(default_factory=list, max_length=5)
    needs_knowledge_base: bool = Field(
        description="True when internal documents should ground the answer."
    )
    feasibility: Feasibility
    rejection_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _consistent(self) -> PromptSpec:
        if self.feasibility is Feasibility.REJECTED:
            if not self.rejection_reason:
                raise ValueError("rejection_reason is required when feasibility is 'rejected'")
            return self
        missing = [
            name
            for name in ("requirements", "deliverables", "acceptance_criteria")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"an actionable spec needs at least one item in: {', '.join(missing)}")
        return self
