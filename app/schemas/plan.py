"""Output of the Planner: a DAG of steps, each assigned to a specialised worker agent."""

from __future__ import annotations

import re
from collections import deque
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.core.text import strip_accents
from app.schemas.common import StrictModel

STEP_ID_MAX_CHARS = 40
_NOT_ID_CHARS = re.compile(r"[^a-z0-9]+")


def canonical_step_id(value: object) -> object:
    """Normalise a model-written step id instead of rejecting it ('Design-API' -> 'design_api').

    Live models keep writing long or hyphenated ids even when given the pattern; the spelling of
    an identifier carries no meaning, so ids (and the dependencies that reference them) are made
    canonical rather than failing the whole plan. Ids still invalid afterwards are rejected.
    """
    if not isinstance(value, str):
        return value
    slug = _NOT_ID_CHARS.sub("_", strip_accents(value).lower()).strip("_")
    if slug[:1].isdigit():
        slug = f"step_{slug}"
    return slug[:STEP_ID_MAX_CHARS].rstrip("_")


WorkerName = Literal["research", "data", "coding", "rag"]


class StepType(StrEnum):
    RESEARCH = "research"  # gather facts with tools (knowledge base, task history)
    DATA = "data"  # quantitative analysis of platform data
    CODING = "coding"  # design / implementation proposal
    KNOWLEDGE = "knowledge"  # grounded answer from internal documents (RAG)


# Which worker may execute which kind of step; the first one is the default.
CAPABLE_AGENTS: dict[StepType, tuple[WorkerName, ...]] = {
    StepType.RESEARCH: ("research", "rag"),
    StepType.DATA: ("data",),
    StepType.CODING: ("coding",),
    StepType.KNOWLEDGE: ("rag", "research"),
}

Criterion = Annotated[str, Field(min_length=3, max_length=300)]


class PlanStep(StrictModel):
    id: str = Field(
        pattern=r"^[a-z][a-z0-9_]{1,39}$",
        description="Short snake_case identifier, e.g. design_api.",
    )
    type: StepType
    title: str = Field(min_length=3, max_length=100)
    description: str = Field(min_length=10, max_length=800)
    recommended_agent: WorkerName
    dependencies: list[str] = Field(default_factory=list, max_length=6)
    priority: int = Field(ge=1, le=5, description="1 = most important")
    success_criteria: list[Criterion] = Field(min_length=1, max_length=5)

    _canonical_id = field_validator("id", mode="before")(canonical_step_id)

    @field_validator("dependencies", mode="before")
    @classmethod
    def _canonical_dependencies(cls, value: object) -> object:
        return [canonical_step_id(item) for item in value] if isinstance(value, list) else value


class Plan(StrictModel):
    rationale: str = Field(min_length=10, max_length=800)
    steps: list[PlanStep] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def _valid_dag(self) -> Plan:
        ids = [step.id for step in self.steps]
        duplicates = sorted({step_id for step_id in ids if ids.count(step_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate step ids: {', '.join(duplicates)}")
        known = set(ids)
        for step in self.steps:
            if step.id in step.dependencies:
                raise ValueError(f"step '{step.id}' depends on itself")
            unknown = [dep for dep in step.dependencies if dep not in known]
            if unknown:
                raise ValueError(f"step '{step.id}' depends on unknown steps: {', '.join(unknown)}")
        self.topological_order()  # raises on cycles
        return self

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; ties keep priority then declaration order."""
        indegree = {step.id: len(step.dependencies) for step in self.steps}
        dependents: dict[str, list[str]] = {step.id: [] for step in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                dependents[dep].append(step.id)
        rank = {step.id: (step.priority, index) for index, step in enumerate(self.steps)}
        ready = deque(sorted((s for s, d in indegree.items() if d == 0), key=rank.__getitem__))
        order: list[str] = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for child in dependents[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready = deque(sorted(ready, key=rank.__getitem__))
        if len(order) != len(self.steps):
            cyclic = sorted(set(indegree) - set(order))
            raise ValueError(f"dependency cycle between steps: {', '.join(cyclic)}")
        return order

    def step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)
