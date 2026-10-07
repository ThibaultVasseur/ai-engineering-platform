"""Helpers shared by the worker agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agents.base import AgentContext
from app.schemas.agent import WorkerInput, WorkerOutput


def worker_payload(data: WorkerInput) -> dict[str, Any]:
    """The JSON context a worker sees: its step and the minimum around it."""
    return {
        "objective": data.objective,
        "step": {
            "id": data.step.id,
            "type": data.step.type.value,
            "title": data.step.title,
            "description": data.step.description,
            "success_criteria": data.step.success_criteria,
        },
        "constraints": data.constraints,
        "acceptance_criteria": data.acceptance_criteria,
        "dependency_results": [result.model_dump() for result in data.dependency_results],
        "feedback": data.feedback,
        "instructions": data.instructions,
        "attempt": data.attempt,
    }


def known_sources_only(ctx: AgentContext) -> Callable[[WorkerOutput], None]:
    """Reject outputs citing source ids that were never retrieved in this run."""

    def check(value: WorkerOutput) -> None:
        known = ctx.journal.sources.labels()
        unknown = sorted(set(value.cited_sources()) - known)
        if unknown:
            raise ValueError(
                f"unknown source ids {unknown}: cite only source_id values returned by "
                f"knowledge_search in this run ({sorted(known) or 'none retrieved yet'})"
            )

    return check
