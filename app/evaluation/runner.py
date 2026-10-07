"""Runs an evaluation dataset through the real orchestration (same code path as the API)."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from app.evaluation.datasets import EvaluationCase, EvaluationDataset
from app.evaluation.evaluators import CaseOutcome, check
from app.evaluation.metrics import CaseResult, EvaluationSummary, summarise
from app.observability.tracing import EventType, InMemoryTracer
from app.orchestration.factory import RunDependencies, build_run_context
from app.orchestration.policies import RunLimits
from app.orchestration.runner import GraphRunner, RunAbortedError


class EvaluationReport(BaseModel):
    dataset: str
    dataset_version: str
    llm_provider: str
    llm_model: str
    started_at: datetime
    duration_ms: float
    summary: EvaluationSummary
    cases: list[CaseResult]


def diagnostics(outcome: CaseOutcome) -> dict[str, Any]:
    """What a reviewer needs to understand a case: answer, stops, review, plan, failures, tools."""
    final = outcome.final_result
    failures = [
        f"{event.agent}{f'/{event.step_id}' if event.step_id else ''}: "
        f"{event.payload.get('error', '')}"[:500]
        for event in outcome.events
        if event.type is EventType.AGENT_FAILED
    ]
    tools = [
        {"tool": c.tool, "agent": c.agent, "status": c.status.value, "arguments": c.arguments}
        for c in outcome.tool_calls
    ]
    spec = next(
        (
            {
                key: event.payload.get(key)
                for key in (
                    "feasibility",
                    "needs_knowledge_base",
                    "objective",
                    "acceptance_criteria",
                )
            }
            for event in outcome.events
            if event.type is EventType.PROMPT_OPTIMIZED
        ),
        None,
    )
    if final is None:
        return {"agent_failures": failures, "tools": tools, "spec": spec}
    return {
        "spec": spec,
        "answer": final.answer.full_text() if final.answer else None,
        "warnings": final.warnings,
        "abort_reason": final.abort_reason,
        "review_issues": [issue.description for issue in final.review.issues]
        if final.review
        else [],
        "steps": [
            f"{step.step_id} -> {step.agent or '-'} ({step.status}, {step.attempts} attempt(s))"
            for step in final.steps
        ],
        "agent_failures": failures,
        "tools": tools,
    }


def _limits(base: RunLimits, overrides: dict[str, float]) -> RunLimits:
    unknown = set(overrides) - set(RunLimits.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown limit override(s): {sorted(unknown)}")
    typed = {key: type(getattr(base, key))(value) for key, value in overrides.items()}
    return replace(base, **typed)


class EvaluationRunner:
    def __init__(
        self, deps: RunDependencies, *, tenant_id: str, runner: GraphRunner | None = None
    ) -> None:
        self._deps = deps
        self._tenant_id = tenant_id
        self._runner = runner or GraphRunner()

    async def run_case(self, case: EvaluationCase) -> CaseOutcome:
        deps = self._deps
        if case.limits:
            deps = replace(deps, limits=_limits(deps.limits, case.limits))
        tracer = InMemoryTracer()
        context = build_run_context(
            deps, run_id=f"eval-{case.id}", tenant_id=self._tenant_id, tracer=tracer
        )
        started = time.perf_counter()
        try:
            state = await self._runner.run(context, user_request=case.input)
        except RunAbortedError as exc:
            return CaseOutcome(
                final_result=None,
                tool_calls=context.agent_ctx.journal.tool_calls,
                events=tracer.events,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error=str(exc),
            )
        return CaseOutcome(
            final_result=state.get("final_result"),
            tool_calls=list(state.get("tool_results", [])),
            events=tracer.events,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def run(
        self, dataset: EvaluationDataset, *, case_ids: list[str] | None = None
    ) -> EvaluationReport:
        cases = [case for case in dataset.cases if not case_ids or case.id in case_ids]
        if not cases:
            raise ValueError("no evaluation case selected")
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        outcomes: list[CaseOutcome] = []
        results: list[CaseResult] = []
        for case in cases:
            outcome = await self.run_case(case)
            checks = [check(criterion, outcome) for criterion in case.criteria]
            final = outcome.final_result
            outcomes.append(outcome)
            results.append(
                CaseResult(
                    id=case.id,
                    category=case.category,
                    passed=all(c.passed for c in checks),
                    checks=checks,
                    final_status=final.status.value if final else None,
                    critic_score=final.review.score if final and final.review else None,
                    latency_ms=outcome.latency_ms,
                    llm_calls=final.metrics.llm_calls if final else 0,
                    tool_calls=len(outcome.tool_calls),
                    error=outcome.error,
                    **diagnostics(outcome),
                )
            )
        return EvaluationReport(
            dataset=dataset.name,
            dataset_version=dataset.version,
            llm_provider=self._deps.llm.provider,
            llm_model=self._deps.llm.model,
            started_at=started_at,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            summary=summarise(cases, outcomes, results),
            cases=results,
        )
