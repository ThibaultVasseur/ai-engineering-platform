"""Aggregate metrics over an evaluation run."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.datasets import EvaluationCase, ExpectedSources
from app.evaluation.evaluators import CaseOutcome, CriterionCheck, expected_source_hits
from app.schemas.tool import ToolCallStatus


class CaseResult(BaseModel):
    id: str
    category: str
    passed: bool
    checks: list[CriterionCheck]
    final_status: str | None
    critic_score: int | None
    latency_ms: float
    llm_calls: int
    tool_calls: int
    error: str | None = None
    # Diagnostics: a failed case must be explainable from the report alone (a lesson from the
    # first live evaluation, where "none of the expected terms" was all there was to go on).
    spec: dict[str, Any] | None = None  # what the optimizer understood (objective, criteria...)
    answer: str | None = None
    warnings: list[str] = Field(default_factory=list)
    abort_reason: str | None = None
    review_issues: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    agent_failures: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    cases: int
    passed: int
    success_rate: float
    by_category: dict[str, dict[str, Any]]
    avg_critic_score: float | None
    tool_calls: int
    tool_success_rate: float | None
    tool_denials: int
    retrieval_relevance: float | None
    latency_ms_p50: float
    latency_ms_p95: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return round(ordered[index], 1)


def summarise(
    cases: list[EvaluationCase], outcomes: list[CaseOutcome], results: list[CaseResult]
) -> EvaluationSummary:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result.passed)

    scores = [r.critic_score for r in results if r.critic_score is not None]
    tool_records = [record for outcome in outcomes for record in outcome.tool_calls]
    executed = [r for r in tool_records if r.status is not ToolCallStatus.DENIED]
    retrieval: list[float] = []
    for case, outcome in zip(cases, outcomes, strict=True):
        for criterion in case.criteria:
            if isinstance(criterion, ExpectedSources):
                hits, total = expected_source_hits(criterion, outcome)
                retrieval.append(hits / total)

    finals = [o.final_result for o in outcomes if o.final_result is not None]
    passed = sum(result.passed for result in results)
    return EvaluationSummary(
        cases=len(results),
        passed=passed,
        success_rate=round(passed / len(results), 3) if results else 0.0,
        by_category={
            category: {
                "cases": len(flags),
                "passed": sum(flags),
                "success_rate": round(sum(flags) / len(flags), 3),
            }
            for category, flags in sorted(by_category.items())
        },
        avg_critic_score=round(statistics.fmean(scores), 1) if scores else None,
        tool_calls=len(tool_records),
        tool_success_rate=round(
            sum(r.status is ToolCallStatus.OK for r in executed) / len(executed), 3
        )
        if executed
        else None,
        tool_denials=sum(r.status is ToolCallStatus.DENIED for r in tool_records),
        retrieval_relevance=round(statistics.fmean(retrieval), 3) if retrieval else None,
        latency_ms_p50=_percentile([r.latency_ms for r in results], 0.5),
        latency_ms_p95=_percentile([r.latency_ms for r in results], 0.95),
        llm_calls=sum(f.metrics.llm_calls for f in finals),
        input_tokens=sum(f.metrics.input_tokens for f in finals),
        output_tokens=sum(f.metrics.output_tokens for f in finals),
        cost_usd=round(sum(f.metrics.cost_usd for f in finals), 6),
    )
