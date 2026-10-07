"""The evaluation dataset as a regression suite (offline provider, in-memory runtime).

On the offline simulator every case must pass: a failure means a change broke orchestration,
retrieval, tools or guardrails. With a real LLM the same suite measures quality instead.
"""

from __future__ import annotations

from collections import Counter

import pytest

from app.evaluation.datasets import (
    KnowledgeConsulted,
    ToolCalled,
    available_datasets,
    load_dataset,
)
from app.evaluation.evaluators import CaseOutcome, check
from app.evaluation.runner import EvaluationReport, EvaluationRunner
from app.observability.tracing import EventType, TraceEvent
from app.orchestration.local import build_in_memory_dependencies
from app.schemas.run import FinalResult, FinalStatus
from app.schemas.tool import ToolCallRecord, ToolCallStatus
from tests.support import make_settings

TENANT = "evaluation"


@pytest.fixture(scope="module")
async def report() -> EvaluationReport:
    deps = await build_in_memory_dependencies(make_settings(), tenant_id=TENANT)
    return await EvaluationRunner(deps, tenant_id=TENANT).run(load_dataset("default"))


def test_dataset_covers_the_required_categories() -> None:
    dataset = load_dataset("default")
    assert len(dataset.cases) >= 10
    categories = Counter(case.category for case in dataset.cases)
    for required in (
        "simple",
        "complex",
        "ambiguous",
        "error",
        "prompt_injection",
        "rag",
        "tool_calling",
    ):
        assert categories[required] >= 1, required
    assert len({case.id for case in dataset.cases}) == len(dataset.cases)
    assert "default" in available_datasets()


async def test_every_case_passes_offline(report: EvaluationReport) -> None:
    failures = {
        case.id: [f"{c.type}: {c.detail}" for c in case.checks if not c.passed]
        for case in report.cases
        if not case.passed
    }
    assert failures == {}
    assert report.summary.success_rate == 1.0


async def test_summary_metrics_are_computed(report: EvaluationReport) -> None:
    summary = report.summary
    assert summary.cases == len(report.cases)
    assert summary.tool_calls > 0
    assert summary.tool_success_rate == 1.0
    assert summary.retrieval_relevance == 1.0
    assert summary.avg_critic_score is not None
    assert summary.latency_ms_p95 >= summary.latency_ms_p50
    assert summary.input_tokens > 0
    assert set(summary.by_category) >= {"rag", "prompt_injection", "complex"}


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("nope")


def test_evaluators_report_failures_with_details() -> None:
    case = next(c for c in load_dataset().cases if c.id == "rag-quote-validity")
    outcome = CaseOutcome(final_result=FinalResult(status=FinalStatus.FAILED, accepted=False))
    checks = [check(criterion, outcome) for criterion in case.criteria]
    assert not any(c.passed for c in checks)
    assert checks[0].detail == "final status: failed"


def test_knowledge_can_be_consulted_by_the_rag_agent_or_the_search_tool() -> None:
    """A real planner may send document analysis to the research agent: both routes count."""
    criterion = KnowledgeConsulted(type="knowledge_consulted")
    assert not check(criterion, CaseOutcome(final_result=None)).passed
    searched = CaseOutcome(
        final_result=None,
        tool_calls=[
            ToolCallRecord(tool="knowledge_search", agent="research", status=ToolCallStatus.OK)
        ],
    )
    assert check(criterion, searched).passed
    denied = CaseOutcome(
        final_result=None,
        tool_calls=[
            ToolCallRecord(tool="knowledge_search", agent="data", status=ToolCallStatus.DENIED)
        ],
    )
    assert not check(criterion, denied).passed
    rag = CaseOutcome(
        final_result=None, events=[TraceEvent(type=EventType.AGENT_COMPLETED, agent="rag")]
    )
    assert check(criterion, rag).passed


def test_reports_explain_each_case(report: EvaluationReport) -> None:
    """A failed live case must be explainable from the JSON report alone."""
    by_id = {case.id: case for case in report.cases}
    answered = by_id["rag-quote-validity"]
    assert answered.answer is not None
    assert "30 jours" in answered.answer
    assert answered.steps
    assert "->" in answered.steps[0]
    assert answered.spec is not None
    assert answered.spec["acceptance_criteria"]
    stopped = by_id["agent-call-budget"]
    assert stopped.abort_reason is not None
    assert "max_agent_calls" in stopped.abort_reason
    assert by_id["injection-system-prompt"].warnings
    assert by_id["data-failure-rate"].tools[0]["tool"] == "database_read"


def test_alternative_tools_can_satisfy_one_need() -> None:
    criterion = ToolCalled(type="tool_called", value=["task_lookup", "database_read"])

    def ran(tool: str) -> CaseOutcome:
        record = ToolCallRecord(tool=tool, agent="data", status=ToolCallStatus.OK)
        return CaseOutcome(final_result=None, tool_calls=[record])

    assert check(criterion, ran("database_read")).passed
    assert check(criterion, ran("task_lookup")).passed
    assert not check(criterion, ran("calculator")).passed
