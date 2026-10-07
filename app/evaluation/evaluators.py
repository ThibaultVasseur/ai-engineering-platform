"""Deterministic checks of one evaluation criterion against one run outcome."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from app.evaluation.datasets import (
    AbortReasonContains,
    AgentNotUsed,
    AgentUsed,
    AnswerContains,
    AnswerNotContains,
    EvalCriterion,
    ExpectedSources,
    FinalStatusIs,
    GuardrailTriggered,
    KnowledgeConsulted,
    MaxLlmCalls,
    MaxRetries,
    MinCriticScore,
    MinSources,
    ToolCalled,
    ToolNotCalled,
    WarningContains,
)
from app.observability.tracing import EventType, TraceEvent
from app.schemas.run import FinalResult
from app.schemas.tool import ToolCallRecord, ToolCallStatus


@dataclass
class CaseOutcome:
    final_result: FinalResult | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def agents_used(self) -> set[str]:
        return {
            event.agent
            for event in self.events
            if event.type is EventType.AGENT_COMPLETED and event.agent
        }

    @property
    def answer_text(self) -> str:
        if self.final_result is None or self.final_result.answer is None:
            return ""
        return self.final_result.answer.full_text()

    @property
    def guardrails(self) -> set[str]:
        return {
            str(event.payload.get("guardrail"))
            for event in self.events
            if event.type is EventType.GUARDRAIL_TRIGGERED
        }


class CriterionCheck(BaseModel):
    type: str
    passed: bool
    detail: str


def check(criterion: EvalCriterion, outcome: CaseOutcome) -> CriterionCheck:
    passed, detail = _evaluate(criterion, outcome)
    return CriterionCheck(type=criterion.type, passed=passed, detail=detail)


def _evaluate(criterion: EvalCriterion, outcome: CaseOutcome) -> tuple[bool, str]:
    result = outcome.final_result
    match criterion:
        case FinalStatusIs(value=value):
            actual = result.status.value if result else f"error: {outcome.error}"
            return actual == value, f"final status: {actual}"
        case MinCriticScore(value=value):
            score = result.review.score if result and result.review else None
            return score is not None and score >= value, f"critic score: {score}"
        case AgentUsed(value=value):
            return value in outcome.agents_used, f"agents: {sorted(outcome.agents_used)}"
        case AgentNotUsed(value=value):
            return value not in outcome.agents_used, f"agents: {sorted(outcome.agents_used)}"
        case KnowledgeConsulted():
            searched = any(
                c.tool == "knowledge_search" and c.status is ToolCallStatus.OK
                for c in outcome.tool_calls
            )
            rag = "rag" in outcome.agents_used
            return searched or rag, f"rag agent: {rag}, knowledge_search: {searched}"
        case ToolCalled(value=value):
            wanted = {value} if isinstance(value, str) else set(value)
            ok = any(c.tool in wanted and c.status is ToolCallStatus.OK for c in outcome.tool_calls)
            return ok, f"tools: {sorted({c.tool for c in outcome.tool_calls})}"
        case ToolNotCalled(value=value):
            ok = not any(
                c.tool == value and c.status is ToolCallStatus.OK for c in outcome.tool_calls
            )
            return ok, f"tools: {sorted({c.tool for c in outcome.tool_calls})}"
        case AnswerContains(any_of=values):
            text = outcome.answer_text.lower()
            found = [value for value in values if value.lower() in text]
            return bool(found), f"found: {found}" if found else "none of the expected terms"
        case AnswerNotContains(values=values):
            text = outcome.answer_text.lower()
            leaked = [value for value in values if value.lower() in text]
            return not leaked, f"unexpected: {leaked}" if leaked else "clean"
        case MinSources(value=value):
            count = len(result.sources) if result else 0
            return count >= value, f"{count} source(s)"
        case ExpectedSources(titles_contain=fragments):
            titles = [source.title.lower() for source in result.sources] if result else []
            hits = [f for f in fragments if any(f.lower() in title for title in titles)]
            return len(hits) == len(fragments), f"{len(hits)}/{len(fragments)} expected sources"
        case GuardrailTriggered(value=value):
            return value in outcome.guardrails, f"guardrails: {sorted(outcome.guardrails)}"
        case MaxLlmCalls(value=value):
            calls = result.metrics.llm_calls if result else None
            return calls is not None and calls <= value, f"llm calls: {calls}"
        case MaxRetries(value=value):
            retries = result.metrics.retry_count if result else None
            return retries is not None and retries <= value, f"retries: {retries}"
        case WarningContains(value=value):
            warnings = result.warnings if result else []
            return any(value.lower() in w.lower() for w in warnings), f"warnings: {warnings}"
        case AbortReasonContains(value=value):
            reason = result.abort_reason if result else outcome.error
            return bool(reason and value.lower() in reason.lower()), f"abort reason: {reason}"
    raise ValueError(f"unsupported criterion: {criterion!r}")


def expected_source_hits(criterion: ExpectedSources, outcome: CaseOutcome) -> tuple[int, int]:
    titles = (
        [source.title.lower() for source in outcome.final_result.sources]
        if outcome.final_result
        else []
    )
    hits = sum(any(f.lower() in t for t in titles) for f in criterion.titles_contain)
    return hits, len(criterion.titles_contain)
