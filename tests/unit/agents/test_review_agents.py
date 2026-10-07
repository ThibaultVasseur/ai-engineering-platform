import json

import pytest

from app.agents.critic import apply_review_policy
from app.agents.synthesizer import SynthesizerAgent
from app.llm.providers.offline.review import evaluate_criterion
from app.observability.tracing import EventType
from app.schemas.review import (
    FinalDraft,
    IssueSeverity,
    Review,
    ReviewInput,
    ReviewIssue,
    ReviewStatus,
    Section,
    SourceDigest,
    StepDigest,
    SynthesisInput,
)
from tests.fakes import ScriptedLLM, make_context

STEPS = [
    StepDigest(step_id="kb", title="Docs", agent="rag", summary="Docs"),
    StepDigest(
        step_id="design",
        title="Design",
        agent="coding",
        summary="Quote service",
        key_points=["GET /api/v1/quotes", "entity Quote(id, tenant_id)"],
    ),
]


def draft(text: str = "Quotes are valid 30 days [S1].") -> FinalDraft:
    return FinalDraft(
        title="Quote module",
        executive_summary="Summary of the design.",
        sections=[Section(heading="Design", content=text)],
    )


def review_input(text: str = "Quotes are valid 30 days [S1].") -> ReviewInput:
    return ReviewInput(
        objective="Design a quote module",
        acceptance_criteria=["c1"],
        plan_steps=STEPS,
        draft=draft(text),
        available_sources=["S1"],
        pass_threshold=70,
    )


def review(status: ReviewStatus, score: int, **extra: object) -> Review:
    return Review.model_validate({"status": status, "score": score, **extra})


# --------------------------------------------------------------------------------------
# Review policy: deterministic, can only make the verdict stricter
# --------------------------------------------------------------------------------------
def test_model_pass_below_threshold_becomes_fail() -> None:
    result = apply_review_policy(review(ReviewStatus.PASS, 60), review_input())
    assert result.status is ReviewStatus.FAIL


def test_model_pass_with_critical_issue_becomes_fail() -> None:
    critical = ReviewIssue(severity=IssueSeverity.CRITICAL, description="Leaks tenant data")
    result = apply_review_policy(review(ReviewStatus.PASS, 95, issues=[critical]), review_input())
    assert result.status is ReviewStatus.FAIL


def test_model_fail_is_never_upgraded() -> None:
    result = apply_review_policy(review(ReviewStatus.FAIL, 99), review_input())
    assert result.status is ReviewStatus.FAIL


def test_well_supported_pass_is_kept() -> None:
    result = apply_review_policy(review(ReviewStatus.PASS, 85), review_input())
    assert result.status is ReviewStatus.PASS


def test_unknown_citations_are_added_as_critical_whatever_the_model_said() -> None:
    result = apply_review_policy(
        review(ReviewStatus.PASS, 95), review_input("Totals use Decimal [S1] and VAT [S9].")
    )
    assert result.status is ReviewStatus.FAIL
    assert any("S9" in issue.description for issue in result.issues)


def test_rework_targets_are_restricted_to_plan_steps() -> None:
    issue = ReviewIssue(severity=IssueSeverity.MAJOR, description="Missing taxes", step_id="design")
    result = apply_review_policy(
        review(ReviewStatus.FAIL, 40, issues=[issue], rework_steps=["ghost", "kb"]),
        review_input(),
    )
    assert result.rework_steps == ["kb", "design"]


# --------------------------------------------------------------------------------------
# Synthesizer
# --------------------------------------------------------------------------------------
async def test_synthesizer_cannot_cite_unknown_sources() -> None:
    bad = draft("Totals use Decimal [S7].").model_dump()
    good = draft("Totals use Decimal [S1].").model_dump()
    llm = ScriptedLLM([json.dumps(bad), json.dumps(good)])
    context, tracer = make_context(llm)
    data = SynthesisInput(
        objective="Design a quote module",
        deliverables=["Design"],
        acceptance_criteria=["c1"],
        steps=STEPS,
        sources=[SourceDigest(label="S1", title="Guide", excerpt="Decimal")],
    )
    outcome = await SynthesizerAgent().run(context, data)
    assert "[S1]" in outcome.value.full_text()
    assert "unknown sources ['S7']" in llm.requests[1].messages[-1].text
    assert tracer.of_type(EventType.STRUCTURED_OUTPUT_REPAIRED)


# --------------------------------------------------------------------------------------
# Offline critic: explicit checks and correct blame assignment
# --------------------------------------------------------------------------------------
STEP_VIEW = [{"step_id": s.step_id, "agent": s.agent} for s in STEPS]


@pytest.mark.parametrize(
    ("criterion", "text", "expected"),
    [
        ("Statements based on internal documents cite their sources", "Valid 30 days [S1].", True),
        ("Statements based on internal documents cite their sources", "Valid 30 days.", False),
        (
            "Statements based on internal documents cite their sources",
            "The internal documents do not contain information about this question.",
            True,
        ),
        (
            "The proposal specifies components, API endpoints and the data model",
            "POST /api/v1/quotes; entity Quote",
            True,
        ),
        (
            "Security and multi-tenant data isolation risks are addressed",
            "Use RLS per tenant",
            True,
        ),
        ("The answer explicitly addresses: quotes, taxes", "Only quotes here", False),
    ],
)
def test_offline_criterion_checks(criterion: str, text: str, expected: bool) -> None:
    assert evaluate_criterion(criterion, text, STEP_VIEW)[0] is expected


def test_missing_design_in_draft_blames_the_coding_step() -> None:
    _, _, step_id = evaluate_criterion(
        "The proposal specifies components, API endpoints and the data model", "No API", STEP_VIEW
    )
    assert step_id == "design"
