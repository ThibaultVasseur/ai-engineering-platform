"""Offline handlers for the synthesizer and the critic.

The offline critic is rule-based and explicit: each acceptance criterion is checked with a
verifiable test (citations present, HTTP endpoints and entities described, tenant isolation
mentioned, listed topics covered...). Unmet criteria become MAJOR issues pointing at the step
that should be reworked.
"""

from __future__ import annotations

import re

from app.core.text import keyword_coverage, normalized_keywords, strip_accents, truncate
from app.llm.providers.offline import OfflineReply
from app.llm.providers.offline.common import read_context, reply_json
from app.llm.types import LLMRequest
from app.schemas.review import (
    CriterionCoverage,
    CriterionResult,
    FinalDraft,
    IssueSeverity,
    Review,
    ReviewIssue,
    ReviewStatus,
    Section,
)

_CITATION = re.compile(r"\[S\d{1,3}\]")
_HTTP_ENDPOINT = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE) /")
_MISSING_INFORMATION_MARKERS = ("do not contain", "no document", "ne contiennent pas")
_CRITERION_FILLER = normalized_keywords(
    "answer explicitly addresses address covers cover specifies specify statements statement "
    "based proposal findings supported evidence computed identified described includes listed"
)


# --------------------------------------------------------------------------------------
# Synthesizer
# --------------------------------------------------------------------------------------
def handle_synthesizer(request: LLMRequest) -> OfflineReply:
    context = read_context(request)
    sections: list[Section] = []
    for step in context["steps"]:
        lines = [step["summary"], *[f"- {point}" for point in step["key_points"][:20]]]
        if step["source_ids"]:
            lines.append("Sources: " + ", ".join(f"[{label}]" for label in step["source_ids"]))
        sections.append(
            Section(heading=truncate(step["title"], 120), content=truncate("\n".join(lines), 6000))
        )

    def best_section(criterion: str) -> tuple[Section, float]:
        best = max(sections, key=lambda s: keyword_coverage(criterion, f"{s.heading} {s.content}"))
        return best, keyword_coverage(criterion, f"{best.heading} {best.content}")

    criteria = context["acceptance_criteria"][:8]
    feedback = context.get("review_feedback", [])
    if feedback:
        # Asked to fix the draft: make the mapping between criteria and content explicit.
        mapping = [
            f"- {criterion} -> see '{best_section(criterion)[0].heading}'" for criterion in criteria
        ]
        sections.append(
            Section(
                heading="Acceptance criteria traceability",
                content=truncate("\n".join(mapping), 6000),
            )
        )
        sections.append(
            Section(
                heading="Review feedback addressed",
                content=truncate("\n".join(f"- {item}" for item in feedback[:8]), 6000),
            )
        )

    coverage = []
    for criterion in criteria:
        best, score = best_section(criterion)
        coverage.append(
            CriterionCoverage(criterion=criterion[:300], addressed=score >= 0.5, where=best.heading)
        )
    open_questions = [f"To confirm: {item}" for item in context.get("assumptions", [])][:4]
    open_questions += context.get("clarifying_questions", [])[:4]
    summaries = " ".join(step["summary"] for step in context["steps"])
    draft = FinalDraft(
        title=truncate(context["objective"], 150),
        executive_summary=truncate(summaries or context["objective"], 1500),
        sections=sections[:10],
        recommendations=[
            "Validate the proposed design with the domain experts before implementation",
            "Deliver the smallest slice that satisfies the acceptance criteria first",
        ],
        open_questions=[truncate(item, 300) for item in open_questions][:8],
        criteria_coverage=coverage,
    )
    return reply_json(draft)


# --------------------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------------------
def _step_for(steps: list[dict[str, str]], *agents: str) -> str | None:
    return next((step["step_id"] for step in steps if step["agent"] in agents), None)


def evaluate_criterion(
    criterion: str, draft_text: str, steps: list[dict[str, str]]
) -> tuple[bool, str, str | None]:
    """(satisfied, evidence, step to rework) for one acceptance criterion."""
    wanted = strip_accents(criterion.lower())
    text = strip_accents(draft_text.lower())
    if "cite" in wanted or "source" in wanted:
        found = len(_CITATION.findall(draft_text))
        if not found and any(marker in text for marker in _MISSING_INFORMATION_MARKERS):
            return True, "no document-based statement: missing information is reported", None
        return bool(found), f"{found} [S#] citation(s)", _step_for(steps, "rag", "research")
    if "metric" in wanted or "platform data" in wanted:
        has_numbers = bool(re.search(r"\d", draft_text))
        return (
            has_numbers,
            "figures present" if has_numbers else "no figure",
            _step_for(steps, "data"),
        )
    if "endpoint" in wanted or "data model" in wanted or "component" in wanted:
        ok = bool(_HTTP_ENDPOINT.search(draft_text)) and "entity" in text
        return (
            ok,
            "endpoints and entities described" if ok else "API or data model missing",
            _step_for(steps, "coding"),
        )
    if "security" in wanted or "isolation" in wanted:
        ok = any(
            word in text
            for word in ("tenant", "isolation", "rls", "row-level", "securite", "security")
        )
        return (
            ok,
            "isolation discussed" if ok else "no isolation discussion",
            _step_for(steps, "coding"),
        )
    if ":" in criterion:
        if any(marker in text for marker in _MISSING_INFORMATION_MARKERS):
            return True, "the draft states that the information is not available", None
        share = keyword_coverage(criterion.split(":", 1)[1], draft_text)
        return share >= 0.6, f"{share:.0%} of the listed topics appear in the draft", None
    words = normalized_keywords(criterion) - _CRITERION_FILLER
    share = len(words & normalized_keywords(draft_text)) / len(words) if words else 1.0
    return share >= 0.5, f"{share:.0%} of the criterion keywords appear", None


def handle_critic(request: LLMRequest) -> OfflineReply:
    context = read_context(request)
    draft = FinalDraft.model_validate(context["draft"])
    text = draft.body_text()  # the title restates the request: it proves nothing
    steps = context["plan_steps"]
    step_texts = {
        step["step_id"]: "\n".join([step["summary"], *step["key_points"]]) for step in steps
    }
    results: list[CriterionResult] = []
    issues: list[ReviewIssue] = []
    for criterion in context["acceptance_criteria"][:10]:
        satisfied, evidence, step_id = evaluate_criterion(criterion, text, steps)
        if not satisfied and step_id is not None:
            # Blame the right party: if the step output satisfies the criterion, the problem is
            # in the synthesis, and reworking the step would not fix anything.
            in_step, _, _ = evaluate_criterion(criterion, step_texts.get(step_id, ""), steps)
            if in_step:
                step_id = None
                evidence += "; present in the step output but missing from the draft"
        results.append(
            CriterionResult(criterion=criterion[:300], satisfied=satisfied, evidence=evidence[:300])
        )
        if not satisfied:
            issues.append(
                ReviewIssue(
                    severity=IssueSeverity.MAJOR,
                    description=truncate(f"Criterion not satisfied: {criterion} ({evidence})", 500),
                    step_id=step_id,
                    criterion=criterion[:300],
                )
            )
    for item in draft.criteria_coverage:
        if not item.addressed:
            issues.append(
                ReviewIssue(
                    severity=IssueSeverity.MAJOR,
                    description=truncate(
                        f"The draft does not explicitly address the criterion: {item.criterion}",
                        500,
                    ),
                    criterion=item.criterion,
                )
            )
    total = max(len(results), 1)
    explicitness_issues = sum(1 for item in draft.criteria_coverage if not item.addressed)
    score = round(100 * sum(result.satisfied for result in results) / total)
    score = max(0, score - 10 * explicitness_issues)
    passed = score >= context["pass_threshold"] and not issues
    review = Review(
        status=ReviewStatus.PASS if passed else ReviewStatus.FAIL,
        score=score,
        issues=issues,
        suggestions=[
            truncate(f"Rework '{issue.step_id}' to satisfy: {issue.criterion}", 300)
            if issue.step_id
            else truncate(f"Make the answer explicitly cover: {issue.criterion}", 300)
            for issue in issues
        ][:10],
        criteria_results=results,
        rework_steps=sorted({issue.step_id for issue in issues if issue.step_id}),
    )
    return reply_json(review)
