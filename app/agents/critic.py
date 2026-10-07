"""Critic: reviews the synthesised draft; a deterministic policy has the last word.

The model produces the review, then ``apply_review_policy`` normalises it:
* rework targets are restricted to existing plan steps (and completed from step-level issues);
* citations of sources that were never retrieved are added as a CRITICAL issue — checked in
  code, whatever the model noticed;
* the verdict can only be made *stricter*: PASS requires the model's PASS, a score at or above
  CRITIC_PASS_THRESHOLD and no critical issue.
"""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.agents.synthesizer import cited_labels
from app.llm.prompts import CRITIC
from app.schemas.review import IssueSeverity, Review, ReviewInput, ReviewIssue, ReviewStatus


def apply_review_policy(review: Review, data: ReviewInput) -> Review:
    step_ids = {step.step_id for step in data.plan_steps}
    issues = list(review.issues)

    unknown = sorted(cited_labels(data.draft.full_text()) - set(data.available_sources))
    if unknown:
        issues.append(
            ReviewIssue(
                severity=IssueSeverity.CRITICAL,
                description=f"The draft cites sources that were never retrieved: {unknown}",
            )
        )

    rework = [step for step in review.rework_steps if step in step_ids]
    rework += [
        issue.step_id
        for issue in issues
        if issue.step_id in step_ids and issue.step_id not in rework
    ]
    critical = any(issue.severity is IssueSeverity.CRITICAL for issue in issues)
    passed = (
        review.status is ReviewStatus.PASS and review.score >= data.pass_threshold and not critical
    )
    return review.model_copy(
        update={
            "status": ReviewStatus.PASS if passed else ReviewStatus.FAIL,
            "issues": issues[:15],
            "rework_steps": list(dict.fromkeys(step for step in rework if step))[:10],
        }
    )


class CriticAgent(BaseAgent[ReviewInput, Review]):
    profile = AGENT_PROFILES[AgentName.CRITIC]
    prompt = CRITIC

    async def execute(self, ctx: AgentContext, data: ReviewInput, *, step_id: str | None) -> Review:
        review = await self._generate(ctx, data.model_dump(mode="json"), Review, step_id=step_id)
        return apply_review_policy(review, data)

    def summarize(self, value: Review) -> str:
        return f"{value.status.value} score={value.score} issues={len(value.issues)}"
