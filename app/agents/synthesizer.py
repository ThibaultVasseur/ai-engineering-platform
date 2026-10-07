"""Synthesizer: merges the step results into the final draft reviewed by the critic."""

from __future__ import annotations

import re
from collections.abc import Callable

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.llm.prompts import SYNTHESIZER
from app.schemas.review import FinalDraft, SynthesisInput

CITATION = re.compile(r"\[(S\d{1,3})\]")


def cited_labels(text: str) -> set[str]:
    return set(CITATION.findall(text))


def citations_within(known: set[str]) -> Callable[[FinalDraft], None]:
    def check(draft: FinalDraft) -> None:
        unknown = sorted(cited_labels(draft.full_text()) - known)
        if unknown:
            raise ValueError(
                f"the draft cites unknown sources {unknown}; available: {sorted(known) or 'none'}"
            )

    return check


class SynthesizerAgent(BaseAgent[SynthesisInput, FinalDraft]):
    profile = AGENT_PROFILES[AgentName.SYNTHESIZER]
    prompt = SYNTHESIZER

    async def execute(
        self, ctx: AgentContext, data: SynthesisInput, *, step_id: str | None
    ) -> FinalDraft:
        known = {source.label for source in data.sources}
        return await self._generate(
            ctx,
            data.model_dump(mode="json"),
            FinalDraft,
            step_id=step_id,
            validate=citations_within(known),
        )

    def summarize(self, value: FinalDraft) -> str:
        return f"{value.title} ({len(value.sections)} section(s))"[:240]
