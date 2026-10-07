"""RAG agent: a fixed retrieval pipeline followed by one grounded generation.

1. hybrid retrieval in the tenant's knowledge base (injection-flagged chunks excluded);
2. if nothing relevant is found, answer "not in the documents" WITHOUT calling the LLM;
3. otherwise generate an answer whose citations are verified: every cited id must be one of the
   retrieved sources and every quote must be verbatim in that source — failures are sent back
   to the model for repair.

"Verbatim" compares the sequence of words: case, punctuation, quote marks, dashes and spacing
are ignored (typographic variants are not fabrications), elisions marked with "…" are allowed
when every fragment appears in order, and a cut word at either end of a fragment is accepted.
A paraphrase, a translation or words stitched from different places do not match.

Quotes shown to users are verbatim by construction: a near-verbatim quote (live models drop a
word or two without marking it) is replaced by the source sentence it reproduces, provided at
least 80 % of its words are in that one sentence and every number is identical. The model
points at the evidence; the code supplies its exact text.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable

from app.agents.base import AgentContext, AgentError, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.core.logging import log_event
from app.core.text import normalize_text, split_sentences, strip_accents, truncate
from app.llm.prompts import RAG
from app.observability.tracing import EventType
from app.schemas.agent import Citation, RagAnswer, WorkerInput
from app.schemas.knowledge import RetrievedChunk

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+")
_ELISION = re.compile(r"\.\.\.|…|\[\s*(?:…|\.\.\.)\s*\]")
MIN_WORDS_PER_FRAGMENT = 3  # when a quote has elisions: no stitching of isolated words
SNAP_MIN_WORDS = 3
SNAP_MIN_OVERLAP = 0.8


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text).lower())


def _matches_at(words: list[str], fragment: list[str], start: int) -> bool:
    last = len(fragment) - 1
    for offset, word in enumerate(fragment):
        candidate = words[start + offset]
        if last == 0:
            matched = word in candidate
        elif offset == 0:
            matched = candidate.endswith(word)
        elif offset == last:
            matched = candidate.startswith(word)
        else:
            matched = candidate == word
        if not matched:
            return False
    return True


def is_verbatim(quote: str, source: str) -> bool:
    """True when ``quote`` is an excerpt of ``source`` (see the module docstring)."""
    words = _words(source)
    fragments = [_words(part) for part in _ELISION.split(quote)]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return False
    if len(fragments) > 1 and min(map(len, fragments)) < MIN_WORDS_PER_FRAGMENT:
        return False
    position = 0
    for fragment in fragments:
        found = next(
            (
                start
                for start in range(position, len(words) - len(fragment) + 1)
                if _matches_at(words, fragment, start)
            ),
            None,
        )
        if found is None:
            return False
        position = found + len(fragment)
    return True


def _folded(text: str) -> list[str]:
    return [strip_accents(word) for word in _words(text)]


def resolve_quote(quote: str, source: str) -> str | None:
    """The verbatim evidence for ``quote`` in ``source``, or None when there is none.

    An exact excerpt is kept as written; a near-verbatim quote becomes the source sentence it
    reproduces (>= 80 % of its words in that sentence, same numbers: a changed figure is a
    fabrication, not a typo); a paraphrase, a translation or an invention resolves to None.
    """
    if is_verbatim(quote, source):
        return quote
    wanted = _folded(quote)
    if len(wanted) < SNAP_MIN_WORDS:
        return None
    numbers = {word for word in wanted if word.isdigit()}
    best: str | None = None
    best_ratio = 0.0
    for sentence in split_sentences(source):
        words = _folded(sentence)
        if not numbers <= set(words):
            continue
        available = Counter(words)
        matched = sum(min(count, available[word]) for word, count in Counter(wanted).items())
        if matched / len(wanted) > best_ratio:
            best, best_ratio = sentence, matched / len(wanted)
    return best if best_ratio >= SNAP_MIN_OVERLAP else None


def with_verbatim_quotes(answer: RagAnswer, sources: dict[str, RetrievedChunk]) -> RagAnswer:
    """Replace each (validated) quote by its verbatim evidence."""
    citations = []
    for citation in answer.citations:
        resolved = resolve_quote(citation.quote, sources[citation.source_id].content)
        if resolved is None:  # unreachable after grounded_citations; never show an unverified quote
            raise AgentError(f"unverifiable quote for {citation.source_id}")
        if resolved != citation.quote:
            log_event(
                logger,
                "citation_resolved",
                source_id=citation.source_id,
                quote=truncate(citation.quote, 120),
                resolved=truncate(resolved, 120),
            )
        citations.append(Citation(source_id=citation.source_id, quote=resolved))
    return answer.model_copy(update={"citations": citations})


def grounded_citations(sources: dict[str, RetrievedChunk]) -> Callable[[RagAnswer], None]:
    def check(answer: RagAnswer) -> None:
        problems = []
        for citation in answer.citations:
            chunk = sources.get(citation.source_id)
            if chunk is None:
                problems.append(f"{citation.source_id} is not one of the provided sources")
            elif resolve_quote(citation.quote, chunk.content) is None:
                problems.append(
                    f"the quote attributed to {citation.source_id} "
                    f'("{truncate(citation.quote, 60)}") is not verbatim: copy its words '
                    "exactly as written in the source, in the document's language — or remove "
                    "this citation if the source does not state it"
                )
        if not answer.citations and not answer.missing_information:
            problems.append("cite at least one source, or list the missing information")
        if problems:
            raise ValueError("; ".join(problems))

    return check


class RagAgent(BaseAgent[WorkerInput, RagAnswer]):
    profile = AGENT_PROFILES[AgentName.RAG]
    prompt = RAG

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    async def execute(
        self, ctx: AgentContext, data: WorkerInput, *, step_id: str | None
    ) -> RagAnswer:
        if ctx.retriever is None:
            raise AgentError("no knowledge retriever configured for this run")
        question = data.step.description
        result = await ctx.retriever.search(ctx.tenant_id, question, top_k=self._top_k)
        labelled = {ctx.journal.sources.register(chunk).label: chunk for chunk in result.chunks}
        await ctx.emit(
            EventType.RETRIEVAL,
            agent=self.profile.name.value,
            step_id=step_id,
            query=question,
            results=[
                {"source_id": label, "title": chunk.title, "score": chunk.score}
                for label, chunk in labelled.items()
            ],
            filtered_out=result.filtered_out,
            latency_ms=result.latency_ms,
        )
        if result.filtered_out:
            await ctx.emit(
                EventType.GUARDRAIL_TRIGGERED,
                agent=self.profile.name.value,
                step_id=step_id,
                guardrail="indirect_prompt_injection",
                target="retrieved_chunks",
                action="excluded",
                count=result.filtered_out,
            )
        if not labelled:
            return RagAnswer(
                summary="The knowledge base has no document relevant to this step.",
                confidence="low",
                answer="The internal documents do not contain information about this question.",
                missing_information=[question[:300]],
            )
        payload = {
            "question": question,
            "objective": data.objective,
            "feedback": data.feedback,
            "sources": [
                {
                    "source_id": label,
                    "title": chunk.title,
                    "content": chunk.content,
                    "flags": chunk.metadata.get("injection_signals", []),
                }
                for label, chunk in labelled.items()
            ],
        }
        answer = await self._generate(
            ctx, payload, RagAnswer, step_id=step_id, validate=grounded_citations(labelled)
        )
        return with_verbatim_quotes(answer, labelled)

    def summarize(self, value: RagAnswer) -> str:
        return value.summary[:240]
