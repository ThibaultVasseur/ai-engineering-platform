"""Offline handler for the RAG agent: extractive answer built from the best-matching sentences.

A sentence qualifies when, together with its context (the section heading above it and the
document title), it covers at least two of the question's keywords (one for one- or two-word
questions); qualifying sentences are ranked by coverage. Nothing qualifies -> the answer says
the documents do not contain the information (no grounding on noise).
"""

from __future__ import annotations

from app.core.text import normalized_keywords, split_sentences, truncate
from app.llm.providers.offline import OfflineReply
from app.llm.providers.offline.common import read_context, reply_json
from app.llm.types import LLMRequest
from app.schemas.agent import Citation, RagAnswer

MAX_SENTENCES = 3
MIN_MATCHED_TERMS = 2
HEADING_MAX_CHARS = 80


def _is_heading(sentence: str) -> bool:
    return len(sentence) < HEADING_MAX_CHARS and not sentence.endswith((".", "!", "?", ";", ":"))


def _not_found(question: str) -> OfflineReply:
    return reply_json(
        RagAnswer(
            summary="The provided documents do not answer the question.",
            confidence="low",
            answer="The provided documents do not contain the requested information.",
            missing_information=[truncate(question, 300)],
        )
    )


def handle_rag(request: LLMRequest) -> OfflineReply:
    context = read_context(request)
    question: str = context["question"]
    wanted = normalized_keywords(question)
    if not wanted:
        return _not_found(question)

    needed = 1 if len(wanted) <= 2 else MIN_MATCHED_TERMS
    candidates: list[tuple[int, int, str, str]] = []
    for source in context["sources"]:
        title_terms = normalized_keywords(source["title"]) & wanted
        heading_terms: set[str] = set()
        for sentence in split_sentences(source["content"]):
            terms = normalized_keywords(sentence) & wanted
            if _is_heading(sentence):
                heading_terms = terms
                continue
            covered = len(terms | heading_terms | title_terms)
            if terms and len(sentence) >= 20 and covered >= needed:
                candidates.append((covered, -len(candidates), source["source_id"], sentence))
    candidates.sort(reverse=True)

    picked: list[tuple[str, str]] = []
    for _, _, source_id, sentence in candidates:
        if len(picked) == MAX_SENTENCES:
            break
        if all(sentence != chosen for _, chosen in picked):
            picked.append((source_id, sentence))
    if not picked:
        return _not_found(question)

    text = " ".join(f"{sentence.rstrip('. ')} [{source_id}]." for source_id, sentence in picked)
    for feedback in context.get("feedback", [])[:2]:
        text += f" Regarding the review feedback ({truncate(feedback, 150)}), see [{picked[0][0]}]."
    answer = RagAnswer(
        summary=truncate(f"From {len({s for s, _ in picked})} internal source(s): {text}", 600),
        confidence="medium",
        answer=truncate(text, 3000),
        citations=[
            Citation(source_id=source_id, quote=sentence[:280]) for source_id, sentence in picked
        ],
    )
    return reply_json(answer)
