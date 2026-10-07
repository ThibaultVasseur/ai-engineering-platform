"""Helpers shared by the offline handlers: read the agents' JSON context, detect intents."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from app.core.text import strip_accents
from app.llm.providers.offline import OfflineReply
from app.llm.types import LLMRequest, ToolCall, ToolResult

_CONTEXT = re.compile(r"<context>\s*(\{.*\})\s*</context>", re.DOTALL)
_CLAUSE_SPLIT = re.compile(r"[;\n]|\.\s|,\s(?:and|et|puis|then)\s|\s(?:and then|et ensuite|puis)\s")

INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "knowledge": (
        "document",
        "documentation",
        "docs",
        "knowledge",
        "connaissance",
        "politique",
        "policy",
        "policies",
        "guide",
        "specification",
        "spec",
        "interne",
        "internal",
        "procedure",
        "regle",
        "contrat",
        "manuel",
        "handbook",
        "faq",
    ),
    "data": (
        "donnee",
        "data",
        "statisti",
        "metrique",
        "metric",
        "kpi",
        "chiffre",
        "combien",
        "taux",
        "rate",
        "volume",
        "tendance",
        "trend",
        "how many",
        "nombre de",
        "repartition",
    ),
    "coding": (
        "implement",
        "code",
        "api",
        "endpoint",
        "architecture",
        "systeme",
        "system",
        "backend",
        "schema",
        "base de donnees",
        "database",
        "module",
        "service",
        "application",
        "saas",
        "developp",
        "develop",
        "build",
        "construi",
        "refactor",
        "migration",
        "integration",
    ),
    "research": (
        "recherche",
        "research",
        "analys",
        "compar",
        "etat de l'art",
        "option",
        "alternative",
        "benchmark",
        "precedent",
        "similaire",
        "similar",
        "investigat",
        "explore",
        "evalu",
        "best practice",
        "bonnes pratiques",
    ),
}
_INTENT_PATTERNS = {
    intent: re.compile(r"\b(" + "|".join(re.escape(word) for word in words) + ")")
    for intent, words in INTENT_KEYWORDS.items()
}


def read_context(request: LLMRequest) -> dict[str, Any]:
    """The JSON payload the agent placed in the ``<context>`` block of its first message."""
    for message in request.messages:
        if message.role == "user" and message.text:
            match = _CONTEXT.search(message.text)
            if match:
                payload: dict[str, Any] = json.loads(match.group(1))
                return payload
    raise ValueError("offline provider: no <context> block in the request")


def tool_results(request: LLMRequest) -> list[ToolResult]:
    return [result for message in request.messages for result in message.tool_results]


def previous_tool_calls(request: LLMRequest) -> list[ToolCall]:
    return [call for message in request.messages for call in message.tool_calls]


def results_by_tool(request: LLMRequest) -> dict[str, list[ToolResult]]:
    names = {call.id: call.name for call in previous_tool_calls(request)}
    grouped: dict[str, list[ToolResult]] = {}
    for result in tool_results(request):
        grouped.setdefault(names.get(result.tool_call_id, "unknown"), []).append(result)
    return grouped


def tool_exchanges(request: LLMRequest) -> list[tuple[ToolCall, ToolResult]]:
    """(call, result) pairs of the tool loop so far, in order."""
    calls = {call.id: call for call in previous_tool_calls(request)}
    return [
        (calls[result.tool_call_id], result)
        for result in tool_results(request)
        if result.tool_call_id in calls
    ]


def parse_result(result: ToolResult) -> dict[str, Any] | None:
    if result.is_error:
        return None
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def detect_intents(text: str) -> set[str]:
    folded = strip_accents(text.lower())
    return {intent for intent, pattern in _INTENT_PATTERNS.items() if pattern.search(folded)}


def clauses(text: str, limit: int = 6) -> list[str]:
    parts = [part.strip(" .,-") for part in _CLAUSE_SPLIT.split(text)]
    return [part[0].upper() + part[1:] for part in parts if len(part) >= 8][:limit]


def is_french(text: str) -> bool:
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())
    french = sum(
        word in {"le", "la", "les", "des", "une", "et", "pour", "est", "du"} for word in words
    )
    english = sum(word in {"the", "and", "for", "is", "of", "to", "with", "a"} for word in words)
    return french > english


def reply_json(model: BaseModel) -> OfflineReply:
    return OfflineReply(text=model.model_dump_json())
