"""Heuristic prompt-injection detection (English + French).

This is ONE layer, deliberately simple and explainable — paraphrases, other languages or
encodings will get through. The platform does not rely on it alone: tools are least-privilege
and permission-checked, retrieved documents are wrapped as untrusted data, outputs are
schema-validated and reviewed by the critic (see docs/security.md).

Signals are scored; a score of 3 or more is HIGH risk (blocked before any LLM call when it is
the user request, excluded from the context when it is a retrieved chunk).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.text import count_invisible_characters

_VERBS_EXFIL = (
    r"(reveal|leak|dump|exfiltrate|send|post|print out|show me|révèle[rz]?|divulgue[rz]?|"
    r"envoie[rz]?|transmets?|transmettre|donne[rz]?[- ]moi)"
)
_SECRETS = (
    r"(api[_ -]?keys?|secret keys?|secrets|passwords?|credentials|access tokens?|"
    r"clés? (?:d'|d’)?api|mots? de passe|identifiants)"
)

# (signal name, pattern, score)
_PATTERNS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all|any|your|system)\b[^.\n]{0,30}"
            r"\b(instructions?|prompts?|rules?|directives?|guidelines?)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "instruction_override",
        re.compile(
            r"\b(ignore[rz]?|oublie[rz]?|ne tiens? pas compte (?:de|des)|contourne[rz]?)\b"
            r"[^.\n]{0,40}\b(instructions?|consignes?|règles?|directives?|prompt)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(reveal|print|show|display|repeat|leak|output|révèle[rz]?|affiche[rz]?|"
            r"répète[rz]?|donne[rz]?[- ]moi)\b[^.\n]{0,40}"
            r"(system prompt|prompt syst[eè]me|hidden instructions|initial instructions|"
            r"your instructions|tes instructions|vos instructions|instructions cachées)",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(call|use|invoke|run|execute|trigger|appelle[rz]?|utilise[rz]?|exécute[rz]?|"
            r"lance[rz]?)\b[^.\n]{0,30}"
            r"\b(database_write|shell|bash|os\.system|subprocess|rm -rf|drop table|"
            r"delete from|truncate table)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "role_hijack",
        re.compile(
            r"(you are now|from now on,? you (?:are|will)|act as (?:an? )?(?:unrestricted|"
            r"jailbroken|unfiltered)|developer mode|tu es (?:maintenant|désormais)|"
            r"à partir de maintenant,? tu)",
            re.IGNORECASE,
        ),
        2,
    ),
    (
        "fake_role_tag",
        re.compile(
            r"(<\s*/?\s*(system|assistant|instructions?)\s*>|\[/?(?:system|inst)\]|"
            r"^\s*#{2,}\s*system\b)",
            re.IGNORECASE | re.MULTILINE,
        ),
        2,
    ),
    (
        "secret_exfiltration",
        re.compile(
            rf"\b{_VERBS_EXFIL}\b[^.\n]{{0,40}}\b{_SECRETS}|\b{_SECRETS}\b[^.\n]{{0,40}}"
            rf"\b{_VERBS_EXFIL}\b",
            re.IGNORECASE,
        ),
        2,
    ),
)

HIGH_RISK_SCORE = 3


class InjectionRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class InjectionScan:
    risk: InjectionRisk
    score: int
    signals: list[str] = field(default_factory=list)

    @property
    def is_high(self) -> bool:
        return self.risk is InjectionRisk.HIGH


def scan_for_injection(text: str) -> InjectionScan:
    """Score ``text`` (raw, before normalisation — hidden characters are a signal)."""
    signals: list[str] = []
    score = 0
    for name, pattern, weight in _PATTERNS:
        if name not in signals and pattern.search(text):
            signals.append(name)
            score += weight
    if count_invisible_characters(text):
        signals.append("hidden_characters")
        score += 1
    if score >= HIGH_RISK_SCORE:
        risk = InjectionRisk.HIGH
    elif score > 0:
        risk = InjectionRisk.LOW
    else:
        risk = InjectionRisk.NONE
    return InjectionScan(risk=risk, score=score, signals=signals)
