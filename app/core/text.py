"""Text normalisation shared by user requests and ingested documents.

Normalising *before* anything else matters for security: hidden characters (zero-width
spaces, bidirectional overrides) and full-width look-alikes are classic ways to smuggle
instructions past filters and human reviewers.
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE_CHARS = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_EXCESS_SPACES = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    """NFKC-normalise, drop invisible/control characters and tidy whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _EXCESS_SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("\n", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def count_invisible_characters(text: str) -> int:
    """Number of characters ``normalize_text`` would silently remove (a tampering signal)."""
    return len(_INVISIBLE_CHARS.findall(text)) + len(_CONTROL_CHARS.findall(text))


def truncate(text: str, limit: int, marker: str = "… [truncated]") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


# --------------------------------------------------------------------------------------
# Lightweight lexical analysis (keyword search, hashing embeddings, offline heuristics)
# --------------------------------------------------------------------------------------
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")

_STOPWORDS_EN = (
    "a an and are as at be by for from has have how i in is it its of on or that the this to "
    "was what when where which who will with you your we our can should would could into about"
)
_STOPWORDS_FR = (
    "le la les un une des du de d l et ou en au aux pour par sur dans avec sans est sont qui "
    "que quoi dont ce cet cette ces se sa son ses leur leurs nous vous ils elles il elle on ne "
    "pas plus mais si comme aussi tout tous toute toutes être avoir fait faire peut doit me te "
    "moi toi lui y à ça cela ceci afin entre chez sous très quel quelle quels quelles comment "
    "pourquoi quand où lequel laquelle lesquels lesquelles"
)
STOPWORDS = frozenset(_STOPWORDS_EN.split()) | frozenset(_STOPWORDS_FR.split())


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (letters and digits, any script)."""
    return _WORD.findall(text.lower())


def keywords(text: str, limit: int | None = None) -> list[str]:
    """Distinct non-stopword tokens of 3+ characters, in order of first appearance."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit():
            seen.setdefault(token, None)
    words = list(seen)
    return words if limit is None else words[:limit]


def normalized_keywords(text: str) -> set[str]:
    """Accent-insensitive keyword set with a naive plural folding (devis/devises → devi...)."""
    return {_fold(strip_accents(word)) for word in keywords(text)}


def keyword_coverage(reference: str, candidate: str) -> float:
    """Share of the reference's keywords that also appear in the candidate (0..1)."""
    wanted = normalized_keywords(reference)
    if not wanted:
        return 1.0
    return len(wanted & normalized_keywords(candidate)) / len(wanted)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


def _fold(word: str) -> str:
    if len(word) > 4 and word.endswith(("s", "x")):
        return word[:-1]
    return word
