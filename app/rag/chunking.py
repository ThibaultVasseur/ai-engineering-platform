"""Structure-aware chunking.

Text is cut into *units* — paragraphs; paragraphs longer than the chunk size are split into
sentences; sentences longer than the chunk size are hard-wrapped. Units are packed greedily into
chunks of at most ``size`` characters. Consecutive chunks overlap by whole trailing units (up to
``overlap`` characters) so that an idea straddling a boundary is retrievable from both sides.
Offsets into the source text are kept for every chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    start: int
    end: int


def _units(text: str, size: int) -> list[tuple[int, int]]:
    units: list[tuple[int, int]] = []
    for paragraph in _PARAGRAPH.finditer(text):
        if paragraph.end() - paragraph.start() <= size:
            units.append((paragraph.start(), paragraph.end()))
            continue
        for sentence in _SENTENCE.finditer(text, paragraph.start(), paragraph.end()):
            start, end = sentence.start(), sentence.end()
            while text[start:end].strip() and end - start > size:
                units.append((start, start + size))
                start += size
            if text[start:end].strip():
                units.append((start, end))
    return units


def chunk_text(text: str, *, size: int = 900, overlap: int = 150) -> list[TextChunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    units = _units(text, size)
    chunks: list[TextChunk] = []
    current: list[tuple[int, int]] = []

    def emit() -> None:
        start, end = current[0][0], current[-1][1]
        chunks.append(TextChunk(len(chunks), text[start:end].strip(), start, end))

    for unit in units:
        if current and unit[1] - current[0][0] > size:
            emit()
            carried: list[tuple[int, int]] = []
            for previous in reversed(current):
                if (current[-1][1] - previous[0]) > overlap or unit[1] - previous[0] > size:
                    break
                carried.insert(0, previous)
            current = carried
        current.append(unit)
    if current:
        emit()
    return [chunk for chunk in chunks if chunk.content]
