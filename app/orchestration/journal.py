"""Per-run journal shared by the agents of a run.

* ``SourceRegistry`` numbers retrieved chunks once per run (``S1``, ``S2``...): every agent and
  the final answer cite the same label for the same chunk.
* Tool calls are recorded as they happen; graph nodes copy the new ones into the state, which
  remains the single source of truth once the run is over.
"""

from __future__ import annotations

from app.core.text import truncate
from app.schemas.knowledge import RetrievedChunk, SourceRef
from app.schemas.tool import ToolCallRecord


class SourceRegistry:
    def __init__(self) -> None:
        self._by_chunk: dict[str, SourceRef] = {}

    def register(self, chunk: RetrievedChunk) -> SourceRef:
        existing = self._by_chunk.get(chunk.chunk_id)
        if existing is not None:
            return existing
        reference = SourceRef(
            label=f"S{len(self._by_chunk) + 1}",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            source=chunk.source,
            score=round(chunk.score, 4),
            excerpt=truncate(chunk.content, 300),
        )
        self._by_chunk[chunk.chunk_id] = reference
        return reference

    def all(self) -> list[SourceRef]:
        return list(self._by_chunk.values())

    def labels(self) -> set[str]:
        return {reference.label for reference in self._by_chunk.values()}


class RunJournal:
    def __init__(self) -> None:
        self.sources = SourceRegistry()
        self._tool_calls: list[ToolCallRecord] = []
        self._cursor = 0

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls.append(record)

    def take_new_tool_calls(self) -> list[ToolCallRecord]:
        new = self._tool_calls[self._cursor :]
        self._cursor = len(self._tool_calls)
        return new

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return list(self._tool_calls)
