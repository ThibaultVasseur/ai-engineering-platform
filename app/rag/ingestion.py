"""Ingestion pipeline: raw text -> cleaned text -> chunks -> embeddings -> metadata.

    DOCUMENT -> EXTRACTION -> CLEANING -> CHUNKING -> INJECTION SCAN -> EMBEDDING -> STORE

* Cleaning removes invisible/bidirectional characters (counted and reported: they are a classic
  way to hide instructions from human reviewers).
* Every chunk is scanned for prompt-injection patterns; the risk is stored in its metadata and
  the retriever excludes HIGH-risk chunks (indirect prompt injection defence).
* Each chunk is embedded together with its document title (a "contextual header"), which helps
  short chunks that do not repeat what they are about.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from app.core.guardrails import InjectionRisk, scan_for_injection
from app.core.text import count_invisible_characters, normalize_text
from app.llm.embeddings import EmbeddingProvider
from app.rag.chunking import chunk_text

MIN_DOCUMENT_CHARS = 20


class IngestionError(ValueError):
    """The document cannot be ingested (e.g. empty after cleaning)."""


@dataclass(frozen=True)
class PreparedChunk:
    index: int
    content: str
    embedding: list[float]
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedDocument:
    title: str
    source: str | None
    content: str
    sha256: str
    metadata: dict[str, Any]
    chunks: list[PreparedChunk] = field(default_factory=list)
    hidden_characters_removed: int = 0

    @property
    def flagged_chunks(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.metadata["injection_risk"] != "none")


class DocumentIngestor:
    def __init__(
        self, embedder: EmbeddingProvider, *, chunk_size: int = 900, chunk_overlap: int = 150
    ) -> None:
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def embedding_model(self) -> str:
        return self._embedder.model

    async def prepare(
        self,
        *,
        title: str,
        content: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreparedDocument:
        hidden = count_invisible_characters(content)
        cleaned = normalize_text(content)
        if len(cleaned) < MIN_DOCUMENT_CHARS:
            raise IngestionError("the document is empty or too short after cleaning")
        document_metadata = dict(metadata or {})
        document_metadata["embedding_model"] = self._embedder.model
        if hidden:
            document_metadata["hidden_characters_removed"] = hidden

        pieces = chunk_text(cleaned, size=self._chunk_size, overlap=self._chunk_overlap)
        vectors = await self._embedder.embed([f"{title}\n{piece.content}" for piece in pieces])
        chunks = []
        for piece, vector in zip(pieces, vectors, strict=True):
            scan = scan_for_injection(piece.content)
            chunk_metadata = {
                key: value for key, value in document_metadata.items() if key == "category"
            }
            chunk_metadata.update(
                {
                    "injection_risk": scan.risk.value,
                    "injection_signals": scan.signals,
                    "char_start": piece.start,
                    "char_end": piece.end,
                }
            )
            if hidden and scan.risk is InjectionRisk.NONE:
                chunk_metadata["injection_risk"] = InjectionRisk.LOW.value
                chunk_metadata["injection_signals"] = ["hidden_characters_in_document"]
            chunks.append(
                PreparedChunk(
                    index=piece.index,
                    content=piece.content,
                    embedding=vector,
                    token_count=max(1, len(piece.content) // 4),
                    metadata=chunk_metadata,
                )
            )
        return PreparedDocument(
            title=title,
            source=source,
            content=cleaned,
            sha256=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
            metadata=document_metadata,
            chunks=chunks,
            hidden_characters_removed=hidden,
        )
