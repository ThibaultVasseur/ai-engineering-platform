"""Knowledge store port and its in-memory implementation.

The retriever only needs two primitives — a vector search and a keyword search — plus the write
side used by ingestion. PostgreSQL implements them with pgvector and ``tsvector``
(``app.db.repositories.documents``); the in-memory store implements the same contract for the
CLI demo, the evaluation suite and unit tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from app.core.text import tokenize
from app.llm.embeddings import cosine_similarity
from app.rag.ingestion import PreparedDocument
from app.schemas.knowledge import RetrievedChunk, SearchFilters


@dataclass(frozen=True)
class StoredDocument:
    id: str
    title: str
    source: str | None
    metadata: dict[str, Any]
    chunk_count: int
    created_at: datetime


class KnowledgeStore(Protocol):
    async def add_document(
        self, tenant_id: str, document: PreparedDocument
    ) -> tuple[StoredDocument, bool]:
        """Store the document; returns (document, created). Idempotent on content hash."""
        ...

    async def list_documents(
        self, tenant_id: str, *, limit: int, offset: int
    ) -> list[StoredDocument]: ...

    async def vector_search(
        self,
        tenant_id: str,
        embedding: list[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]: ...

    async def text_search(
        self,
        tenant_id: str,
        terms: list[str],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]: ...


@dataclass
class _MemoryChunk:
    id: str
    document: StoredDocument
    tenant_id: str
    index: int
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    tokens: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = tokenize(self.content)

    def to_retrieved(self, **scores: float | None) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.id,
            document_id=self.document.id,
            title=self.document.title,
            source=self.document.source,
            chunk_index=self.index,
            content=self.content,
            metadata=self.metadata,
            **scores,  # type: ignore[arg-type]
        )


def _matches(chunk: _MemoryChunk, filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    if filters.category and chunk.metadata.get("category") != filters.category:
        return False
    return not (
        filters.document_ids and chunk.document.id not in {str(d) for d in filters.document_ids}
    )


class InMemoryKnowledgeStore:
    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], StoredDocument] = {}  # (tenant, sha256)
        self._chunks: list[_MemoryChunk] = []

    async def add_document(
        self, tenant_id: str, document: PreparedDocument
    ) -> tuple[StoredDocument, bool]:
        key = (tenant_id, document.sha256)
        if key in self._documents:
            return self._documents[key], False
        stored = StoredDocument(
            id=str(uuid.uuid4()),
            title=document.title,
            source=document.source,
            metadata=document.metadata,
            chunk_count=len(document.chunks),
            created_at=datetime.now(UTC),
        )
        self._documents[key] = stored
        self._chunks.extend(
            _MemoryChunk(
                id=str(uuid.uuid4()),
                document=stored,
                tenant_id=tenant_id,
                index=chunk.index,
                content=chunk.content,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
            )
            for chunk in document.chunks
        )
        return stored, True

    async def list_documents(
        self, tenant_id: str, *, limit: int, offset: int
    ) -> list[StoredDocument]:
        documents = [doc for (tenant, _), doc in self._documents.items() if tenant == tenant_id]
        documents.sort(key=lambda doc: doc.created_at, reverse=True)
        return documents[offset : offset + limit]

    async def vector_search(
        self,
        tenant_id: str,
        embedding: list[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        scored = [
            (cosine_similarity(embedding, chunk.embedding), chunk)
            for chunk in self._chunks
            if chunk.tenant_id == tenant_id and _matches(chunk, filters)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk.to_retrieved(vector_score=score) for score, chunk in scored[:limit]]

    async def text_search(
        self,
        tenant_id: str,
        terms: list[str],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        scored = []
        for chunk in self._chunks:
            if chunk.tenant_id != tenant_id or not _matches(chunk, filters):
                continue
            hits = sum(chunk.tokens.count(term) for term in terms)
            if hits:
                scored.append((hits / (1 + len(chunk.tokens) / 100), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk.to_retrieved(text_rank=rank) for rank, chunk in scored[:limit]]
