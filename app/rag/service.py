"""Knowledge base use cases (API side): ingest, list, search."""

from __future__ import annotations

import uuid

from app.core.exceptions import InvalidRequestError
from app.core.security import Tenant
from app.rag.ingestion import DocumentIngestor, IngestionError
from app.rag.retrieval import HybridRetriever
from app.rag.store import KnowledgeStore, StoredDocument
from app.schemas.common import Page
from app.schemas.knowledge import (
    DocumentCreate,
    DocumentRead,
    IngestionReport,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


def to_read(document: StoredDocument) -> DocumentRead:
    return DocumentRead(
        id=uuid.UUID(document.id),
        title=document.title,
        source=document.source,
        metadata_=document.metadata,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )


class KnowledgeService:
    def __init__(
        self, store: KnowledgeStore, ingestor: DocumentIngestor, retriever: HybridRetriever
    ) -> None:
        self._store = store
        self._ingestor = ingestor
        self._retriever = retriever

    async def ingest(self, tenant: Tenant, payload: DocumentCreate) -> IngestionReport:
        try:
            prepared = await self._ingestor.prepare(
                title=payload.title,
                content=payload.content,
                source=payload.source,
                metadata=dict(payload.metadata),
            )
        except IngestionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        stored, created = await self._store.add_document(tenant.id, prepared)
        return IngestionReport(
            document=to_read(stored),
            created=created,
            flagged_chunks=prepared.flagged_chunks,
            hidden_characters_removed=prepared.hidden_characters_removed,
        )

    async def list(self, tenant: Tenant, *, limit: int, offset: int) -> Page[DocumentRead]:
        documents = await self._store.list_documents(tenant.id, limit=limit, offset=offset)
        return Page[DocumentRead](
            items=[to_read(document) for document in documents], limit=limit, offset=offset
        )

    async def search(self, tenant: Tenant, request: SearchRequest) -> SearchResponse:
        result = await self._retriever.search(
            tenant.id, request.query, top_k=request.top_k, filters=request.filters
        )
        return SearchResponse(
            query=request.query,
            hits=[
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    source=chunk.source,
                    score=chunk.score,
                    vector_score=chunk.vector_score,
                    text_rank=chunk.text_rank,
                    content=chunk.content,
                    injection_risk=str(chunk.metadata.get("injection_risk", "none")),
                )
                for chunk in result.chunks
            ],
            filtered_out=result.filtered_out,
            latency_ms=result.latency_ms,
        )
