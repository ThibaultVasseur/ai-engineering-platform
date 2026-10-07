"""PostgreSQL knowledge store: pgvector cosine search + full-text search, tenant-scoped.

Every method opens its own tenant-scoped transaction (``Database.session``), so row-level
security applies on top of the explicit ``tenant_id`` filters.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Select, func, select, text
from sqlalchemy.dialects.postgresql import insert

from app.db.database import Database
from app.db.models import DocumentChunkModel, DocumentModel
from app.rag.ingestion import PreparedDocument
from app.rag.store import StoredDocument
from app.schemas.knowledge import RetrievedChunk, SearchFilters


def _stored(model: DocumentModel) -> StoredDocument:
    return StoredDocument(
        id=str(model.id),
        title=model.title,
        source=model.source,
        metadata=model.metadata_,
        chunk_count=model.chunk_count,
        created_at=model.created_at,
    )


def _apply_filters(stmt: Select[Any], filters: SearchFilters | None) -> Select[Any]:
    if filters is None:
        return stmt
    if filters.category:
        stmt = stmt.where(DocumentChunkModel.metadata_.contains({"category": filters.category}))
    if filters.document_ids:
        stmt = stmt.where(DocumentChunkModel.document_id.in_(filters.document_ids))
    return stmt


class PostgresKnowledgeStore:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def add_document(
        self, tenant_id: str, document: PreparedDocument
    ) -> tuple[StoredDocument, bool]:
        async with self._db.session(tenant_id) as session:
            document_id = uuid.uuid4()
            inserted = await session.scalar(
                insert(DocumentModel)
                .values(
                    id=document_id,
                    tenant_id=tenant_id,
                    title=document.title,
                    source=document.source,
                    content=document.content,
                    content_sha256=document.sha256,
                    metadata_=document.metadata,
                    chunk_count=len(document.chunks),
                )
                .on_conflict_do_nothing(constraint="uq_documents_tenant_sha")
                .returning(DocumentModel.id)
            )
            if inserted is None:  # identical content already ingested for this tenant
                existing = await session.scalar(
                    select(DocumentModel).where(
                        DocumentModel.tenant_id == tenant_id,
                        DocumentModel.content_sha256 == document.sha256,
                    )
                )
                assert existing is not None  # noqa: S101 - guaranteed by the conflict
                return _stored(existing), False
            if document.chunks:
                # ORM bulk insert: keys are ORM *attribute* names (metadata_, not metadata).
                await session.execute(
                    insert(DocumentChunkModel),
                    [
                        {
                            "id": uuid.uuid4(),
                            "document_id": document_id,
                            "tenant_id": tenant_id,
                            "chunk_index": chunk.index,
                            "content": chunk.content,
                            "embedding": chunk.embedding,
                            "metadata_": chunk.metadata,
                            "token_count": chunk.token_count,
                        }
                        for chunk in document.chunks
                    ],
                )
            created = await session.get_one(DocumentModel, document_id)
            return _stored(created), True

    async def list_documents(
        self, tenant_id: str, *, limit: int, offset: int
    ) -> list[StoredDocument]:
        async with self._db.session(tenant_id) as session:
            rows = await session.scalars(
                select(DocumentModel)
                .where(DocumentModel.tenant_id == tenant_id)
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [_stored(row) for row in rows]

    async def vector_search(
        self,
        tenant_id: str,
        embedding: list[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        distance = DocumentChunkModel.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(DocumentChunkModel, DocumentModel.title, DocumentModel.source, distance)
            .join(DocumentModel, DocumentModel.id == DocumentChunkModel.document_id)
            .where(DocumentChunkModel.tenant_id == tenant_id)
            .order_by(distance)
            .limit(limit)
        )
        async with self._db.session(tenant_id) as session:
            # With filters (tenant, RLS, metadata) an HNSW scan may return fewer rows than asked;
            # iterative scans (pgvector >= 0.8) keep scanning until enough rows qualify.
            await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            rows = (await session.execute(_apply_filters(stmt, filters))).all()
        return [
            self._to_chunk(row[0], row[1], row[2], vector_score=round(1 - float(row[3]), 6))
            for row in rows
        ]

    async def text_search(
        self,
        tenant_id: str,
        terms: list[str],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        # Terms come from app.core.text.keywords(): letters and digits only, so the OR query
        # below cannot contain tsquery operators injected by the user.
        safe_terms = [term for term in terms if term.isalnum()][:12]
        if not safe_terms:
            return []
        query = func.to_tsquery("simple", " | ".join(safe_terms))
        rank = func.ts_rank_cd(DocumentChunkModel.content_tsv, query).label("rank")
        stmt = (
            select(DocumentChunkModel, DocumentModel.title, DocumentModel.source, rank)
            .join(DocumentModel, DocumentModel.id == DocumentChunkModel.document_id)
            .where(
                DocumentChunkModel.tenant_id == tenant_id,
                DocumentChunkModel.content_tsv.op("@@")(query),
            )
            .order_by(rank.desc())
            .limit(limit)
        )
        async with self._db.session(tenant_id) as session:
            rows = (await session.execute(_apply_filters(stmt, filters))).all()
        return [
            self._to_chunk(row[0], row[1], row[2], text_rank=round(float(row[3]), 6))
            for row in rows
        ]

    @staticmethod
    def _to_chunk(
        chunk: DocumentChunkModel, title: str, source: str | None, **scores: float
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            title=title,
            source=source,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            metadata=chunk.metadata_,
            **scores,
        )
