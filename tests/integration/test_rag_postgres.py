"""RAG on the real stack: PostgreSQL + pgvector + full-text search + RLS."""

import pytest
from sqlalchemy import text

from app.core.security import Tenant
from app.db.database import Database
from app.db.repositories.documents import PostgresKnowledgeStore
from app.llm.embeddings import HashingEmbedder
from app.rag.demo import demo_documents
from app.rag.ingestion import DocumentIngestor
from app.rag.reranking import LexicalReranker
from app.rag.retrieval import HybridRetriever
from app.rag.service import KnowledgeService
from app.schemas.knowledge import SearchFilters, SearchRequest

pytestmark = pytest.mark.integration

ACME = Tenant(id="acme")


def knowledge(database: Database) -> tuple[KnowledgeService, HybridRetriever]:
    embedder = HashingEmbedder()
    store = PostgresKnowledgeStore(database)
    retriever = HybridRetriever(store, embedder, reranker=LexicalReranker())
    return KnowledgeService(store, DocumentIngestor(embedder), retriever), retriever


async def seed(service: KnowledgeService, tenant: Tenant = ACME) -> None:
    for document in demo_documents():
        await service.ingest(tenant, document)


async def test_ingestion_stores_chunks_with_vectors_and_tsvector(database: Database) -> None:
    service, _ = knowledge(database)
    await seed(service)
    async with database.session("acme") as session:
        row = (
            await session.execute(
                text(
                    "SELECT count(*), bool_and(content_tsv IS NOT NULL), "
                    "max(vector_dims(embedding)) FROM document_chunks"
                )
            )
        ).one()
    assert row[0] >= 8
    assert row[1] is True
    assert row[2] == 512


async def test_chunk_and_document_metadata_are_persisted(database: Database) -> None:
    """Regression: metadata (category, injection risk) must reach the JSONB columns."""
    service, _ = knowledge(database)
    await seed(service)
    async with database.session("acme") as session:
        rows = await session.execute(
            text(
                "SELECT metadata->>'category', max(metadata->>'injection_risk') "
                "FROM document_chunks GROUP BY 1"
            )
        )
        risks = {category: risk for category, risk in rows.all()}  # noqa: C416
        categories = set(
            (await session.scalars(text("SELECT metadata->>'category' FROM documents"))).all()
        )
    assert risks == {"policy": "none", "product": "none", "technical": "none", "external": "high"}
    assert categories == {"policy", "product", "technical", "external"}


async def test_hybrid_search_combines_vector_and_full_text(database: Database) -> None:
    service, _ = knowledge(database)
    await seed(service)
    response = await service.search(
        ACME, SearchRequest(query="durée de validité d'un devis", top_k=3)
    )
    top = response.hits[0]
    assert "30 jours" in top.content
    assert top.vector_score is not None
    assert top.text_rank is not None
    assert response.filtered_out == 1  # the poisoned vendor document is excluded


async def test_reingesting_identical_content_is_idempotent(database: Database) -> None:
    service, _ = knowledge(database)
    first = await service.ingest(ACME, demo_documents()[0])
    second = await service.ingest(ACME, demo_documents()[0])
    assert first.created
    assert not second.created
    assert first.document.id == second.document.id


async def test_category_filter_uses_jsonb_containment(database: Database) -> None:
    service, retriever = knowledge(database)
    await seed(service)
    result = await retriever.search(
        "acme", "devis", top_k=5, filters=SearchFilters(category="technical")
    )
    assert result.chunks
    assert {c.metadata["category"] for c in result.chunks} == {"technical"}


async def test_knowledge_is_isolated_between_tenants(database: Database) -> None:
    service, retriever = knowledge(database)
    await seed(service)
    result = await retriever.search("globex", "durée de validité d'un devis", top_k=5)
    assert result.chunks == []
    listed = await service.list(Tenant(id="globex"), limit=10, offset=0)
    assert listed.items == []
