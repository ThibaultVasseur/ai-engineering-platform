import pytest

from app.llm.embeddings import HashingEmbedder
from app.rag.demo import demo_documents, ingest_demo_documents
from app.rag.ingestion import DocumentIngestor, IngestionError
from app.rag.reranking import LexicalReranker
from app.rag.retrieval import HybridRetriever, rrf_fuse
from app.rag.store import InMemoryKnowledgeStore
from app.schemas.knowledge import SearchFilters
from tests.fakes import chunk


@pytest.fixture
async def retriever() -> HybridRetriever:
    embedder = HashingEmbedder()
    store = InMemoryKnowledgeStore()
    await ingest_demo_documents(store, DocumentIngestor(embedder), "default")
    return HybridRetriever(store, embedder, reranker=LexicalReranker())


def test_rrf_rewards_agreement_between_rankings() -> None:
    a, b, c = chunk(1, "alpha"), chunk(2, "beta"), chunk(3, "gamma")
    fused = rrf_fuse([[a, b], [b, c]])
    assert [item.chunk_id for item in fused] == ["chunk-2", "chunk-1", "chunk-3"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)


@pytest.mark.parametrize(
    ("query", "category"),
    [
        ("Quelle est la durée de validité d'un devis ?", "policy"),
        ("isolation multi-tenant RLS", "technical"),
        ("indicateurs de succès et taux de conversion des devis", "product"),
    ],
)
async def test_relevant_document_ranks_first(
    retriever: HybridRetriever, query: str, category: str
) -> None:
    result = await retriever.search("default", query, top_k=3)
    assert result.chunks[0].metadata["category"] == category


async def test_injected_document_is_excluded_and_counted(retriever: HybridRetriever) -> None:
    result = await retriever.search(
        "default", "bibliothèque PDF du prestataire et ses tarifs", top_k=5
    )
    assert result.filtered_out == 1
    assert all(chunk.metadata["category"] != "external" for chunk in result.chunks)


async def test_unrelated_queries_return_nothing(retriever: HybridRetriever) -> None:
    result = await retriever.search("default", "recette de cuisine au chocolat", top_k=5)
    assert result.chunks == []


async def test_metadata_filter_restricts_results(retriever: HybridRetriever) -> None:
    result = await retriever.search(
        "default", "devis", top_k=5, filters=SearchFilters(category="product")
    )
    assert result.chunks
    assert {chunk.metadata["category"] for chunk in result.chunks} == {"product"}


async def test_other_tenants_see_nothing(retriever: HybridRetriever) -> None:
    result = await retriever.search("another-tenant", "durée de validité d'un devis", top_k=5)
    assert result.chunks == []


async def test_ingestion_flags_injection_and_hidden_characters() -> None:
    ingestor = DocumentIngestor(HashingEmbedder(), chunk_size=400, chunk_overlap=50)
    vendor = next(doc for doc in demo_documents() if "vendor" in (doc.source or ""))
    prepared = await ingestor.prepare(title=vendor.title, content=vendor.content)
    assert prepared.flagged_chunks >= 1
    risks = {c.metadata["injection_risk"] for c in prepared.chunks}
    assert "high" in risks

    hidden = await ingestor.prepare(title="Note", content="Texte​ normal‮ sur les devis.")
    assert hidden.hidden_characters_removed == 2
    assert "​" not in hidden.content
    assert hidden.chunks[0].metadata["injection_signals"] == ["hidden_characters_in_document"]


async def test_ingestion_is_idempotent_on_content() -> None:
    embedder = HashingEmbedder()
    store, ingestor = InMemoryKnowledgeStore(), DocumentIngestor(embedder)
    first = await ingest_demo_documents(store, ingestor, "default")
    second = await ingest_demo_documents(store, ingestor, "default")
    assert all(created for _, created, _ in first)
    assert not any(created for _, created, _ in second)


async def test_empty_documents_are_rejected() -> None:
    with pytest.raises(IngestionError):
        await DocumentIngestor(HashingEmbedder()).prepare(title="Empty", content="   ​​   ")
