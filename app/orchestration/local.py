"""Self-contained runtime without PostgreSQL: in-memory knowledge base loaded with the demo
corpus, in-memory platform data. Used by the CLI demo, the evaluation runner and tests, so the
whole multi-agent pipeline can run on a fresh clone with zero infrastructure.
"""

from __future__ import annotations

from app.core.config import Settings
from app.llm.embeddings import build_embedder
from app.llm.factory import build_llm_client, build_price_book
from app.llm.types import LLMClient
from app.orchestration.factory import RunDependencies, build_agent_suite
from app.orchestration.policies import RunLimits
from app.rag.demo import ingest_demo_documents
from app.rag.ingestion import DocumentIngestor
from app.rag.reranking import LexicalReranker
from app.rag.retrieval import HybridRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.services.platform_data import InMemoryPlatformData
from app.tools.catalog import default_tool_registry


async def build_in_memory_dependencies(
    settings: Settings,
    *,
    tenant_id: str,
    llm: LLMClient | None = None,
    with_demo_data: bool = True,
) -> RunDependencies:
    embedder = build_embedder(settings)
    store = InMemoryKnowledgeStore()
    if with_demo_data:
        ingestor = DocumentIngestor(
            embedder, chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
        )
        await ingest_demo_documents(store, ingestor, tenant_id)
    retriever = HybridRetriever(
        store,
        embedder,
        reranker=LexicalReranker(),
        candidate_pool=settings.rag_candidate_pool,
        block_suspicious=settings.rag_block_suspicious_chunks,
    )
    return RunDependencies(
        llm=llm or build_llm_client(settings),
        prices=build_price_book(settings),
        tools=default_tool_registry(
            timeout_seconds=settings.tool_timeout_seconds,
            max_output_chars=settings.tool_max_output_chars,
        ),
        agents=build_agent_suite(
            supervisor_strategy=settings.supervisor_strategy, rag_top_k=settings.rag_top_k
        ),
        limits=RunLimits.from_settings(settings),
        retriever=retriever,
        platform_data=InMemoryPlatformData.with_demo_data(tenant_id)
        if with_demo_data
        else InMemoryPlatformData(),
        structured_max_attempts=settings.llm_structured_max_attempts,
    )
