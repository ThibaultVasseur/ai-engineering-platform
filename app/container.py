"""Composition root: the only place where concrete implementations are chosen and wired.

Everything else receives its collaborators through constructors (dependency injection), which
is what makes the providers swappable and the components testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.rate_limit import RateLimiter
from app.core.security import ApiKeyAuthenticator
from app.db.database import Database
from app.db.repositories.documents import PostgresKnowledgeStore
from app.llm.embeddings import EmbeddingProvider, build_embedder
from app.llm.factory import build_llm_client, build_price_book
from app.observability.langfuse import build_langfuse_tracer_factory
from app.orchestration.factory import RunDependencies, build_agent_suite
from app.orchestration.graph import build_graph
from app.orchestration.policies import RunLimits
from app.orchestration.runner import GraphRunner
from app.rag.ingestion import DocumentIngestor
from app.rag.reranking import LexicalReranker
from app.rag.retrieval import HybridRetriever
from app.rag.service import KnowledgeService
from app.rag.store import KnowledgeStore
from app.services.evaluation_service import EvaluationService
from app.services.platform_data import SqlPlatformData
from app.services.run_service import RunService
from app.services.task_service import TaskService
from app.tools.catalog import default_tool_registry
from app.tools.registry import ToolRegistry


@dataclass
class Container:
    settings: Settings
    database: Database
    authenticator: ApiKeyAuthenticator
    embedder: EmbeddingProvider
    knowledge_store: KnowledgeStore
    retriever: HybridRetriever
    tool_registry: ToolRegistry
    run_dependencies: RunDependencies
    graph_mermaid: str
    task_service: TaskService
    knowledge_service: KnowledgeService
    run_service: RunService
    evaluation_service: EvaluationService
    rate_limiter: RateLimiter

    async def aclose(self) -> None:
        await self.run_service.shutdown()
        await self.database.dispose()


def build_retriever(
    settings: Settings, store: KnowledgeStore, embedder: EmbeddingProvider
) -> HybridRetriever:
    return HybridRetriever(
        store,
        embedder,
        reranker=LexicalReranker(),
        candidate_pool=settings.rag_candidate_pool,
        block_suspicious=settings.rag_block_suspicious_chunks,
    )


def build_ingestor(settings: Settings, embedder: EmbeddingProvider) -> DocumentIngestor:
    return DocumentIngestor(
        embedder, chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
    )


def build_container(settings: Settings) -> Container:
    database = Database(
        settings.database_url.get_secret_value(),
        pool_size=settings.db_pool_size,
        echo=settings.db_echo,
    )
    embedder = build_embedder(settings)
    store = PostgresKnowledgeStore(database)
    retriever = build_retriever(settings, store, embedder)
    tools = default_tool_registry(
        timeout_seconds=settings.tool_timeout_seconds,
        max_output_chars=settings.tool_max_output_chars,
    )
    graph = build_graph()
    run_dependencies = RunDependencies(
        llm=build_llm_client(settings),
        prices=build_price_book(settings),
        tools=tools,
        agents=build_agent_suite(
            supervisor_strategy=settings.supervisor_strategy, rag_top_k=settings.rag_top_k
        ),
        limits=RunLimits.from_settings(settings),
        retriever=retriever,
        platform_data=SqlPlatformData(database),
        structured_max_attempts=settings.llm_structured_max_attempts,
    )
    return Container(
        settings=settings,
        database=database,
        authenticator=ApiKeyAuthenticator(settings.api_keys),
        embedder=embedder,
        knowledge_store=store,
        retriever=retriever,
        tool_registry=tools,
        run_dependencies=run_dependencies,
        graph_mermaid=graph.get_graph().draw_mermaid(),
        task_service=TaskService(database),
        knowledge_service=KnowledgeService(store, build_ingestor(settings, embedder), retriever),
        run_service=RunService(
            database,
            run_dependencies,
            settings,
            runner=GraphRunner(graph),
            tracer_factories=[f for f in (build_langfuse_tracer_factory(settings),) if f],
        ),
        evaluation_service=EvaluationService(database, settings),
        rate_limiter=RateLimiter(settings.rate_limit_per_minute),
    )
