"""Hybrid retrieval.

    QUESTION -> EMBEDDING -> VECTOR SEARCH ---\
             -> KEYWORDS  -> FULL-TEXT SEARCH -+-> RRF FUSION -> INJECTION FILTER -> RERANK -> TOP K

* Vector search finds paraphrases; full-text search nails exact terms (codes, names, acronyms).
  Reciprocal Rank Fusion merges the two rankings without having to calibrate their scores.
* Metadata filters (category, document ids) are applied inside both searches.
* HIGH-risk chunks (prompt-injection signals detected at ingestion) are excluded and counted.
"""

from __future__ import annotations

import time

from app.core.text import keywords
from app.llm.embeddings import EmbeddingProvider
from app.rag.reranking import Reranker
from app.rag.store import KnowledgeStore
from app.schemas.knowledge import RetrievalResult, RetrievedChunk, SearchFilters

RRF_K = 60


def rrf_fuse(rankings: list[list[RetrievedChunk]], *, k: int = RRF_K) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            known = merged.get(chunk.chunk_id)
            if known is None:
                merged[chunk.chunk_id] = chunk
            else:
                merged[chunk.chunk_id] = known.model_copy(
                    update={
                        "vector_score": known.vector_score
                        if known.vector_score is not None
                        else chunk.vector_score,
                        "text_rank": known.text_rank
                        if known.text_rank is not None
                        else chunk.text_rank,
                    }
                )
    fused = [
        chunk.model_copy(update={"score": round(scores[chunk_id], 6)})
        for chunk_id, chunk in merged.items()
    ]
    return sorted(fused, key=lambda chunk: chunk.score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        embedder: EmbeddingProvider,
        *,
        reranker: Reranker | None = None,
        candidate_pool: int = 20,
        block_suspicious: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._candidate_pool = candidate_pool
        self._block_suspicious = block_suspicious

    async def search(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        [vector] = await self._embedder.embed([query])
        terms = keywords(query, limit=12)
        vector_hits = await self._store.vector_search(
            tenant_id, vector, limit=self._candidate_pool, filters=filters
        )
        text_hits = await self._store.text_search(
            tenant_id, terms, limit=self._candidate_pool, filters=filters
        )
        candidates = rrf_fuse([vector_hits, text_hits])

        filtered_out = 0
        if self._block_suspicious:
            safe = [c for c in candidates if c.metadata.get("injection_risk") != "high"]
            filtered_out = len(candidates) - len(safe)
            candidates = safe
        if self._reranker is not None:
            candidates = self._reranker.rerank(query, candidates)
        return RetrievalResult(
            query=query,
            chunks=candidates[:top_k],
            filtered_out=filtered_out,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
