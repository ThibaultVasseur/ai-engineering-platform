"""Optional reranking stage.

``LexicalReranker`` combines the fused retrieval rank with the share of query keywords present in
the chunk, and drops chunks that have neither lexical overlap nor a reasonable semantic score.
It is cheap and deterministic; a cross-encoder or an LLM judge can implement the same
``Reranker`` protocol when quality matters more than latency.
"""

from __future__ import annotations

from typing import Protocol

from app.core.text import keyword_coverage
from app.schemas.knowledge import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class LexicalReranker:
    # Unrelated texts reach ~0.2 cosine with the 512-d hashing embedder (bucket collisions);
    # without any shared keyword, a chunk must be clearly closer than that to be kept.
    def __init__(self, *, fusion_weight: float = 0.5, min_vector_score: float = 0.3) -> None:
        self._fusion_weight = fusion_weight
        self._min_vector_score = min_vector_score

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        best_fused = max(chunk.score for chunk in chunks) or 1.0
        reranked = []
        for chunk in chunks:
            coverage = keyword_coverage(query, f"{chunk.title} {chunk.content}")
            if coverage == 0 and (chunk.vector_score or 0.0) < self._min_vector_score:
                continue  # no shared keyword and weak semantic similarity: noise
            relevance = (
                self._fusion_weight * (chunk.score / best_fused)
                + (1 - self._fusion_weight) * coverage
            )
            reranked.append(chunk.model_copy(update={"score": round(relevance, 4)}))
        return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)
