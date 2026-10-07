"""Embedding providers.

* ``HashingEmbedder`` (default): deterministic feature hashing of word unigrams and bigrams into a
  512-dimension, L2-normalised vector. No model download, no network, identical results in CI.
  It captures lexical similarity only (no synonyms) — the hybrid retriever and the reranker
  compensate, and the production path is an embedding model.
* ``OpenAICompatibleEmbedder``: any ``/embeddings`` endpoint (OpenAI, Ollama, vLLM...). The
  ``dimensions`` parameter pins the size to the database column (vector(512)).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Protocol

import openai

from app.core.config import EMBEDDING_DIMENSIONS, Settings
from app.core.text import STOPWORDS, strip_accents, tokenize
from app.llm.types import LLMError


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    BIGRAM_WEIGHT = 0.5

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    @property
    def model(self) -> str:
        return f"hashing-{self.dimensions}-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        tokens = [
            _fold(strip_accents(token))
            for token in tokenize(text)
            if len(token) >= 2 and token not in STOPWORDS
        ]
        unigrams = Counter(tokens)
        bigrams = Counter(f"{a}_{b}" for a, b in pairwise(tokens))
        vector = [0.0] * self.dimensions
        for features, weight in ((unigrams, 1.0), (bigrams, self.BIGRAM_WEIGHT)):
            for feature, count in features.items():
                # blake2b, not hash(): Python's string hash is randomised per process.
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0  # signed hashing limits collision bias
                vector[index] += sign * weight * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # A zero vector has no cosine distance; use a fixed unit vector instead.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


def _fold(word: str) -> str:
    if len(word) > 4 and word.endswith(("s", "x")):
        return word[:-1]
    return word


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._client = client or openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = await self._client.embeddings.create(
                    model=self._model, input=batch, dimensions=EMBEDDING_DIMENSIONS
                )
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                raise LLMError(f"embedding request failed ({type(exc).__name__})") from exc
            for item in sorted(response.data, key=lambda item: item.index):
                if len(item.embedding) != EMBEDDING_DIMENSIONS:
                    raise LLMError(
                        f"embedding model returned {len(item.embedding)} dimensions; the schema "
                        f"requires {EMBEDDING_DIMENSIONS} (use a model supporting `dimensions`)"
                    )
                vectors.append(list(item.embedding))
        return vectors


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        api_key = settings.embedding_api_key or settings.llm_api_key
        return OpenAICompatibleEmbedder(
            model=settings.embedding_model,
            api_key=api_key.get_secret_value() if api_key else None,
            base_url=settings.embedding_base_url,
        )
    return HashingEmbedder()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
