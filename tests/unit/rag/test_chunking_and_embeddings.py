import math
from itertools import pairwise

import pytest

from app.core.config import EMBEDDING_DIMENSIONS
from app.llm.embeddings import HashingEmbedder, cosine_similarity
from app.rag.chunking import chunk_text

PARAGRAPH = "Les montants sont stockés en NUMERIC et manipulés avec Decimal. " * 4


def test_chunks_respect_size_and_keep_offsets() -> None:
    text = "\n\n".join(f"Section {i}. {PARAGRAPH}" for i in range(8))
    chunks = chunk_text(text, size=600, overlap=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 600
        assert text[chunk.start : chunk.end].strip() == chunk.content
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap() -> None:
    sentences = " ".join(f"Phrase numéro {i} sur les devis." for i in range(80))
    chunks = chunk_text(sentences, size=300, overlap=80)
    for previous, current in pairwise(chunks):
        assert current.start < previous.end  # shared trailing sentence(s)


def test_huge_sentences_are_hard_wrapped() -> None:
    chunks = chunk_text("x" * 2_000, size=500, overlap=100)
    assert all(len(chunk.content) <= 500 for chunk in chunks)
    assert sum(len(chunk.content) for chunk in chunks) >= 2_000


def test_small_text_is_one_chunk_and_empty_text_none() -> None:
    assert len(chunk_text("Un seul paragraphe court.", size=900, overlap=150)) == 1
    assert chunk_text("", size=900, overlap=150) == []
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", size=100, overlap=100)


async def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder()
    [first], [second] = (
        await embedder.embed(["Durée de validité d'un devis"]),
        await embedder.embed(["Durée de validité d'un devis"]),
    )
    assert first == second
    assert len(first) == EMBEDDING_DIMENSIONS
    assert math.isclose(sum(v * v for v in first), 1.0, rel_tol=1e-9)


async def test_related_texts_are_closer_than_unrelated_ones() -> None:
    embedder = HashingEmbedder()
    query, related, unrelated = await embedder.embed(
        [
            "validité des devis expirés",
            "Un devis expiré ne peut plus être accepté, sa validité est de 30 jours.",
            "Recette de gâteau au chocolat et à la vanille.",
        ]
    )
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


async def test_accents_and_plurals_are_folded() -> None:
    embedder = HashingEmbedder()
    plain, accented = await embedder.embed(["securite des donnees", "Sécurité des données"])
    assert cosine_similarity(plain, accented) == pytest.approx(1.0)


async def test_text_without_tokens_still_has_a_unit_vector() -> None:
    [vector] = await HashingEmbedder().embed(["!!! ..."])
    assert math.isclose(sum(v * v for v in vector), 1.0)
