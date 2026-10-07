import json

import pytest

from app.agents.rag import RagAgent, is_verbatim, resolve_quote
from app.llm.embeddings import HashingEmbedder
from app.llm.providers.offline import OfflineLLM
from app.observability.tracing import EventType
from app.rag.demo import ingest_demo_documents
from app.rag.ingestion import DocumentIngestor
from app.rag.reranking import LexicalReranker
from app.rag.retrieval import HybridRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.schemas.agent import WorkerInput
from app.schemas.plan import PlanStep
from tests.fakes import ScriptedLLM, StaticRetriever, chunk, make_context


async def demo_retriever() -> HybridRetriever:
    embedder = HashingEmbedder()
    store = InMemoryKnowledgeStore()
    await ingest_demo_documents(store, DocumentIngestor(embedder), "tenant-test")
    return HybridRetriever(store, embedder, reranker=LexicalReranker())


def rag_input(question: str) -> WorkerInput:
    step = PlanStep(
        id="knowledge_review",
        type="knowledge",  # type: ignore[arg-type]
        title="Review docs",
        description=question,
        recommended_agent="rag",
        priority=1,
        success_criteria=["sources cited"],
    )
    return WorkerInput(objective="Concevoir un module de devis", step=step)


async def test_offline_rag_answers_with_verified_citations() -> None:
    context, tracer = make_context(OfflineLLM(), retriever=await demo_retriever())
    outcome = await RagAgent().run(
        context, rag_input("Quelle est la durée de validité d'un devis ?"), step_id="kb"
    )
    answer = outcome.value
    assert "30 jours" in answer.answer
    assert answer.citations
    assert set(answer.cited_sources()) <= context.journal.sources.labels()
    (retrieval,) = tracer.of_type(EventType.RETRIEVAL)
    assert retrieval.payload["filtered_out"] == 1  # the poisoned vendor document
    assert tracer.of_type(EventType.GUARDRAIL_TRIGGERED)


async def test_no_relevant_document_means_no_llm_call() -> None:
    llm = ScriptedLLM([])
    context, _ = make_context(llm, retriever=await demo_retriever())
    outcome = await RagAgent().run(context, rag_input("recette de cuisine au chocolat"))
    assert outcome.llm_calls == 0
    assert outcome.value.missing_information
    assert outcome.value.confidence == "low"


async def test_fabricated_quotes_are_sent_back_for_repair() -> None:
    retriever = StaticRetriever([chunk(0, "Un devis est valable 30 jours après son émission.")])
    fabricated = {
        "summary": "Les devis sont valables 60 jours.",
        "confidence": "high",
        "answer": "Un devis est valable 60 jours [S1].",
        "citations": [{"source_id": "S1", "quote": "valable 60 jours"}],
        "missing_information": [],
    }
    grounded = {
        **fabricated,
        "answer": "Un devis est valable 30 jours [S1].",
        "citations": [{"source_id": "S1", "quote": "valable 30 jours"}],
    }
    llm = ScriptedLLM([json.dumps(fabricated), json.dumps(grounded)])
    context, _ = make_context(llm, retriever=retriever)
    outcome = await RagAgent().run(context, rag_input("Durée de validité d'un devis"))
    assert outcome.value.citations[0].quote == "valable 30 jours"
    assert "not verbatim" in llm.requests[1].messages[-1].text


SOURCE = (
    "Les montants sont stockés en NUMERIC(12,2) — jamais en float. Un devis « accepté » est figé."
)


@pytest.mark.parametrize(
    "quote",
    [
        "les montants sont stockés en numeric(12,2)",  # case
        "Les montants sont stockés en NUMERIC(12,2) - jamais en float",  # dash variant
        'Un devis "accepté" est figé',  # quote marks
        "Les montants sont stockés … jamais en float",  # elision, fragments in order
        "ontants sont stockés en NUMERIC",  # word cut at the start of the excerpt
    ],
)
def test_typographic_differences_and_marked_elisions_are_verbatim(quote: str) -> None:
    assert is_verbatim(quote, SOURCE)


@pytest.mark.parametrize(
    "quote",
    [
        "Amounts are stored as NUMERIC(12,2)",  # translation (seen with a live model)
        "Les montants sont enregistrés en NUMERIC(12,2)",  # paraphrase
        "jamais en float … Les montants sont stockés",  # fragments out of order
        "montants … float … figé",  # isolated words stitched together
        "…",  # nothing quoted
    ],
)
def test_translations_paraphrases_and_stitching_are_not_verbatim(quote: str) -> None:
    assert not is_verbatim(quote, SOURCE)


async def test_citing_an_unknown_source_is_rejected() -> None:
    retriever = StaticRetriever([chunk(0, "Les montants sont stockés en NUMERIC(12,2).")])
    invented = {
        "summary": "Montants en NUMERIC.",
        "confidence": "high",
        "answer": "Les montants sont en NUMERIC [S4].",
        "citations": [{"source_id": "S4", "quote": "NUMERIC"}],
        "missing_information": [],
    }
    fixed = {**invented, "citations": [{"source_id": "S1", "quote": "stockés en NUMERIC(12,2)"}]}
    llm = ScriptedLLM([json.dumps(invented), json.dumps(fixed)])
    context, _ = make_context(llm, retriever=retriever)
    await RagAgent().run(context, rag_input("Comment sont stockés les montants ?"))
    assert "S4 is not one of the provided sources" in llm.requests[1].messages[-1].text


async def test_over_long_verbatim_quotes_are_shortened_not_refused() -> None:
    """Live models quote whole passages whatever the limit: the quote is cut, still verified."""
    passage = " ".join(["Les devis sont valables 30 jours après leur émission."] * 12)
    retriever = StaticRetriever([chunk(0, passage)])
    answer = {
        "summary": "Un devis est valable 30 jours.",
        "confidence": "high",
        "answer": "Un devis est valable 30 jours [S1].",
        "citations": [{"source_id": "S1", "quote": passage}],
        "missing_information": [],
    }
    llm = ScriptedLLM([json.dumps(answer)])
    context, _ = make_context(llm, retriever=retriever)
    outcome = await RagAgent().run(context, rag_input("Durée de validité d'un devis"))
    quote = outcome.value.citations[0].quote
    assert len(llm.requests) == 1  # accepted without a repair round
    assert len(quote) <= 300
    assert quote.endswith("…")
    assert is_verbatim(quote, passage)


POLICY = (
    "La durée de validité par défaut d'un devis est de 30 jours. Passé ce délai, le devis "
    "passe automatiquement au statut « expiré »."
)


def test_near_verbatim_quotes_resolve_to_the_source_sentence() -> None:
    # Seen live: "par défaut" dropped without marking the cut.
    near = "La durée de validité d'un devis est de 30 jours"
    assert resolve_quote(near, POLICY) == (
        "La durée de validité par défaut d'un devis est de 30 jours."
    )
    exact = "Passé ce délai, le devis passe automatiquement"
    assert resolve_quote(exact, POLICY) == exact  # an exact excerpt is kept as written


@pytest.mark.parametrize(
    "quote",
    [
        "La durée de validité par défaut d'un devis est de 60 jours",  # changed figure
        "The default validity of a quote is 30 days",  # translation
        "Un devis reste valable sans limite de durée",  # invention
        "valable 30 jours",  # too little of any sentence
    ],
)
def test_unsupported_quotes_do_not_resolve(quote: str) -> None:
    assert resolve_quote(quote, POLICY) is None


async def test_near_verbatim_quotes_are_replaced_by_the_source_text() -> None:
    answer = {
        "summary": "Un devis est valable 30 jours.",
        "confidence": "high",
        "answer": "Un devis est valable 30 jours par défaut [S1].",
        "citations": [
            {"source_id": "S1", "quote": "La durée de validité d'un devis est de 30 jours"}
        ],
        "missing_information": [],
    }
    llm = ScriptedLLM([json.dumps(answer)])
    context, _ = make_context(llm, retriever=StaticRetriever([chunk(0, POLICY)]))
    outcome = await RagAgent().run(context, rag_input("Durée de validité d'un devis"))
    assert len(llm.requests) == 1  # no repair round
    assert outcome.value.citations[0].quote == (
        "La durée de validité par défaut d'un devis est de 30 jours."
    )
