import pytest
from pydantic import ValidationError

from app.llm.factory import build_price_book
from app.llm.metering import MeteredLLMClient
from app.llm.pricing import ModelPricing, PriceBook
from app.llm.types import (
    BudgetExceededError,
    CallMetadata,
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    Usage,
)
from app.observability.tracing import EventType, InMemoryTracer
from tests.fakes import ScriptedLLM
from tests.support import make_settings


def test_opus_cost_uses_list_prices() -> None:
    book = PriceBook()
    usage = Usage(
        input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000
    )  # 4.00 + 2.00 + 0.20
    assert book.cost("claude-opus-5-5", usage) == pytest.approx(6.2)


def test_unknown_model_is_free_unless_overridden() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert PriceBook().cost("some-local-model", usage) == 0.0
    overridden = PriceBook(override=ModelPricing(0.5, 1.5))
    assert overridden.cost("some-local-model", usage) == pytest.approx(2.0)


class CountingBudget:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self.cost = 0.0

    def check_llm_call(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceededError("max_llm_calls", "exhausted")

    def record_llm_call(self, usage: Usage, cost_usd: float) -> None:
        self.calls += 1
        self.cost += cost_usd


def request() -> LLMRequest:
    return LLMRequest(
        system="s",
        messages=[Message.user("u")],
        metadata=CallMetadata(
            agent="planner", prompt_name="planner", prompt_version="v1", step_id="s1"
        ),
    )


async def test_metered_client_records_usage_and_traces_calls() -> None:
    tracer = InMemoryTracer()
    budget = CountingBudget(max_calls=5)
    prices = PriceBook({"scripted-model": ModelPricing(1_000_000.0, 0.0)})  # $1 per input token
    client = MeteredLLMClient(ScriptedLLM(["{}"]), budget=budget, tracer=tracer, prices=prices)

    await client.complete(request())

    assert budget.calls == 1
    assert budget.cost == pytest.approx(100.0)  # 100 input tokens
    (event,) = tracer.of_type(EventType.LLM_CALL)
    assert event.agent == "planner"
    assert event.step_id == "s1"
    assert event.payload["prompt_version"] == "v1"
    assert event.payload["input_tokens"] == 100


async def test_metered_client_refuses_calls_beyond_budget() -> None:
    llm = ScriptedLLM(["{}", "{}"])
    client = MeteredLLMClient(
        llm, budget=CountingBudget(max_calls=1), tracer=InMemoryTracer(), prices=PriceBook()
    )
    await client.complete(request())
    with pytest.raises(BudgetExceededError):
        await client.complete(request())
    assert len(llm.requests) == 1  # the provider was never called the second time


def test_a_model_without_a_price_is_refused_at_start_up() -> None:
    """Costing calls at $0 would silently disable MAX_RUN_COST_USD."""
    with pytest.raises(ValueError, match="No price is known for model 'gpt-test-mini'"):
        build_price_book(make_settings(llm_provider="openai", llm_model="gpt-test-mini"))
    priced = build_price_book(
        make_settings(
            llm_provider="openai",
            llm_model="gpt-test-mini",
            llm_price_input_per_mtok=0.4,
            llm_price_output_per_mtok=1.6,
        )
    )
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert priced.cost("gpt-test-mini", usage) == pytest.approx(2.0)
    free_local_model = build_price_book(
        make_settings(
            llm_provider="openai",
            llm_model="llama3",
            llm_price_input_per_mtok=0,
            llm_price_output_per_mtok=0,
        )
    )
    assert free_local_model.knows("llama3")


def test_built_in_models_need_no_price_configuration() -> None:
    assert build_price_book(make_settings()).knows("offline-simulator-v1")
    assert build_price_book(make_settings(llm_provider="anthropic")).knows("claude-opus-5-5")


def test_prices_are_configured_together() -> None:
    with pytest.raises(ValidationError, match="Set both LLM_PRICE"):
        make_settings(llm_price_input_per_mtok=1.0)


class RenamingLLM:
    """Answers under another model name (a dated snapshot or a server-side fallback model)."""

    provider = "fake"
    model = "priced-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="{}",
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=1_000_000),
            model="priced-model-2026-10-01",
            provider="fake",
        )


async def test_calls_served_by_an_unpriced_model_cost_the_configured_model_price() -> None:
    budget = CountingBudget(max_calls=1)
    prices = PriceBook({"priced-model": ModelPricing(2.0, 0.0)})
    client = MeteredLLMClient(RenamingLLM(), budget=budget, tracer=InMemoryTracer(), prices=prices)
    await client.complete(request())
    assert budget.cost == pytest.approx(2.0)
