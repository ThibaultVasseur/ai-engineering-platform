"""Provider selection from configuration (the only place that knows concrete providers)."""

from __future__ import annotations

from app.core.config import Settings
from app.llm.pricing import ModelPricing, PriceBook
from app.llm.providers.anthropic import AnthropicLLM
from app.llm.providers.offline import OfflineLLM
from app.llm.providers.openai import OpenAICompatibleLLM
from app.llm.types import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    match settings.llm_provider:
        case "offline":
            return OfflineLLM()
        case "anthropic":
            return AnthropicLLM(
                model=settings.effective_llm_model,
                api_key=api_key,
                effort=settings.llm_effort,
                max_output_tokens=settings.llm_max_output_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                refusal_fallback=settings.llm_refusal_fallback,
            )
        case "openai":
            return OpenAICompatibleLLM(
                model=settings.effective_llm_model,
                api_key=api_key,
                base_url=settings.llm_base_url,
                max_output_tokens=settings.llm_max_output_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )


def build_price_book(settings: Settings) -> PriceBook:
    """Prices used to enforce MAX_RUN_COST_USD.

    Fails fast when the configured model has no price: costing its calls at 0 would silently
    disable the cost limit of every run.
    """
    override = None
    if (
        settings.llm_price_input_per_mtok is not None
        and settings.llm_price_output_per_mtok is not None
    ):
        override = ModelPricing(
            settings.llm_price_input_per_mtok, settings.llm_price_output_per_mtok
        )
    book = PriceBook(override=override)
    model = settings.effective_llm_model
    if not book.knows(model):
        raise ValueError(
            f"No price is known for model '{model}': set LLM_PRICE_INPUT_PER_MTOK and "
            "LLM_PRICE_OUTPUT_PER_MTOK (USD per million tokens, 0 for a free local model) "
            "so that MAX_RUN_COST_USD can be enforced"
        )
    return book
