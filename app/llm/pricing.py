"""Per-model token prices, used to estimate run cost and enforce MAX_RUN_COST_USD.

Prices are USD per million tokens (Anthropic first-party API list prices, October 2026).
Other models need LLM_PRICE_*_PER_MTOK: ``build_price_book`` refuses to start without a price
for the configured model, so the cost limit can never be silently disabled. Costs are
estimates from token counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.types import Usage


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0

    def cost(self, usage: Usage) -> float:
        total = (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_read_tokens * self.cache_read_per_mtok
            + usage.cache_write_tokens * self.cache_write_per_mtok
        )
        return round(total / 1_000_000, 6)


DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-opus-5-5": ModelPricing(4.0, 20.0, cache_read_per_mtok=0.20, cache_write_per_mtok=5.0),
    "claude-sonnet-5-5": ModelPricing(
        2.0, 10.0, cache_read_per_mtok=0.20, cache_write_per_mtok=2.50
    ),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0, cache_read_per_mtok=0.10, cache_write_per_mtok=1.25),
    "offline-simulator-v1": ModelPricing(0.0, 0.0),
}


class PriceBook:
    def __init__(
        self,
        prices: dict[str, ModelPricing] | None = None,
        override: ModelPricing | None = None,
    ) -> None:
        self._prices = dict(DEFAULT_PRICING if prices is None else prices)
        self._override = override

    def knows(self, model: str) -> bool:
        return self._override is not None or model in self._prices

    def cost(self, model: str, usage: Usage) -> float:
        pricing = self._override or self._prices.get(model)
        return pricing.cost(usage) if pricing else 0.0
