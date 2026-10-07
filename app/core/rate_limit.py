"""Per-tenant token bucket.

Each tenant gets RATE_LIMIT_PER_MINUTE tokens refilled continuously; expensive endpoints spend
tokens (an evaluation costs more than creating a task). State lives in process memory, which is
correct for a single instance only — behind several replicas the bucket must move to a shared
store (Redis) or to the API gateway (see docs/security.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = float(per_minute)
        self._refill_per_second = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last refill)

    def acquire(self, key: str, cost: float = 1.0) -> float | None:
        """Spend ``cost`` tokens; returns None when allowed, else seconds until it would be."""
        now = self._clock()
        tokens, last = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._refill_per_second)
        if tokens >= cost:
            self._buckets[key] = (tokens - cost, now)
            return None
        self._buckets[key] = (tokens, now)
        return (cost - tokens) / self._refill_per_second
