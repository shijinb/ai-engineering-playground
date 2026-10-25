"""Retries with backoff and provider fallback, applied the same way to every provider."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from aiplay.providers.base import Provider
from aiplay.types import Completion, CompletionRequest, ProviderError, RefusedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3  # total tries
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.2


class RetryingProvider:
    """Retries retryable errors (429, 5xx, timeouts) with exponential backoff; honours retry-after."""

    def __init__(
        self,
        inner: Provider,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.inner = inner
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self.name = inner.name
        self.model = inner.model

    async def complete(self, request: CompletionRequest) -> Completion:
        for attempt in range(1, self.policy.attempts + 1):
            try:
                return await self.inner.complete(request)
            except ProviderError as exc:
                if not exc.retryable or attempt == self.policy.attempts:
                    raise
                delay = exc.retry_after if exc.retry_after is not None else self.policy.base_delay * 2 ** (attempt - 1)
                delay = min(delay, self.policy.max_delay) * (1 + random.uniform(-self.policy.jitter, self.policy.jitter))
                logger.warning("retrying", extra={"provider": self.name, "attempt": attempt, "delay": round(delay, 2), "error": str(exc)})
                await self._sleep(max(0.0, delay))
        raise AssertionError("unreachable")  # pragma: no cover


class FallbackProvider:
    """Tries providers in order. Moves on after retryable failures and refusals; other errors
    (invalid request, auth) are raised immediately because another provider would not fix them."""

    def __init__(self, providers: Sequence[Provider]) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = list(providers)
        self.name = "fallback(" + ",".join(p.name for p in providers) + ")"
        self.model = self.providers[0].model

    async def complete(self, request: CompletionRequest) -> Completion:
        last: ProviderError | None = None
        for provider in self.providers:
            try:
                return await provider.complete(request)
            except ProviderError as exc:
                if not (exc.retryable or isinstance(exc, RefusedError)):
                    raise
                logger.warning("falling back", extra={"from": provider.name, "error": str(exc)})
                last = exc
        assert last is not None
        raise last
