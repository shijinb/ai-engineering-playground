"""Scripted provider for tests and offline demos. Never calls a network."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from aiplay.types import Completion, CompletionRequest, ProviderError, Usage

Reply = str | ProviderError | Callable[[CompletionRequest], str]


class FakeProvider:
    def __init__(self, replies: Iterable[Reply] = (), *, name: str = "fake", model: str = "fake-model", latency_ms: float = 5.0):
        self.name = name
        self.model = model
        self.latency_ms = latency_ms
        self._replies = list(replies)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"{self.name}: no scripted reply left")
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, ProviderError):
            raise reply
        text = reply(request) if callable(reply) else reply
        words = len(request.prompt.split())
        return Completion(text, self.name, self.model, Usage(words * 2, len(text) // 4), self.latency_ms, "stop")
