from __future__ import annotations

from typing import Protocol

from aiplay.types import Completion, CompletionRequest


class Provider(Protocol):
    """Anything that can complete a request. Adapters wrap each vendor SDK behind this interface."""

    name: str
    model: str

    async def complete(self, request: CompletionRequest) -> Completion: ...
