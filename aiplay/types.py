"""Provider-neutral request/response types and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    system: str | None = None
    max_output_tokens: int = 4096
    # When set, the provider is asked for JSON matching this JSON Schema (native structured output).
    json_schema: dict[str, Any] | None = None
    schema_name: str = "result"
    # Extra conversation turns appended after `prompt` (used for repair re-asks): (role, text) pairs.
    followups: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    request_id: str | None = None


class ProviderError(Exception):
    """Base error. `retryable` says whether the same request may succeed later."""

    retryable = False

    def __init__(self, provider: str, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status
        self.retry_after = retry_after


class RateLimitedError(ProviderError):
    retryable = True


class TransientError(ProviderError):
    """Timeouts, connection failures and 5xx responses."""

    retryable = True


class RefusedError(ProviderError):
    """The model declined the request (e.g. a safety refusal). Retrying unchanged will not help."""


class TruncatedError(ProviderError):
    """Output hit the max token limit before finishing."""
