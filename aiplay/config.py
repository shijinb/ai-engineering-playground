"""Builds providers from environment variables. Keys are read by each SDK from its own variable:
ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY)."""

from __future__ import annotations

import os

from aiplay.providers.base import Provider
from aiplay.reliability import RetryingProvider, RetryPolicy

PROVIDERS = ("anthropic", "openai", "gemini")


def build(name: str, *, model: str | None = None, retries: int = 3) -> Provider:
    provider: Provider
    if name == "anthropic":
        from aiplay.providers.anthropic_provider import DEFAULT_MODEL, AnthropicProvider

        provider = AnthropicProvider(
            model or os.environ.get("AIPLAY_ANTHROPIC_MODEL", DEFAULT_MODEL),
            effort=os.environ.get("AIPLAY_ANTHROPIC_EFFORT") or None,
            refusal_fallbacks=os.environ.get("AIPLAY_ANTHROPIC_FALLBACKS", "default") != "off",
        )
    elif name == "openai":
        from aiplay.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model or os.environ.get("AIPLAY_OPENAI_MODEL", ""))
    elif name == "gemini":
        from aiplay.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(model or os.environ.get("AIPLAY_GEMINI_MODEL", ""))
    else:
        raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
    return RetryingProvider(provider, RetryPolicy(attempts=retries))
