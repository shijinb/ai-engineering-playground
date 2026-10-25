# Changelog

## [1.0.0] – 2026-10-25

First release.

- Provider adapters for Anthropic Claude, OpenAI (Responses API) and Google Gemini on their official SDKs, with shared error types for rate limits, transient failures, refusals and truncation.
- Retry with exponential backoff and `Retry-After`; fallback chains across providers.
- Typed structured output from Pydantic models with validation and one repair attempt.
- Versioned prompt templates, JSON telemetry with cost from a dated price table.
- Evaluation runner with two example tasks and synthetic datasets; `aiplay` CLI.
