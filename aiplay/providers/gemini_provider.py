"""Google Gemini via the official `google-genai` SDK.

Structured output uses `response_mime_type="application/json"` with `response_json_schema`.
There is no default model: set one your account can use (see README).
"""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from aiplay.types import (
    Completion,
    CompletionRequest,
    ProviderError,
    RateLimitedError,
    RefusedError,
    TransientError,
    TruncatedError,
    Usage,
)

_REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, *, client: genai.Client | None = None) -> None:
        if not model:
            raise ValueError("Gemini model is required (set AIPLAY_GEMINI_MODEL)")
        self.model = model
        self.client = client or genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def _call(self, request: CompletionRequest) -> tuple[list[types.Content], types.GenerateContentConfig]:
        contents = [types.Content(role="user", parts=[types.Part(text=request.prompt)])]
        for role, text in request.followups:
            contents.append(types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=text)]))
        config: dict[str, Any] = {"max_output_tokens": request.max_output_tokens}
        if request.system:
            config["system_instruction"] = request.system
        if request.json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = request.json_schema
        return contents, types.GenerateContentConfig(**config)

    async def complete(self, request: CompletionRequest) -> Completion:
        contents, config = self._call(request)
        started = time.perf_counter()
        try:
            response = await self.client.aio.models.generate_content(model=self.model, contents=contents, config=config)
        except genai_errors.APIError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            if status == 429:
                raise RateLimitedError(self.name, str(exc), status=429) from exc
            error_cls = TransientError if status >= 500 else ProviderError
            raise error_cls(self.name, str(exc), status=status or None) from exc
        except (TimeoutError, ConnectionError) as exc:
            raise TransientError(self.name, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise RefusedError(self.name, f"prompt blocked ({response.prompt_feedback.block_reason})")
        candidate = response.candidates[0] if response.candidates else None
        reason = str(getattr(candidate.finish_reason, "value", candidate.finish_reason)) if candidate and candidate.finish_reason else None
        if reason in _REFUSAL_REASONS:
            raise RefusedError(self.name, f"response blocked ({reason})")
        if reason == "MAX_TOKENS":
            raise TruncatedError(self.name, f"output truncated at max_output_tokens={request.max_output_tokens}")

        meta = response.usage_metadata
        return Completion(
            text=response.text or "",
            provider=self.name,
            model=response.model_version or self.model,
            usage=Usage((meta.prompt_token_count or 0) if meta else 0, (meta.candidates_token_count or 0) if meta else 0),
            latency_ms=latency_ms,
            finish_reason=reason,
            request_id=response.response_id,
        )
