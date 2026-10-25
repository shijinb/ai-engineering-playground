"""OpenAI via the official `openai` SDK (Responses API).

Structured output uses `text={"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": True}}`.
Strict mode requires every object in the schema to list all properties as required and set
`additionalProperties: false`; `aiplay.structured.strict_schema()` produces that form.
There is no default model: set one your account can use (see README).
"""

from __future__ import annotations

import time
from typing import Any

import openai

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


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, *, client: openai.AsyncOpenAI | None = None, timeout: float = 60.0) -> None:
        if not model:
            raise ValueError("OpenAI model is required (set AIPLAY_OPENAI_MODEL)")
        self.model = model
        self.client = client or openai.AsyncOpenAI(max_retries=0, timeout=timeout)

    def _params(self, request: CompletionRequest) -> dict[str, Any]:
        turns: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
        turns += [{"role": role, "content": text} for role, text in request.followups]
        params: dict[str, Any] = {"model": self.model, "input": turns, "max_output_tokens": request.max_output_tokens}
        if request.system:
            params["instructions"] = request.system
        if request.json_schema is not None:
            params["text"] = {"format": {"type": "json_schema", "name": request.schema_name, "schema": request.json_schema, "strict": True}}
        return params

    async def complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        try:
            response: Any = await self.client.responses.create(**self._params(request))
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after")
            raise RateLimitedError(self.name, str(exc), status=429, retry_after=float(retry_after) if retry_after else None) from exc
        except openai.APIStatusError as exc:
            error_cls = TransientError if exc.status_code >= 500 else ProviderError
            raise error_cls(self.name, exc.message, status=exc.status_code) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise TransientError(self.name, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        for item in response.output or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise RefusedError(self.name, getattr(part, "refusal", "model declined the request"))
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                raise TruncatedError(self.name, f"output truncated at max_output_tokens={request.max_output_tokens}")
            raise ProviderError(self.name, f"incomplete response ({reason})")

        usage = response.usage
        return Completion(
            text=response.output_text,
            provider=self.name,
            model=response.model,
            usage=Usage(usage.input_tokens if usage else 0, usage.output_tokens if usage else 0),
            latency_ms=latency_ms,
            finish_reason=response.status,
            request_id=getattr(response, "_request_id", None),
        )
