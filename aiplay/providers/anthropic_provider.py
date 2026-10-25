"""Anthropic Claude via the official `anthropic` SDK (Messages API).

Notes from the current API (checked 2026-10-25):
- Structured output uses `output_config={"format": {"type": "json_schema", "schema": ...}}`.
- `temperature`/`top_p`/`top_k` are not sent: current models reject sampling parameters.
- Thinking depth is controlled with `output_config.effort`; Claude Opus 5.5 defaults to "medium".
- A safety decline returns HTTP 200 with `stop_reason == "refusal"`; server-side fallbacks
  (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`) re-run a declined request on
  another model inside the same call. Enabled by default here; pass `refusal_fallbacks=False` to opt out.
  Haiku models have no server-side fallback, so the parameter is never sent for them.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic

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

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        effort: str | None = None,
        refusal_fallbacks: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        # max_retries=0: retries are handled uniformly across providers by aiplay.reliability.
        self.client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=timeout)
        self.effort = effort
        self.refusal_fallbacks = refusal_fallbacks and "haiku" not in model

    def _params(self, request: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
        messages += [{"role": role, "content": text} for role, text in request.followups]
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        params: dict[str, Any] = {"model": self.model, "max_tokens": request.max_output_tokens, "messages": messages}
        if request.system:
            params["system"] = request.system
        if output_config:
            params["output_config"] = output_config
        return params

    async def complete(self, request: CompletionRequest) -> Completion:
        params = self._params(request)
        started = time.perf_counter()
        try:
            if self.refusal_fallbacks:
                response: Any = await self.client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                response = await self.client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after")
            raise RateLimitedError(self.name, str(exc), status=429, retry_after=float(retry_after) if retry_after else None) from exc
        except anthropic.APIStatusError as exc:
            error_cls = TransientError if exc.status_code >= 500 else ProviderError
            raise error_cls(self.name, exc.message, status=exc.status_code) from exc
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise TransientError(self.name, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise RefusedError(self.name, f"model declined the request (category: {category})")
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            raise TruncatedError(self.name, f"output truncated at max_tokens={request.max_output_tokens}")

        return Completion(
            text=text,
            provider=self.name,
            model=response.model,  # may differ from self.model when a fallback model served the request
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            latency_ms=latency_ms,
            finish_reason=response.stop_reason,
            request_id=getattr(response, "_request_id", None),
        )
