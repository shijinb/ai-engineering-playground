"""Adapters tested through the real vendor SDKs with a mocked HTTP transport: these check the exact
requests each SDK sends and how each response shape is mapped. No network, no API keys."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx
import httpx2
import openai
import pytest
from google import genai
from google.genai import types as genai_types

from aiplay.providers.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from aiplay.providers.gemini_provider import GeminiProvider
from aiplay.providers.openai_provider import OpenAIProvider
from aiplay.types import CompletionRequest, ProviderError, RateLimitedError, RefusedError, TransientError, TruncatedError

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}
REQUEST = CompletionRequest(prompt="Extract x", system="Be precise.", max_output_tokens=512, json_schema=SCHEMA, schema_name="Thing")


class Recorder:
    def __init__(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        self.status, self.body, self.headers = status, body, headers or {}
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        module = httpx2 if isinstance(request, httpx2.Request) else httpx
        return module.Response(self.status, json=self.body, headers=self.headers)

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


# --- Anthropic ------------------------------------------------------------------------------------


def claude_message(text: str = '{"x": 1}', stop: str = "end_turn", **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 8},
        **extra,
    }


def claude(recorder: Recorder, **kwargs: Any) -> AnthropicProvider:
    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(recorder))
    client = anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)
    return AnthropicProvider(client=client, **kwargs)


async def test_anthropic_request_shape_and_usage() -> None:
    recorder = Recorder(200, claude_message())
    completion = await claude(recorder, effort="low").complete(REQUEST)

    request = recorder.requests[0]
    body = recorder.last_json
    assert request.url.path == "/v1/messages"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 512
    assert body["system"] == "Be precise."
    assert body["messages"] == [{"role": "user", "content": "Extract x"}]
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body  # current Claude models reject sampling parameters
    assert completion.text == '{"x": 1}'
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (120, 8)


async def test_anthropic_without_fallbacks_uses_plain_messages_endpoint() -> None:
    recorder = Recorder(200, claude_message())
    await claude(recorder, refusal_fallbacks=False).complete(CompletionRequest(prompt="hi"))
    assert "anthropic-beta" not in recorder.requests[0].headers
    assert "fallbacks" not in recorder.last_json and "output_config" not in recorder.last_json


@pytest.mark.parametrize(
    ("stop", "error"),
    [("refusal", RefusedError), ("max_tokens", TruncatedError)],
)
async def test_anthropic_stop_reasons(stop: str, error: type[ProviderError]) -> None:
    extra = {"stop_details": {"type": "refusal", "category": "cyber", "explanation": None}} if stop == "refusal" else {}
    with pytest.raises(error):
        await claude(Recorder(200, claude_message(stop=stop, **extra))).complete(REQUEST)


@pytest.mark.parametrize(
    ("status", "headers", "error"),
    [(429, {"retry-after": "7"}, RateLimitedError), (529, {}, TransientError), (400, {}, ProviderError)],
)
async def test_anthropic_http_errors(status: int, headers: dict[str, str], error: type[ProviderError]) -> None:
    body = {"type": "error", "error": {"type": "x", "message": "boom"}}
    with pytest.raises(error) as caught:
        await claude(Recorder(status, body, headers)).complete(REQUEST)
    assert type(caught.value) is error
    if status == 429:
        assert caught.value.retry_after == 7.0


# --- OpenAI ---------------------------------------------------------------------------------------


def openai_response(text: str = '{"x": 1}', *, status: str = "completed", refusal: str | None = None, **extra: Any) -> dict[str, Any]:
    part = {"type": "refusal", "refusal": refusal} if refusal else {"type": "output_text", "text": text, "annotations": []}
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1_790_000_000,
        "model": "test-model",
        "status": status,
        "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [part]}],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 90,
            "output_tokens": 6,
            "total_tokens": 96,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        **extra,
    }


def gpt(recorder: Recorder) -> OpenAIProvider:
    http = openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(recorder))  # openai 3.x uses httpx2
    return OpenAIProvider("test-model", client=openai.AsyncOpenAI(api_key="test", http_client=http, max_retries=0))


async def test_openai_request_shape_and_usage() -> None:
    recorder = Recorder(200, openai_response())
    completion = await gpt(recorder).complete(REQUEST)

    body = recorder.last_json
    assert recorder.requests[0].url.path == "/v1/responses"
    assert body["model"] == "test-model" and body["instructions"] == "Be precise."
    assert body["input"] == [{"role": "user", "content": "Extract x"}]
    assert body["max_output_tokens"] == 512
    assert body["text"] == {"format": {"type": "json_schema", "name": "Thing", "schema": SCHEMA, "strict": True}}
    assert completion.text == '{"x": 1}'
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (90, 6)


async def test_openai_refusal_and_truncation() -> None:
    with pytest.raises(RefusedError):
        await gpt(Recorder(200, openai_response(refusal="I can't help with that"))).complete(REQUEST)
    truncated = openai_response(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    with pytest.raises(TruncatedError):
        await gpt(Recorder(200, truncated)).complete(REQUEST)


async def test_openai_rate_limit() -> None:
    body = {"error": {"message": "slow down", "type": "rate_limit", "code": None, "param": None}}
    with pytest.raises(RateLimitedError) as caught:
        await gpt(Recorder(429, body, {"retry-after": "3"})).complete(REQUEST)
    assert caught.value.retry_after == 3.0


def test_openai_requires_a_model() -> None:
    with pytest.raises(ValueError, match="AIPLAY_OPENAI_MODEL"):
        OpenAIProvider("")


# --- Gemini ---------------------------------------------------------------------------------------


def gemini_response(text: str = '{"x": 1}', finish: str = "STOP", **extra: Any) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish, "index": 0}],
        "usageMetadata": {"promptTokenCount": 70, "candidatesTokenCount": 5, "totalTokenCount": 75},
        "modelVersion": "test-gemini",
        "responseId": "r-1",
        **extra,
    }


def gem(recorder: Recorder) -> GeminiProvider:
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    client = genai.Client(api_key="test", http_options=genai_types.HttpOptions(httpx_async_client=http))
    return GeminiProvider("test-gemini", client=client)


async def test_gemini_request_shape_and_usage() -> None:
    recorder = Recorder(200, gemini_response())
    completion = await gem(recorder).complete(REQUEST)

    body = recorder.last_json
    assert recorder.requests[0].url.path.endswith("/models/test-gemini:generateContent")
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Extract x"}]}]
    assert body["systemInstruction"]["parts"][0]["text"] == "Be precise."
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == SCHEMA
    assert config["maxOutputTokens"] == 512
    assert completion.text == '{"x": 1}'
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (70, 5)
    assert completion.model == "test-gemini" and completion.request_id == "r-1"


@pytest.mark.parametrize(("finish", "error"), [("SAFETY", RefusedError), ("MAX_TOKENS", TruncatedError)])
async def test_gemini_finish_reasons(finish: str, error: type[ProviderError]) -> None:
    with pytest.raises(error):
        await gem(Recorder(200, gemini_response(finish=finish))).complete(REQUEST)


@pytest.mark.parametrize(("status", "error"), [(429, RateLimitedError), (503, TransientError), (400, ProviderError)])
async def test_gemini_http_errors(status: int, error: type[ProviderError]) -> None:
    body = {"error": {"code": status, "message": "boom", "status": "X"}}
    with pytest.raises(error) as caught:
        await gem(Recorder(status, body)).complete(REQUEST)
    assert type(caught.value) is error


async def test_anthropic_never_sends_fallbacks_for_haiku() -> None:
    recorder = Recorder(200, claude_message())
    await claude(recorder, model="claude-haiku-5-5").complete(CompletionRequest(prompt="hi"))
    assert "fallbacks" not in recorder.last_json
