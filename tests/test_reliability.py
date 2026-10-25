import pytest

from aiplay.providers.fake_provider import FakeProvider
from aiplay.reliability import FallbackProvider, RetryingProvider, RetryPolicy
from aiplay.types import CompletionRequest, ProviderError, RateLimitedError, RefusedError, TransientError

REQ = CompletionRequest(prompt="hi")


def retrying(provider: FakeProvider, sleeps: list[float], attempts: int = 3) -> RetryingProvider:
    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return RetryingProvider(provider, RetryPolicy(attempts=attempts, base_delay=1.0, jitter=0.0), sleep=sleep)


async def test_retries_transient_errors_with_backoff() -> None:
    sleeps: list[float] = []
    provider = FakeProvider([TransientError("fake", "503"), TransientError("fake", "503"), "ok"])
    assert (await retrying(provider, sleeps).complete(REQ)).text == "ok"
    assert sleeps == [1.0, 2.0]


async def test_retry_after_is_honoured() -> None:
    sleeps: list[float] = []
    provider = FakeProvider([RateLimitedError("fake", "429", retry_after=7.0), "ok"])
    await retrying(provider, sleeps).complete(REQ)
    assert sleeps == [7.0]


async def test_non_retryable_errors_are_raised_immediately() -> None:
    sleeps: list[float] = []
    provider = FakeProvider([ProviderError("fake", "400 bad request"), "ok"])
    with pytest.raises(ProviderError):
        await retrying(provider, sleeps).complete(REQ)
    assert sleeps == [] and len(provider.requests) == 1


async def test_gives_up_after_attempts() -> None:
    provider = FakeProvider([TransientError("fake", "503")])
    with pytest.raises(TransientError):
        await retrying(provider, [], attempts=2).complete(REQ)
    assert len(provider.requests) == 2


@pytest.mark.parametrize("first_error", [TransientError("a", "down"), RefusedError("a", "declined")])
async def test_fallback_moves_on_after_outage_or_refusal(first_error: ProviderError) -> None:
    first = FakeProvider([first_error], name="a")
    second = FakeProvider(["from b"], name="b")
    completion = await FallbackProvider([first, second]).complete(REQ)
    assert completion.text == "from b" and completion.provider == "b"


async def test_fallback_does_not_hide_request_errors() -> None:
    first = FakeProvider([ProviderError("a", "invalid schema")], name="a")
    second = FakeProvider(["unused"], name="b")
    with pytest.raises(ProviderError, match="invalid schema"):
        await FallbackProvider([first, second]).complete(REQ)
    assert second.requests == []
