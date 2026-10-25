"""Token, latency and cost accounting.

Prices live in data/pricing.json with their source and the date they were checked. A model with no
price entry is reported with cost `None` rather than a guess.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from aiplay.types import Completion

logger = logging.getLogger("aiplay.calls")
DEFAULT_PRICING = Path(__file__).resolve().parent.parent / "data" / "pricing.json"


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float


def load_prices(path: Path = DEFAULT_PRICING) -> dict[str, Price]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {model: Price(entry["input_per_million"], entry["output_per_million"]) for model, entry in data["models"].items()}


def cost_usd(completion: Completion, prices: dict[str, Price]) -> float | None:
    price = prices.get(completion.model)
    if price is None:
        return None
    return (completion.usage.input_tokens * price.input_per_million + completion.usage.output_tokens * price.output_per_million) / 1_000_000


@dataclass(frozen=True)
class CallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None
    finish_reason: str | None
    request_id: str | None


def record(completion: Completion, prices: dict[str, Price]) -> CallRecord:
    entry = CallRecord(
        provider=completion.provider,
        model=completion.model,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        latency_ms=round(completion.latency_ms, 1),
        cost_usd=cost_usd(completion, prices),
        finish_reason=completion.finish_reason,
        request_id=completion.request_id,
    )
    logger.info("llm_call %s", json.dumps(asdict(entry)))
    return entry
