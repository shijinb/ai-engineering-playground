"""Compare providers (or prompt versions) on a labelled dataset.

Each case in a JSONL file has an `input` and the `expected` fields. Scoring is per field:
- strings compare case- and whitespace-insensitively,
- lists compare as sets (score = F1 between expected and predicted items),
- other values compare for equality.
A case's score is the mean of its field scores; failures (errors, invalid output) score 0.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aiplay.providers.base import Provider
from aiplay.structured import Extraction, ExtractionError
from aiplay.telemetry import Price, cost_usd
from aiplay.types import ProviderError

Task = Callable[[Provider, str], Awaitable[Extraction[Any]]]


def _norm(value: Any) -> Any:
    return " ".join(value.lower().split()) if isinstance(value, str) else value


def field_score(expected: Any, predicted: Any) -> float:
    if isinstance(expected, list):
        want = {_norm(v) for v in expected}
        got = {_norm(v) for v in (predicted or [])}
        if not want and not got:
            return 1.0
        if not want or not got:
            return 0.0
        overlap = len(want & got)
        precision, recall = overlap / len(got), overlap / len(want)
        return 0.0 if overlap == 0 else 2 * precision * recall / (precision + recall)
    return 1.0 if _norm(expected) == _norm(predicted) else 0.0


def score_case(expected: dict[str, Any], predicted: BaseModel) -> dict[str, float]:
    data = predicted.model_dump(mode="json")
    return {name: field_score(value, data.get(name)) for name, value in expected.items()}


@dataclass
class CaseResult:
    case_id: str
    score: float
    field_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    error: str | None = None


@dataclass
class ProviderReport:
    provider: str
    model: str
    cases: list[CaseResult]

    @property
    def accuracy(self) -> float:
        return statistics.mean(c.score for c in self.cases) if self.cases else 0.0

    @property
    def failures(self) -> int:
        return sum(1 for c in self.cases if c.error)

    @property
    def mean_latency_ms(self) -> float:
        done = [c.latency_ms for c in self.cases if not c.error]
        return statistics.mean(done) if done else 0.0

    @property
    def total_cost(self) -> float | None:
        costs = [c.cost for c in self.cases]
        return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run(task: Task, provider: Provider, cases: Sequence[dict[str, Any]], prices: dict[str, Price]) -> ProviderReport:
    results: list[CaseResult] = []
    model = provider.model
    for case in cases:
        started = time.perf_counter()
        try:
            extraction = await task(provider, case["input"])
        except (ProviderError, ExtractionError) as exc:
            results.append(CaseResult(case["id"], 0.0, latency_ms=(time.perf_counter() - started) * 1000, error=str(exc)))
            continue
        fields = score_case(case["expected"], extraction.value)
        costs = [cost_usd(a, prices) for a in extraction.attempts]
        model = extraction.attempts[-1].model
        results.append(
            CaseResult(
                case["id"],
                statistics.mean(fields.values()),
                fields,
                latency_ms=sum(a.latency_ms for a in extraction.attempts),
                input_tokens=extraction.usage.input_tokens,
                output_tokens=extraction.usage.output_tokens,
                cost=None if any(c is None for c in costs) else sum(c for c in costs if c is not None),
            )
        )
    return ProviderReport(provider.name, model, results)


def markdown(reports: Sequence[ProviderReport]) -> str:
    lines = [
        "| Provider | Model | Accuracy | Failures | Mean latency | Tokens in / out | Cost (USD) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        tokens_in = sum(c.input_tokens for c in r.cases)
        tokens_out = sum(c.output_tokens for c in r.cases)
        cost = "n/a (no price)" if r.total_cost is None else f"{r.total_cost:.4f}"
        lines.append(
            f"| {r.provider} | {r.model} | {r.accuracy:.1%} | {r.failures}/{len(r.cases)} | "
            f"{r.mean_latency_ms:.0f} ms | {tokens_in:,} / {tokens_out:,} | {cost} |"
        )
    return "\n".join(lines)
