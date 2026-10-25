"""Typed extraction: Pydantic model in, validated instance out.

1. The model's JSON Schema is sent through each provider's native structured-output feature.
2. The reply is validated with Pydantic, so a provider bug or a schema the provider only partly
   enforces never leaks invalid data into the caller.
3. On invalid output the model is shown its answer and the validation errors and asked to correct
   it (at most `repairs` times). Every attempt is returned for logging and cost accounting.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from aiplay.providers.base import Provider
from aiplay.types import Completion, CompletionRequest, Usage

T = TypeVar("T", bound=BaseModel)


class ExtractionError(Exception):
    def __init__(self, message: str, attempts: list[Completion]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass
class Extraction(Generic[T]):
    value: T
    attempts: list[Completion] = field(default_factory=list)

    @property
    def usage(self) -> Usage:
        return Usage(sum(a.usage.input_tokens for a in self.attempts), sum(a.usage.output_tokens for a in self.attempts))


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema in the strict form providers accept: refs inlined, every property required
    (optional fields become nullable), additionalProperties false, no defaults/titles."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any, *, field_map: bool = False) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            if field_map:  # keys of "properties" are field names: a field may be called "title"
                return {k: resolve(v) for k, v in node.items()}
            node = {k: resolve(v, field_map=k == "properties") for k, v in node.items() if k not in ("title", "default")}
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    result: dict[str, Any] = resolve(schema)
    return result


def _parse(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):  # tolerate a fenced block if a provider adds one
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(cleaned)


async def extract(
    provider: Provider,
    model: type[T],
    prompt: str,
    *,
    system: str | None = None,
    repairs: int = 1,
    max_output_tokens: int = 4096,
) -> Extraction[T]:
    schema = strict_schema(model)
    attempts: list[Completion] = []
    followups: tuple[tuple[str, str], ...] = ()

    for _ in range(repairs + 1):
        completion = await provider.complete(
            CompletionRequest(
                prompt=prompt,
                system=system,
                json_schema=schema,
                schema_name=model.__name__,
                max_output_tokens=max_output_tokens,
                followups=followups,
            )
        )
        attempts.append(completion)
        try:
            return Extraction(model.model_validate(_parse(completion.text)), attempts)
        except (json.JSONDecodeError, ValidationError) as exc:
            problem = str(exc) if isinstance(exc, ValidationError) else f"not valid JSON ({exc.msg})"
            followups += (
                ("assistant", completion.text),
                ("user", f"That output is invalid: {problem}\nReturn only corrected JSON matching the schema."),
            )

    raise ExtractionError(f"{provider.name} returned invalid output after {repairs + 1} attempts", attempts)
