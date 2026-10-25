import json

import pytest
from pydantic import BaseModel

from aiplay.providers.fake_provider import FakeProvider
from aiplay.structured import ExtractionError, extract, strict_schema
from aiplay.tasks import JobClassification, Resume


def test_strict_schema_inlines_refs_and_requires_every_property() -> None:
    schema = strict_schema(Resume)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    item = schema["properties"]["experience"]["items"]
    assert item["additionalProperties"] is False and set(item["required"]) == {"employer", "title", "start", "end"}
    assert "title" not in schema and "default" not in json.dumps(schema)
    # Enums survive inlining
    assert strict_schema(JobClassification)["properties"]["category"]["enum"][0] == "engineering"


class Point(BaseModel):
    x: int
    y: int


async def test_valid_output_needs_one_call() -> None:
    provider = FakeProvider(['{"x": 1, "y": 2}'])
    result = await extract(provider, Point, "give a point")
    assert result.value == Point(x=1, y=2)
    assert len(result.attempts) == 1
    assert provider.requests[0].json_schema == strict_schema(Point)


async def test_invalid_output_is_repaired_with_the_validation_error() -> None:
    provider = FakeProvider(['{"x": "one"}', '```json\n{"x": 1, "y": 2}\n```'])
    result = await extract(provider, Point, "give a point", repairs=1)
    assert result.value == Point(x=1, y=2)
    assert len(result.attempts) == 2
    roles = [role for role, _ in provider.requests[1].followups]
    assert roles == ["assistant", "user"]
    assert "y" in provider.requests[1].followups[1][1]  # the missing field is named in the repair prompt
    assert result.usage.input_tokens == sum(a.usage.input_tokens for a in result.attempts)


async def test_gives_up_after_repairs() -> None:
    provider = FakeProvider(["not json at all"])
    with pytest.raises(ExtractionError) as caught:
        await extract(provider, Point, "give a point", repairs=2)
    assert len(caught.value.attempts) == 3
