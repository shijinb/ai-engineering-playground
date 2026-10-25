import json
from pathlib import Path

import pytest

from aiplay import evaluation, prompts
from aiplay.cli import main
from aiplay.providers.fake_provider import FakeProvider
from aiplay.tasks import classify_job
from aiplay.telemetry import Price, cost_usd, load_prices
from aiplay.types import Completion, TransientError, Usage


def test_bundled_prompts_load_and_render() -> None:
    template = prompts.load("resume_extract")
    assert template.id == "resume_extract@v1"
    assert template.system and "never guess" in template.system
    assert "Jane" in template.render(resume_text="Jane Doe, engineer")


def test_render_rejects_missing_or_extra_variables() -> None:
    template = prompts.load("job_classify")
    with pytest.raises(ValueError, match="missing"):
        template.render()
    with pytest.raises(ValueError, match="unexpected"):
        template.render(job_description="x", salary="y")


def test_placeholders_must_match_declared_variables() -> None:
    with pytest.raises(ValueError, match="do not match"):
        prompts.parse("---\nname: t\nversion: 1\nvariables: a\n---\nHello ${b}\n")


def test_field_score() -> None:
    assert evaluation.field_score("Senior", " senior ") == 1.0
    assert evaluation.field_score(["Python", "SQL"], ["python", "sql"]) == 1.0
    assert evaluation.field_score(["Python", "SQL"], ["python"]) == pytest.approx(2 / 3)
    assert evaluation.field_score([], []) == 1.0
    assert evaluation.field_score(True, False) == 0.0


async def test_eval_run_scores_cases_and_counts_failures() -> None:
    cases = evaluation.load_cases(Path("data/eval/job_classification.jsonl"))[:2]
    good = json.dumps({"category": "engineering", "seniority": "senior", "remote": True, "key_skills": ["Python"]})
    provider = FakeProvider([good, TransientError("fake", "down")], model="claude-opus-5-5")
    report = await evaluation.run(classify_job, provider, cases, {"claude-opus-5-5": Price(4.0, 20.0)})
    assert report.cases[0].score == 1.0
    assert report.cases[1].error and report.cases[1].score == 0.0
    assert report.accuracy == 0.5 and report.failures == 1
    assert report.cases[0].cost is not None
    assert "| fake | claude-opus-5-5 | 50.0% | 1/2 |" in evaluation.markdown([report])


def test_cost_is_none_for_unpriced_models() -> None:
    completion = Completion("x", "p", "unknown-model", Usage(1_000_000, 1_000_000))
    assert cost_usd(completion, load_prices()) is None
    priced = Completion("x", "anthropic", "claude-opus-5-5", Usage(1_000_000, 500_000))
    assert cost_usd(priced, load_prices()) == pytest.approx(4.0 + 10.0)


def test_cli_offline_eval(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "jobs", "--offline"]) == 0
    assert "| offline | scripted | 100.0% | 0/8 |" in capsys.readouterr().out


def test_cli_reports_missing_model_configuration(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIPLAY_OPENAI_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert main(["ask", "hi", "--provider", "openai"]) == 1
    assert "AIPLAY_OPENAI_MODEL" in capsys.readouterr().err
