"""aiplay command line.

aiplay ask --provider anthropic "Explain idempotency keys in two sentences"
aiplay resume data/samples/resume_backend.txt --provider anthropic
aiplay job data/samples/job_data_engineer.txt --provider openai
aiplay eval jobs --providers anthropic,openai,gemini
aiplay eval jobs --offline          # pipeline check with a scripted provider, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aiplay import evaluation, tasks
from aiplay.config import PROVIDERS, build
from aiplay.providers.base import Provider
from aiplay.providers.fake_provider import FakeProvider
from aiplay.reliability import FallbackProvider
from aiplay.structured import ExtractionError
from aiplay.telemetry import load_prices, record
from aiplay.types import CompletionRequest, ProviderError

DATA = Path(__file__).resolve().parent.parent / "data"
SUITES: dict[str, tuple[evaluation.Task, Path]] = {
    "jobs": (tasks.classify_job, DATA / "eval" / "job_classification.jsonl"),
    "resumes": (tasks.extract_resume, DATA / "eval" / "resume_extraction.jsonl"),
}


def _provider(args: argparse.Namespace) -> Provider:
    names = [n.strip() for n in args.provider.split(",")]
    built = [build(n, model=args.model if len(names) == 1 else None) for n in names]
    return built[0] if len(built) == 1 else FallbackProvider(built)


async def _ask(args: argparse.Namespace) -> int:
    completion = await _provider(args).complete(CompletionRequest(prompt=args.prompt, system=args.system))
    print(completion.text)
    print(json.dumps(asdict(record(completion, load_prices()))), file=sys.stderr)
    return 0


async def _extract(args: argparse.Namespace, task: evaluation.Task) -> int:
    text = await asyncio.to_thread(Path(args.file).read_text, encoding="utf-8")
    result = await task(_provider(args), text)
    print(result.value.model_dump_json(indent=2))
    prices = load_prices()
    for attempt in result.attempts:
        print(json.dumps(asdict(record(attempt, prices))), file=sys.stderr)
    return 0


def _offline_provider(cases: list[dict[str, Any]]) -> FakeProvider:
    # Replays each case's expected answer: exercises the pipeline and scoring without a model.
    return FakeProvider([json.dumps(_complete_expected(c["expected"])) for c in cases] or ["{}"], name="offline", model="scripted")


def _complete_expected(expected: dict[str, Any]) -> dict[str, Any]:
    defaults = {"remote": False, "key_skills": [], "full_name": "", "experience": []}
    return {**defaults, **expected}


async def _eval(args: argparse.Namespace) -> int:
    task, path = SUITES[args.suite]
    cases = evaluation.load_cases(Path(args.dataset) if args.dataset else path)
    prices = load_prices()
    providers: list[Provider] = [_offline_provider(cases)] if args.offline else [build(n.strip()) for n in args.providers.split(",")]
    reports = [await evaluation.run(task, provider, cases, prices) for provider in providers]
    print(evaluation.markdown(reports))
    if args.details:
        for report in reports:
            for case in report.cases:
                print(f"{report.provider}\t{case.case_id}\t{case.score:.2f}\t{case.error or json.dumps(case.field_scores)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiplay", description="Provider-agnostic LLM playground")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def provider_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", default="anthropic", help=f"one of {', '.join(PROVIDERS)}; comma-separate for fallback order")
        p.add_argument("--model", help="override the model (single provider only)")

    ask = sub.add_parser("ask", help="send one prompt")
    ask.add_argument("prompt")
    ask.add_argument("--system")
    provider_args(ask)

    for name, help_text in (("resume", "extract structured data from a resume text file"), ("job", "classify a job description file")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("file")
        provider_args(p)

    ev = sub.add_parser("eval", help="score providers on a labelled dataset")
    ev.add_argument("suite", choices=sorted(SUITES))
    ev.add_argument("--providers", default="anthropic")
    ev.add_argument("--dataset", help="JSONL file (default: the suite's bundled dataset)")
    ev.add_argument("--offline", action="store_true", help="scripted provider; checks the pipeline, makes no API calls")
    ev.add_argument("--details", action="store_true", help="per-case scores")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    commands = {
        "ask": lambda: _ask(args),
        "resume": lambda: _extract(args, tasks.extract_resume),
        "job": lambda: _extract(args, tasks.classify_job),
        "eval": lambda: _eval(args),
    }
    try:
        return asyncio.run(commands[args.command]())
    except (ProviderError, ExtractionError, ValueError) as exc:
        print(f"aiplay: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
