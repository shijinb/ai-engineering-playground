# AI Engineering Playground

[![CI](https://github.com/shijinb/ai-engineering-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/shijinb/ai-engineering-playground/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Patterns for putting LLMs into production code, built as a small library and CLI (`aiplay`). OpenAI, Google Gemini and Anthropic Claude sit behind one interface, so the same task can run on any of them, fall back between them, and be measured against a labelled dataset.

It covers the parts that matter after the demo works:

| Concern | What the code does |
|---|---|
| **One interface, three vendors** | `Provider` protocol with adapters built on each official SDK (`openai`, `google-genai`, `anthropic`). Vendor quirks (refusals, truncation, error codes, usage fields) map to one set of types. |
| **Typed structured output** | Pydantic model → strict JSON Schema → each vendor's native structured-output mode. The reply is validated; on failure the model is re-asked once with the validation errors. |
| **Retries** | Exponential backoff with jitter for rate limits, timeouts and 5xx. `Retry-After` is honoured. Client errors (bad request, bad schema) are never retried. |
| **Fallback** | Comma-separated providers form a chain: an outage or refusal on one moves to the next; request errors surface immediately instead of being hidden. |
| **Cost and latency** | Every call is logged as one JSON record: provider, model, tokens, latency, cost (from a dated price table, never guessed). |
| **Versioned prompts** | Prompts are files with `name`, `version` and declared variables. Rendering fails on a missing or unknown variable. |
| **Evaluation** | Run a task over a JSONL dataset per provider and get accuracy, failures, latency, tokens and cost as a Markdown table. |

## Quick start

```bash
git clone https://github.com/shijinb/ai-engineering-playground.git
cd ai-engineering-playground
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

aiplay eval jobs --offline          # runs the whole pipeline with a scripted provider: no keys, no cost
pytest -q
```

With keys (see [`.env.example`](.env.example)):

```bash
export ANTHROPIC_API_KEY=...
aiplay ask "Explain idempotency keys in two sentences"
aiplay resume data/samples/resume_backend.txt               # → validated JSON
aiplay job data/samples/job_data_engineer.txt --provider openai,anthropic   # OpenAI first, Claude if it fails
aiplay eval jobs --providers anthropic,openai,gemini --details
```

The output goes to stdout; one JSON telemetry line per model call goes to stderr:

```json
{"provider": "anthropic", "model": "claude-opus-5-5", "input_tokens": 412, "output_tokens": 96, "latency_ms": 2140.3, "cost_usd": 0.003568, "finish_reason": "end_turn", "request_id": "req_..."}
```

(The numbers above illustrate the format; they are not a measurement.)

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | Read by each vendor SDK | none |
| `AIPLAY_ANTHROPIC_MODEL` | Claude model | `claude-opus-5-5` |
| `AIPLAY_ANTHROPIC_EFFORT` | `low` … `max` | unset (model default) |
| `AIPLAY_ANTHROPIC_FALLBACKS` | `off` disables server-side refusal fallback | on |
| `AIPLAY_OPENAI_MODEL`, `AIPLAY_GEMINI_MODEL` | Model names | **required**, no default |

OpenAI and Gemini have no default model on purpose: model names and account access change often, and a silently wrong default is worse than a clear error.

## Layout

```
aiplay/
  types.py            CompletionRequest, Completion, Usage, error hierarchy
  providers/          anthropic_provider, openai_provider, gemini_provider, fake_provider
  reliability.py      RetryingProvider, FallbackProvider
  structured.py       strict_schema(), extract() with validation and one repair
  prompts.py          versioned prompt templates; prompts/*.prompt.md
  tasks.py            example tasks: resume extraction, job classification
  evaluation.py       dataset runner and report
  telemetry.py        cost from data/pricing.json, JSON call records
  cli.py              the aiplay command
data/
  eval/*.jsonl        small synthetic labelled datasets
  samples/            synthetic inputs for the CLI
  pricing.json        prices per 1M tokens, with source and date checked
```

## Design notes

**Strict schemas.** OpenAI strict mode requires every property to be listed in `required` and `additionalProperties: false` on every object; Claude and Gemini accept the same form. `strict_schema()` derives it from the Pydantic model, inlines `$ref`s and removes `title`/`default` keywords (but not fields named `title`: a test caught exactly that bug).

**Claude specifics.** Current Claude models reject sampling parameters, so none are sent. A safety decline arrives as HTTP 200 with `stop_reason: "refusal"`; by default the adapter opts in to server-side fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`), so a declined request can be served by another Claude model in the same call. The returned `model` field shows which model answered. Haiku models have no server-side fallback, so the parameter is not sent for them.

**SDK retries are off.** Each adapter creates its client with `max_retries=0`; retrying happens in one place (`RetryingProvider`), so attempts are counted and logged once.

**Costs.** `data/pricing.json` lists Claude prices (standard rates, checked 2026-10-25 against anthropic.com/pricing; Haiku 5.5 is the ≤100K-token-prompt rate). OpenAI and Gemini models are not listed because the model is your choice: add the rates you use, otherwise cost is reported as `null`, never estimated.

**Evaluation scoring.** Each expected field scores 0–1: exact match after normalising case and whitespace for strings and enums, F1 over the items for lists. A case's score is the mean of its fields. A failed call scores 0 and counts as a failure.

## How this is tested

37 tests, no network:

- **Adapter tests run through the real vendor SDKs** with a mocked HTTP transport. They check the exact request each SDK sends (endpoint, headers, body: schema format, system prompt, token limit, no sampling parameters) and how real response shapes map back (usage, refusals, truncation, 429/5xx/4xx).
- Reliability, structured extraction and repair, prompts, scoring, cost and the CLI are tested with a scripted `FakeProvider`.
- CI runs ruff, mypy (strict) and pytest on Python 3.11 and 3.12.

## Limitations

- **Not yet run against live APIs in CI.** The adapters are verified against the SDKs' request and response formats, not against live endpoints. Run `aiplay eval` with your own keys to measure real behaviour.
- **The datasets are tiny and synthetic** (8 job descriptions, 3 resumes). They show the evaluation workflow; they are not a benchmark, and no accuracy figures are claimed here.
- `--offline` replays the expected answers, so its 100% only proves the pipeline is wired correctly.
- No streaming, tool use or caching yet; requests are single-turn plus the repair follow-up.

## License

[MIT](LICENSE) © 2026 Shijin B Gopal
