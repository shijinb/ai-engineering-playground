"""Versioned prompt templates stored as files.

    ---
    name: resume_extract
    version: 2
    variables: resume_text
    ---
    Extract ... from this resume:
    ${resume_text}

Rendering fails loudly on a missing or unexpected variable, so a renamed placeholder can never
silently send a literal "${resume_text}" to a model.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    variables: frozenset[str]
    system: str | None
    body: str

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, **values: str) -> str:
        missing = self.variables - values.keys()
        unexpected = values.keys() - self.variables
        if missing or unexpected:
            raise ValueError(f"{self.id}: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
        return string.Template(self.body).substitute(values)


def parse(text: str) -> PromptTemplate:
    if not text.startswith("---\n"):
        raise ValueError("prompt file must start with a --- front matter block")
    header, body = text[4:].split("\n---\n", 1)
    meta = dict(line.split(":", 1) for line in header.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    system = None
    if "\n=== user ===\n" in body:  # optional split between system and user parts
        system, body = body.split("\n=== user ===\n", 1)
        system = system.removeprefix("=== system ===\n").strip()
    variables = frozenset(v.strip() for v in meta.get("variables", "").split(",") if v.strip())
    template = PromptTemplate(meta["name"], int(meta["version"]), variables, system, body.strip() + "\n")
    declared = {
        m.group("named") or m.group("braced")
        for m in string.Template.pattern.finditer(template.body)
        if m.group("named") or m.group("braced")
    }
    if declared != set(variables):
        raise ValueError(f"{template.id}: placeholders {sorted(declared)} do not match variables {sorted(variables)}")
    return template


def load(name: str, directory: Path = PROMPT_DIR) -> PromptTemplate:
    return parse((directory / f"{name}.prompt.md").read_text(encoding="utf-8"))
