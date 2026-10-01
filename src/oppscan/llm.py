"""LLM access: one JSON-returning call per task, validated against its schema and cached."""
from __future__ import annotations

import hashlib
import json
from typing import Protocol

import jsonschema

from oppscan import prompts
from oppscan.db import utcnow

# task -> (model, effort)
TASKS: dict[str, tuple[str, str]] = {
    "cluster": ("claude-sonnet-5-5", "medium"),
    "complaints": ("claude-sonnet-5-5", "low"),
    "brief": ("claude-opus-5-5", "high"),
    "summary": ("claude-opus-5-5", "medium"),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMOutputError(Exception):
    """The model's output could not be used (refusal, truncation, invalid JSON or schema)."""


class LLMClient(Protocol):
    def call(self, task: str, payload: dict) -> dict: ...


def validate(task: str, data: dict) -> dict:
    try:
        jsonschema.validate(data, prompts.SCHEMAS[task])
    except jsonschema.ValidationError as e:
        raise LLMOutputError(f"{task}: {e.message}") from e
    return data


class AnthropicLLM:
    def __init__(self, con, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self._con = con
        self._client = client

    def call(self, task: str, payload: dict) -> dict:
        model, effort = TASKS[task]
        system, user = prompts.render(task, payload)
        key = hashlib.sha256(json.dumps([model, effort, system, user]).encode()).hexdigest()
        row = self._con.execute("SELECT response FROM llm_cache WHERE cache_key = ?", [key]).fetchone()
        if row is not None:
            return json.loads(row[0])
        last_error: LLMOutputError | None = None
        for _ in range(2):
            try:
                data = validate(task, self._request(model, effort, system, user, prompts.SCHEMAS[task]))
            except LLMOutputError as e:
                last_error = e
                continue
            self._con.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?)",
                              [key, task, json.dumps(data), utcnow()])
            return data
        raise last_error

    def _request(self, model: str, effort: str, system: str, user: str, schema: dict) -> dict:
        response = self._client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMOutputError(f"stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMOutputError("response had no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"invalid JSON: {e}") from e
