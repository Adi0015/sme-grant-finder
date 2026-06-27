"""Thin, backend-swappable LLM wrapper for the generation layer (Day 4, Block A).

    generate(system, user, json_schema=None, model=None) -> dict | str

One entry point. The backend is chosen by config.LLM_BACKEND (a single constant):

  - "ollama"    : local Ollama HTTP server (no API key, no paid hosted service).
                  Structured output via Ollama's `format` = <json schema>.
  - "anthropic" : PAID hosted API (requires ANTHROPIC_API_KEY). Never selected
                  silently — only when LLM_BACKEND is set to it on purpose.

JSON handling: when json_schema is given we ask the backend for structured output
and json.loads the result. A parse failure is RETRIED ONCE with a corrective nudge;
a second failure returns a clear error object {"_error": ..., "_raw": ...} rather
than raising, so the caller (generate.py) can flag the call instead of crashing.
"""

from __future__ import annotations

import json

import requests

import config


class LLMError(Exception):
    pass


# ── Backend: Ollama (local) ─────────────────────────────────────────────────────
def _ollama_chat(system: str, user: str, json_schema: dict | None, model: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": config.LLM_TEMPERATURE},
    }
    if json_schema is not None:
        payload["format"] = json_schema      # Ollama structured output
    r = requests.post(
        f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=config.LLM_TIMEOUT
    )
    if r.status_code != 200:
        # Surface the server's own message (e.g. the cloud-subscription 403) verbatim.
        raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:300]}")
    return r.json().get("message", {}).get("content", "")


# ── Backend: Anthropic (paid hosted API) ────────────────────────────────────────
def _anthropic_chat(system: str, user: str, json_schema: dict | None, model: str) -> str:
    import os

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY not set (anthropic backend selected)")
    # Force JSON via a single tool when a schema is given (the reliable structured
    # path on the Messages API); otherwise plain text.
    body = {
        "model": model,
        "max_tokens": 2048,
        "temperature": config.LLM_TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if json_schema is not None:
        body["tools"] = [{
            "name": "emit",
            "description": "Emit the structured result.",
            "input_schema": json_schema,
        }]
        body["tool_choice"] = {"type": "tool", "name": "emit"}
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=config.LLM_TIMEOUT,
    )
    if r.status_code != 200:
        raise LLMError(f"anthropic HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if json_schema is not None:
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                return json.dumps(block["input"])
        raise LLMError("anthropic: no tool_use block returned")
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _dispatch(system: str, user: str, json_schema: dict | None, model: str | None) -> str:
    backend = config.LLM_BACKEND
    if backend == "ollama":
        return _ollama_chat(system, user, json_schema, model or config.OLLAMA_MODEL)
    if backend == "anthropic":
        return _anthropic_chat(system, user, json_schema, model or config.ANTHROPIC_MODEL)
    raise LLMError(f"unknown LLM_BACKEND: {backend!r}")


# ── Public entry point ──────────────────────────────────────────────────────────
def generate(system: str, user: str, json_schema: dict | None = None,
             model: str | None = None) -> dict | str:
    """Run one completion. With json_schema set, returns a parsed dict (retrying
    once on bad JSON, then an {"_error", "_raw"} object). Without it, returns text."""
    raw = _dispatch(system, user, json_schema, model)
    if json_schema is None:
        return raw

    parsed = _try_parse(raw)
    if parsed is not None:
        return parsed

    # Retry once with an explicit corrective nudge.
    nudge = (user + "\n\nYour previous reply was not valid JSON. Reply with ONLY a "
             "single valid JSON object matching the schema, no prose, no markdown.")
    raw2 = _dispatch(system, nudge, json_schema, model)
    parsed = _try_parse(raw2)
    if parsed is not None:
        return parsed
    return {"_error": "LLM did not return valid JSON after one retry", "_raw": raw2[:1000]}


def _try_parse(raw: str) -> dict | None:
    """Parse JSON, tolerating ```json fences or leading/trailing prose."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost {...} span.
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


if __name__ == "__main__":
    # Tiny smoke test against the configured backend.
    import sys
    schema = {"type": "object",
              "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    try:
        out = generate("Reply only with JSON.",
                       "What is the capital of France? Return {answer}.",
                       json_schema=schema)
        print("backend:", config.LLM_BACKEND, "| model:",
              config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else config.ANTHROPIC_MODEL)
        print("result:", out)
    except LLMError as e:
        print("LLMError:", e, file=sys.stderr)
        sys.exit(1)
