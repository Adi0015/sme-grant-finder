"""Generation layer: one matched call -> grounded fit + DRAFT eligibility + citations
(Day 4, Block B).

The model sees ONLY: the SME profile + the matched call's own text (the retrieved
scope chunk that matched, plus that call's stated `conditions` from calls.json) +
optionally 1-2 similar funded projects as evidence. Everything carries a source_id.

TWO HARD RULES (enforced in the prompt AND, more importantly, in code below):
  1. CITATIONS ARE MANDATORY. Every fit/eligibility claim names the source_id it came
     from and quotes a verbatim snippet. We DROP any claim whose source_id we did not
     pass, or whose snippet is not actually present in that source's text.
  2. ELIGIBILITY IS A DRAFT, NEVER A VERDICT. Only conditions literally stated in the
     call text. Anything absent -> missing_info ("check the call documents"). The
     model is forbidden from saying "you are / are not eligible"; we also strip such
     phrasing from the summary in code.

`source_id` is CALL-LEVEL: every chunk of a call shares it, so grounding is checked
at the level of "did this text come from a source I handed the model", and snippets
are verified against the exact text passed for that source_id.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import config
import llm

# ── Source material loader ──────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _calls_by_id() -> dict[str, dict]:
    """source_id -> full call record (for the `conditions` text not stored in Chroma)."""
    records = json.loads(config.CALLS_JSON.read_text(encoding="utf-8"))
    return {r["source_id"]: r for r in records}


# ── Grounding helpers (the anti-hallucination check, in code) ────────────────────
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()


def _grounded(snippet: str, haystack: str) -> bool:
    """Is `snippet` actually present in `haystack`? Exact normalized substring is the
    happy path; for longer snippets we accept a high word-overlap (the model often
    lightly reformats whitespace/casing). Short snippets MUST be exact substrings —
    no fuzzy pass for 1-3 word fragments (too easy to match by chance)."""
    if not snippet or not haystack:
        return False
    ns, nh = _norm(snippet), _norm(haystack)
    if ns in nh:
        return True
    words = _WORD.findall(ns)
    if len(words) < 5:
        return False                       # too short to trust fuzzily
    hay_words = set(_WORD.findall(nh))
    hit = sum(1 for w in words if w in hay_words)
    return hit / len(words) >= 0.80


_VERDICT_RE = re.compile(
    r"\b(you|your (?:company|organisation|organization|sme))\s+(are|is|do|does)?\s*"
    r"(not\s+)?(eligible|qualif\w*|ineligible)\b\.?", re.I)


def _strip_verdict(text: str) -> tuple[str, bool]:
    """Neutralize verdict phrasing ('you are eligible' / 'you do not qualify').
    Returns (clean_text, was_modified)."""
    if not text:
        return text, False
    if _VERDICT_RE.search(text):
        clean = _VERDICT_RE.sub("the call states eligibility conditions (see eligibility, to verify)", text)
        return clean, True
    return text, False


# ── Prompt ──────────────────────────────────────────────────────────────────────
SYSTEM = """You are an assistant that helps a European SME understand whether an OPEN EU funding call fits them. You are a DECISION-SUPPORT tool, not a legal authority.

ABSOLUTE RULES — follow exactly:
1. CITE EVERYTHING. Every fit reason and every eligibility item MUST include the source_id it came from and a `snippet` that is COPIED VERBATIM from the provided source text. If you cannot find supporting text in what is provided, DO NOT make the claim.
2. ELIGIBILITY IS A DRAFT, NOT A VERDICT. Only list conditions that are LITERALLY STATED in the provided call text. Never invent conditions (no made-up budget, country, TRL, deadline, or consortium rules). If the provided text states no eligibility conditions, return "eligibility": [] and list what must be checked under "missing_info".
3. NEVER say the applicant "is eligible" or "is not eligible". Never say they "qualify". Frame everything as: the call STATES X; based on the SME profile this APPEARS to match / is unclear / appears not to match / is not stated.
4. Use ONLY the provided SME profile and source texts. Do not use outside knowledge about these programmes.

Return ONLY a single JSON object, no prose, no markdown."""

# Schema handed to the backend for structured output + used to coerce the result.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "fit_summary": {"type": "string"},
        "fit_reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source_id": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["claim", "source_id", "snippet"],
            },
        },
        "eligibility": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "stated_in_call": {"type": "boolean"},
                    "applicant_appears_to_meet": {
                        "type": "string",
                        "enum": ["yes", "unclear", "no", "not_stated"],
                    },
                    "source_id": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["condition", "stated_in_call",
                             "applicant_appears_to_meet", "source_id", "snippet"],
            },
        },
        "missing_info": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["fit_summary", "fit_reasons", "eligibility", "missing_info"],
}


def _build_user(profile, call: dict, scope_text: str, conditions_text: str,
                evidence: list[dict]) -> str:
    p = profile
    prof_lines = [f"- description: {p.description}"]
    for label, val in (("country", p.country), ("org_type", p.org_type),
                       ("budget_eur", p.budget), ("trl", p.trl)):
        if val is not None:
            prof_lines.append(f"- {label}: {val}")
    if p.keywords:
        prof_lines.append(f"- keywords: {', '.join(p.keywords)}")

    parts = [
        "SME PROFILE:",
        "\n".join(prof_lines),
        "",
        f"CALL  [source_id: {call['source_id']}]  (cite this id for call-based claims)",
        f"title: {call.get('title', '')}",
        "",
        "CALL SCOPE (what this call funds):",
        scope_text or "(none provided)",
        "",
        "CALL STATED CONDITIONS (the ONLY basis for eligibility items; may be empty):",
        conditions_text or "(no conditions text provided — return eligibility: [] and put checks in missing_info)",
    ]
    if evidence:
        parts += ["", "SIMILAR FUNDED PROJECTS (context only; cite their source_id if used for a fit reason):"]
        for e in evidence:
            parts.append(f"  [source_id: {e['source_id']}] {e.get('title','')}: {e.get('chunk_text','')[:400]}")
    parts += [
        "",
        "Produce a JSON object with keys: fit_summary (2-4 sentences), "
        "fit_reasons [{claim, source_id, snippet}], "
        "eligibility [{condition, stated_in_call, applicant_appears_to_meet "
        "(yes|unclear|no|not_stated), source_id, snippet}], missing_info [strings]. "
        "Every snippet MUST be copied verbatim from the text above.",
    ]
    return "\n".join(parts)


# ── Per-call generation + grounding enforcement ──────────────────────────────────
def generate_for_call(profile, call: dict, evidence: list[dict] | None = None,
                      model: str | None = None) -> dict:
    """Generate the grounded fit+eligibility block for ONE matched call dict (as
    returned by retrieve_calls). Returns a JSON-serializable object; never raises on
    LLM/parse failure (returns a flagged object instead)."""
    evidence = evidence or []
    sid = call["source_id"]
    record = _calls_by_id().get(sid, {})
    scope_text = call.get("best_chunk_text", "") or record.get("description", "")
    conditions_text = (record.get("conditions") or "")[: config.MAX_CONDITIONS_CHARS]

    # The text we actually handed the model, keyed by source_id — the grounding oracle.
    allowed: dict[str, str] = {sid: f"{call.get('title','')}\n{scope_text}\n{conditions_text}"}
    urls: dict[str, str] = {sid: call.get("source_url", "")}
    for e in evidence:
        allowed[e["source_id"]] = f"{e.get('title','')}\n{e.get('chunk_text','')}"
        urls[e["source_id"]] = e.get("source_url", "")

    user = _build_user(profile, call, scope_text, conditions_text, evidence)
    out = llm.generate(SYSTEM, user, json_schema=JSON_SCHEMA, model=model)

    base = _call_facts(call)          # authoritative metadata from retrieval, not the LLM
    if not isinstance(out, dict) or "_error" in out:
        base.update({
            "fit_summary": "", "fit_reasons": [], "eligibility": [],
            "missing_info": ["LLM generation failed — re-run."],
            "_grounding": {"llm_error": out.get("_error") if isinstance(out, dict) else "non-dict",
                           "dropped_fit": 0, "dropped_eligibility": 0, "warnings": []},
        })
        return base

    warnings: list[str] = []
    fit_summary, modified = _strip_verdict(str(out.get("fit_summary", "")).strip())
    if modified:
        warnings.append("verdict phrasing stripped from fit_summary")

    fit_reasons, dropped_fit = _filter_citations(out.get("fit_reasons"), allowed, urls, "claim")
    eligibility, dropped_elig = _filter_eligibility(out.get("eligibility"), allowed, urls)

    missing = [str(m).strip() for m in (out.get("missing_info") or []) if str(m).strip()]
    if not eligibility and "check the official call documents for eligibility conditions" not in missing:
        missing.append("check the official call documents for eligibility conditions")

    base.update({
        "fit_summary": fit_summary,
        "fit_reasons": fit_reasons,
        "eligibility": eligibility,
        "missing_info": missing,
        "_grounding": {
            "dropped_fit": dropped_fit,
            "dropped_eligibility": dropped_elig,
            "warnings": warnings,
        },
    })
    return base


def _call_facts(call: dict) -> dict:
    """Metadata taken straight from retrieval (authoritative) so the LLM can never
    hallucinate a deadline / programme / URL."""
    return {
        "source_id": call["source_id"],
        "title": call.get("title", ""),
        "source_url": call.get("source_url", ""),
        "score": call.get("best_score", call.get("score")),
        "deadline": call.get("deadline", ""),
        "programme": call.get("programme", ""),
        "type_of_action": call.get("type_of_action", ""),
    }


def _filter_citations(items, allowed: dict[str, str], urls: dict[str, str],
                      text_key: str) -> tuple[list, int]:
    """Keep only claims whose source_id was provided AND whose snippet is grounded in
    that source's text. Returns (kept, n_dropped)."""
    kept, dropped = [], 0
    for it in (items or []):
        if not isinstance(it, dict):
            dropped += 1
            continue
        sid = str(it.get("source_id", "")).strip()
        snippet = str(it.get("snippet", "")).strip()
        if sid not in allowed or not _grounded(snippet, allowed[sid]):
            dropped += 1
            continue
        kept.append({text_key: str(it.get(text_key, "")).strip(),
                     "source_id": sid, "snippet": snippet,
                     "source_url": urls.get(sid, "")})
    return kept, dropped


_VALID_MEET = {"yes", "unclear", "no", "not_stated"}


def _filter_eligibility(items, allowed: dict[str, str], urls: dict[str, str]) -> tuple[list, int]:
    kept, dropped = [], 0
    for it in (items or []):
        if not isinstance(it, dict):
            dropped += 1
            continue
        sid = str(it.get("source_id", "")).strip()
        snippet = str(it.get("snippet", "")).strip()
        # An eligibility item is only legitimate if it is stated in the call AND we can
        # verify the snippet in the call's own text. Otherwise it's an invented condition.
        if sid not in allowed or not _grounded(snippet, allowed[sid]):
            dropped += 1
            continue
        meet = str(it.get("applicant_appears_to_meet", "not_stated")).strip().lower()
        if meet not in _VALID_MEET:
            meet = "not_stated"
        kept.append({
            "condition": str(it.get("condition", "")).strip(),
            "stated_in_call": bool(it.get("stated_in_call", True)),
            "applicant_appears_to_meet": meet,
            "source_id": sid,
            "source_url": urls.get(sid, ""),
            "snippet": snippet,
        })
    return kept, dropped
