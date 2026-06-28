"""Faithfulness / grounding evaluation (Day 5, Block C).

Two layers:
  1. CODE grounding (carried from Day 4): % of generated fit/eligibility claims whose
     snippet is an exact substring of the call text we passed the model. With Day 4's
     re-derivation this should be ~100% by construction — it's a regression guard.
  2. SEMANTIC faithfulness (LLM-as-judge, src/llm.py): for each claim, ask the judge
     whether the CITED SNIPPET actually supports the CLAIM. Report % yes / partial / no.

Emits data/gold/faithfulness_sample.csv for HUMAN spot-checking (~15 rows by hand) — the
judge shares the qwen backend with the generator, so its number is a sanity signal, not
ground truth. The human spot-check is what the report should lean on.

Does NOT need the gold relevance labels (it judges claim<->snippet, not relevance), so it
can run before labeling. Generation is cached to data/gold/faith_generated.json so reruns
are cheap and deterministic.

    python src/eval_faithfulness.py            # sample profiles, generate (cached), judge
"""

from __future__ import annotations

import csv
import json
import re

import config
import llm
from generate import _calls_by_id
from pipeline import run
from schema import SMEProfile

GOLD = config.DATA / "gold"
PROFILES = GOLD / "profiles.json"
GENERATED = GOLD / "faith_generated.json"
SAMPLE_CSV = GOLD / "faithfulness_sample.csv"
OUT = GOLD / "metrics_faithfulness.json"

SAMPLE_N = 4          # profiles sampled for faithfulness (bounded LLM cost)
TOP_N = 3             # calls generated per sampled profile

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()


def _source_text(source_id: str) -> str:
    rec = _calls_by_id().get(source_id, {})
    return f"{rec.get('title','')} {rec.get('description','')} {rec.get('conditions','')}"


# ── Generation (cached) ──────────────────────────────────────────────────────────
def _load_profiles_sample() -> list[tuple[str, SMEProfile]]:
    raw = json.loads(PROFILES.read_text(encoding="utf-8"))[:SAMPLE_N]
    return [(r["id"], SMEProfile(**{k: v for k, v in r.items() if k != "id"})) for r in raw]


def _generate() -> list[dict]:
    if GENERATED.exists():
        return json.loads(GENERATED.read_text(encoding="utf-8"))
    out = []
    for pid, profile in _load_profiles_sample():
        print(f"  generating {pid} (top_n={TOP_N}) ...")
        result = run(profile, top_n=TOP_N)
        out.append({"profile_id": pid, "shortlist": result["shortlist"]})
    GENERATED.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ── Claim extraction ─────────────────────────────────────────────────────────────
def _claims(generated: list[dict]) -> list[dict]:
    claims = []
    for prof in generated:
        pid = prof["profile_id"]
        for call in prof["shortlist"]:
            sid = call["source_id"]
            for fr in call.get("fit_reasons", []):
                claims.append({"profile_id": pid, "source_id": sid, "type": "fit",
                               "claim": fr.get("claim", ""), "snippet": fr.get("snippet", "")})
            for e in call.get("eligibility", []):
                claims.append({"profile_id": pid, "source_id": sid, "type": "eligibility",
                               "claim": e.get("condition", ""), "snippet": e.get("snippet", "")})
    return claims


# ── Layer 2: LLM-as-judge ────────────────────────────────────────────────────────
_JUDGE_SYS = ("You are a strict faithfulness judge for a grant-finder. Given a CLAIM and a "
              "SNIPPET quoted from an EU funding call, decide whether the snippet SUPPORTS "
              "the claim. 'yes' = the snippet directly supports it; 'partial' = related but "
              "incomplete; 'no' = unsupported or contradicted. Judge only from the snippet. "
              "Return JSON only.")
_JUDGE_SCHEMA = {"type": "object",
                 "properties": {"verdict": {"type": "string", "enum": ["yes", "partial", "no"]},
                                "reason": {"type": "string"}},
                 "required": ["verdict", "reason"]}


def _judge(claim: dict) -> dict:
    user = (f"CLAIM: {claim['claim']}\n\nSNIPPET (quoted from the call): {claim['snippet']}\n\n"
            "Does the snippet support the claim? Return {verdict, reason}.")
    out = llm.generate(_JUDGE_SYS, user, json_schema=_JUDGE_SCHEMA)
    if isinstance(out, dict) and out.get("verdict") in ("yes", "partial", "no"):
        return out
    return {"verdict": "no", "reason": "judge parse error"}


# ── Main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    generated = _generate()
    claims = _claims(generated)
    if not claims:
        print("No claims generated — nothing to evaluate.")
        return

    # Layer 1: code grounding (exact substring of passed source).
    for c in claims:
        c["code_grounded"] = _norm(c["snippet"]) in _norm(_source_text(c["source_id"]))
    n = len(claims)
    code_pct = sum(c["code_grounded"] for c in claims) / n

    # Layer 2: LLM judge.
    print(f"Judging {n} claims with {config.LLM_BACKEND}:{config.OLLAMA_MODEL} ...")
    verdicts = {"yes": 0, "partial": 0, "no": 0}
    for i, c in enumerate(claims, 1):
        j = _judge(c)
        c["judge_verdict"] = j["verdict"]
        c["judge_reason"] = j.get("reason", "")
        verdicts[j["verdict"]] += 1
        if i % 5 == 0:
            print(f"  judged {i}/{n}")

    supported_pct = verdicts["yes"] / n
    supported_or_partial = (verdicts["yes"] + verdicts["partial"]) / n

    with SAMPLE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["profile_id", "source_id", "type", "claim",
                                          "snippet", "code_grounded", "judge_verdict",
                                          "judge_reason"])
        w.writeheader()
        w.writerows(claims)

    report = {
        "n_claims": n,
        "code_grounding_pct": round(code_pct, 4),
        "judge_supported_pct": round(supported_pct, 4),
        "judge_supported_or_partial_pct": round(supported_or_partial, 4),
        "judge_verdicts": verdicts,
        "backend": f"{config.LLM_BACKEND}:{config.OLLAMA_MODEL}",
        "sample": {"profiles": SAMPLE_N, "top_n": TOP_N},
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== FAITHFULNESS ===")
    print(f"  claims evaluated      : {n}")
    print(f"  code grounding        : {code_pct:.1%}  (snippet is exact substring of source)")
    print(f"  LLM judge supported   : {supported_pct:.1%}  (yes)")
    print(f"  LLM judge yes+partial : {supported_or_partial:.1%}")
    print(f"  verdict breakdown     : {verdicts}")
    print(f"\n  Spot-check by hand: {SAMPLE_CSV.relative_to(config.ROOT)} (review ~15 rows).")
    print(f"  Wrote {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
