"""CLI demo for the full Day-4 pipeline (Block D).

Runs run() on 2 example SME profiles and pretty-prints each call: title, deadline,
fit_summary, eligibility lines with their [source_url] + appears-to-meet, and
missing_info. Use --json for the raw assembled payload.

    python src/run_pipeline.py                 # 2 example profiles, human-readable
    python src/run_pipeline.py --json          # raw JSON
    python src/run_pipeline.py --top-n 3       # fewer calls per profile
"""

from __future__ import annotations

import argparse
import json

from pipeline import run
from schema import SMEProfile

EXAMPLES = [
    SMEProfile(
        description=(
            "A German SME developing AI-based predictive maintenance software for "
            "manufacturing lines. We use machine learning on sensor data to forecast "
            "equipment failures and reduce downtime."
        ),
        country="DE", org_type="SME", trl=6,
        keywords=["predictive maintenance", "machine learning", "industry 4.0", "sensors"],
    ),
    SMEProfile(
        description=(
            "A health-data startup applying NLP and federated learning to clinical "
            "records to support early diagnosis while preserving patient privacy."
        ),
        country="FR", org_type="startup", trl=4,
        keywords=["health data", "NLP", "federated learning", "privacy"],
    ),
]

_MEET = {"yes": "appears to match", "unclear": "unclear — verify",
         "no": "appears NOT to match", "not_stated": "not stated"}


def _print_call(i: int, c: dict) -> None:
    score = c.get("score")
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    print(f"\n  {i}. [{score_s}] {c.get('title', '')[:80]}")
    print(f"     deadline: {(c.get('deadline') or 'rolling/none')[:10]}  | "
          f"{c.get('type_of_action') or c.get('programme') or ''}")
    print(f"     {c.get('source_url', '')}")
    print(f"\n     FIT: {c.get('fit_summary', '') or '(none)'}")
    for r in c.get("fit_reasons", []):
        print(f"       • {r['claim']}")
        print(f"         ↳ cite {r['source_id']}  «{r['snippet'][:120]}»")
    elig = c.get("eligibility", [])
    print(f"\n     ELIGIBILITY (DRAFT — not a verdict): {len(elig)} stated condition(s)")
    for e in elig:
        print(f"       • {e['condition']}  [{_MEET.get(e['applicant_appears_to_meet'])}]")
        print(f"         ↳ {e['source_url']}")
        print(f"         «{e['snippet'][:140]}»")
    mi = c.get("missing_info", [])
    if mi:
        print(f"\n     MUST VERIFY IN CALL DOCS:")
        for m in mi:
            print(f"       - {m}")
    g = c.get("_grounding", {})
    print(f"\n     [grounding] dropped_fit={g.get('dropped_fit', 0)} "
          f"dropped_eligibility={g.get('dropped_eligibility', 0)} "
          f"warnings={g.get('warnings', [])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="raw JSON output")
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    for p in EXAMPLES:
        result = run(p, top_n=args.top_n)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue
        print("\n" + "=" * 92)
        print(f"SME: {p.description[:120]}")
        print(f"     country={p.country} org_type={p.org_type} trl={p.trl}")
        print("=" * 92)
        for i, c in enumerate(result["shortlist"], 1):
            _print_call(i, c)


if __name__ == "__main__":
    main()
