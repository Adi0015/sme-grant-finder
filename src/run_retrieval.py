"""CLI smoke test for the retrieval module (Day 3, Block D).

Eyeball retrieval quality on a few example SME profiles before we build the gold
eval set on Day 5. Not part of the importable API.

    python src/run_retrieval.py                       # runs the 3 example profiles
    python src/run_retrieval.py "your SME description here"   # ad-hoc one-off
    python src/run_retrieval.py --json "..."          # raw JSON payload
"""

from __future__ import annotations

import argparse
import json

from retrieve import retrieve
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
            "A green-energy startup building offshore wind turbine optimisation and "
            "smart-grid integration to increase renewable energy efficiency."
        ),
        country="DK", org_type="startup", trl=5,
        keywords=["wind energy", "smart grid", "renewables"],
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


def _print_profile(p: SMEProfile) -> None:
    print("\n" + "=" * 90)
    print(f"SME: {p.description[:110]}...")
    print(f"     country={p.country} org_type={p.org_type} trl={p.trl} keywords={p.keywords}")
    print("=" * 90)


def _print_human(result: dict) -> None:
    print("\n  MATCHED CALLS (open, you could apply):")
    for i, c in enumerate(result["matched_calls"], 1):
        print(f"  {i:>2}. {c['best_score']:.3f}  {c['title'][:70]}")
        print(f"      [{c['type_of_action'] or c['programme']}] deadline={c['deadline'][:10] or 'rolling'}")
        print(f"      {c['source_url']}")
    print("\n  EVIDENCE PROJECTS (similar work that was funded before):")
    for i, e in enumerate(result["evidence_projects"], 1):
        print(f"  {i:>2}. {e['score']:.3f}  {e['title'][:70]}  [{e['country']}]")
        print(f"      {e['source_url']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("description", nargs="?", help="ad-hoc SME description")
    ap.add_argument("--json", action="store_true", help="print raw JSON payload")
    ap.add_argument("--k", type=int, default=10, help="number of calls to return")
    args = ap.parse_args()

    profiles = [SMEProfile(description=args.description)] if args.description else EXAMPLES

    for p in profiles:
        result = retrieve(p, k_calls=args.k, k_evidence=5)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_profile(p)
            _print_human(result)


if __name__ == "__main__":
    main()
