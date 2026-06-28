"""Build the gold-set labeling scaffold (Day 5, Block A).

INTEGRITY RULE: this script does NOT assign relevance. It only produces the candidate
pool + an empty labeling sheet for a HUMAN to fill. The `relevant` column is left blank
on purpose. A gold set labeled by an LLM would be circular and worthless.

What it writes (data/gold/):
  - candidates.json  : frozen ranked candidate pool per profile (rank, source_id, score,
                       title, url). Metrics read THIS so the labels map exactly to the
                       ranking that was shown, fully deterministic.
  - to_label.csv     : [profile_id, call_source_id, call_title, source_url, relevant]
                       with `relevant` EMPTY — the human fills 1 (relevant) / 0 (not).

Relevance maps to ragold's relevant-vs-distracting split (1 = relevantChunks,
0 = distractingChunks); we use a flat CSV so it can be labeled in any spreadsheet.

    python src/build_gold.py            # (re)generate candidates.json + to_label.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import config
from retrieve import retrieve_calls
from schema import SMEProfile

GOLD = config.DATA / "gold"
PROFILES = GOLD / "profiles.json"
CANDIDATES = GOLD / "candidates.json"
TO_LABEL = GOLD / "to_label.csv"
LABELED = GOLD / "labeled.csv"          # the human writes this
POOL_K = 20                             # candidate pool depth per profile


def load_profiles() -> list[tuple[str, SMEProfile]]:
    raw = json.loads(PROFILES.read_text(encoding="utf-8"))
    out = []
    for rec in raw:
        pid = rec["id"]
        fields = {k: v for k, v in rec.items() if k != "id"}
        out.append((pid, SMEProfile(**fields)))
    return out


def main() -> None:
    profiles = load_profiles()
    print(f"Loaded {len(profiles)} SME profiles from {PROFILES.relative_to(config.ROOT)}")

    candidates: dict[str, list[dict]] = {}
    rows: list[dict] = []
    for pid, profile in profiles:
        calls = retrieve_calls(profile, k=POOL_K)
        pool = []
        for rank, c in enumerate(calls, 1):
            pool.append({
                "rank": rank,
                "source_id": c["source_id"],
                "title": c["title"],
                "source_url": c["source_url"],
                "score": c["best_score"],
            })
            rows.append({
                "profile_id": pid,
                "call_source_id": c["source_id"],
                "call_title": c["title"],
                "source_url": c["source_url"],
                "relevant": "",          # <-- HUMAN fills this (1 = relevant, 0 = not)
            })
        candidates[pid] = pool
        print(f"  {pid:24s} -> {len(pool)} candidates (top: {pool[0]['title'][:48] if pool else 'none'})")

    GOLD.mkdir(parents=True, exist_ok=True)
    CANDIDATES.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    with TO_LABEL.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["profile_id", "call_source_id", "call_title",
                                          "source_url", "relevant"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {CANDIDATES.relative_to(config.ROOT)} (frozen ranked pool)")
    print(f"Wrote {TO_LABEL.relative_to(config.ROOT)} ({len(rows)} rows, `relevant` EMPTY)")
    _print_instructions(len(profiles), len(rows))


def _print_instructions(n_profiles: int, n_rows: int) -> None:
    print("\n" + "=" * 78)
    print("HUMAN LABELING — required before any metrics can be computed")
    print("=" * 78)
    print(f"""
1. Open  data/gold/to_label.csv  in a spreadsheet (Numbers / Excel / LibreOffice).
2. For each of the {n_rows} rows, read the call_title (open source_url if unsure) and
   put a value in the `relevant` column:
        1  = this call is a plausible funding fit for that SME profile
        0  = not relevant
   Leave nothing blank. Judge from the SME's perspective: would they apply?
3. Save the completed file as  data/gold/labeled.csv  (same columns, `relevant` filled).
4. Tell me it's labeled — I'll run:
        venv-index/bin/python src/eval_retrieval.py
        venv-index/bin/python src/eval_faithfulness.py

NOTE: {n_profiles} profiles x {n_rows // max(n_profiles,1)} candidates each. ~{n_rows} judgments.
I did NOT pre-fill any labels (that would make the gold set circular). The pool ranking
is frozen in candidates.json so your labels map exactly to what the system retrieved.
""")


if __name__ == "__main__":
    main()
