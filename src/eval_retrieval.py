"""Retrieval metrics against the HUMAN-labeled gold set (Day 5, Block B).

Reads the frozen candidate pool (data/gold/candidates.json) and the human labels
(data/gold/labeled.csv) and computes, per profile and as a mean:
  - recall@k for k in {5, 10, 20}
  - MRR (reciprocal rank of the first relevant call)
  - precision@5

Deterministic and re-runnable. Labels never touch retrieval — they are only joined
onto the frozen ranking after the fact.

IMPORTANT (honest framing): recall here is POOL-RELATIVE. The gold labels live inside
each profile's top-20 retrieved pool, so the "relevant universe" is the labeled-relevant
calls within that pool, not the whole corpus. recall@20 is therefore ~1.0 by
construction; recall@5 / @10 and MRR are the informative signals (how well the ranker
pushes the relevant calls to the top). Stated as a limitation in notes/results.md.

    python src/eval_retrieval.py            # prints tables; also writes data/gold/metrics_retrieval.json
"""

from __future__ import annotations

import csv
import json
import sys

import config

GOLD = config.DATA / "gold"
CANDIDATES = GOLD / "candidates.json"
LABELED = GOLD / "labeled.csv"
OUT = GOLD / "metrics_retrieval.json"
KS = (5, 10, 20)


def load_candidates() -> dict[str, list[dict]]:
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))


def load_labels() -> dict[tuple[str, str], int]:
    """(profile_id, source_id) -> 1/0 from the human-labeled CSV."""
    if not LABELED.exists():
        sys.exit(f"ERROR: {LABELED} not found. Label data/gold/to_label.csv first "
                 f"(see `python src/build_gold.py`).")
    labels: dict[tuple[str, str], int] = {}
    with LABELED.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = (row.get("relevant") or "").strip()
            if val not in ("0", "1"):
                continue
            labels[(row["profile_id"], row["call_source_id"])] = int(val)
    return labels


def _metrics_for_profile(pool: list[dict], rel: set[str]) -> dict:
    ranked = [c["source_id"] for c in sorted(pool, key=lambda c: c["rank"])]
    n_rel = len(rel)
    out = {}
    for k in KS:
        topk = ranked[:k]
        hits = sum(1 for sid in topk if sid in rel)
        out[f"recall@{k}"] = hits / n_rel if n_rel else 0.0
    out["precision@5"] = sum(1 for sid in ranked[:5] if sid in rel) / 5.0
    mrr = 0.0
    for i, sid in enumerate(ranked, 1):
        if sid in rel:
            mrr = 1.0 / i
            break
    out["MRR"] = mrr
    out["n_relevant"] = n_rel
    return out


def evaluate() -> dict:
    candidates = load_candidates()
    labels = load_labels()

    per_profile: dict[str, dict] = {}
    skipped: list[str] = []
    for pid, pool in candidates.items():
        rel = {c["source_id"] for c in pool
               if labels.get((pid, c["source_id"]), 0) == 1}
        if not rel:
            skipped.append(pid)
            continue
        per_profile[pid] = _metrics_for_profile(pool, rel)

    metric_keys = [f"recall@{k}" for k in KS] + ["precision@5", "MRR"]
    mean = {m: (sum(p[m] for p in per_profile.values()) / len(per_profile)
                if per_profile else 0.0) for m in metric_keys}
    return {"per_profile": per_profile, "mean": mean, "skipped_no_relevant": skipped,
            "n_evaluated": len(per_profile)}


def _print(report: dict) -> None:
    hdr = ["profile"] + [f"recall@{k}" for k in KS] + ["prec@5", "MRR", "#rel"]
    print("  ".join(f"{h:>22}" if h == "profile" else f"{h:>9}" for h in hdr))
    for pid, m in report["per_profile"].items():
        row = [f"{pid:>22}"] + [f"{m[f'recall@{k}']:.3f}" for k in KS]
        row += [f"{m['precision@5']:.3f}", f"{m['MRR']:.3f}", f"{m['n_relevant']:>4}"]
        print("  ".join(f"{c:>9}" if i else f"{c}" for i, c in enumerate(row)))
    mean = report["mean"]
    mrow = [f"{'MEAN':>22}"] + [f"{mean[f'recall@{k}']:.3f}" for k in KS]
    mrow += [f"{mean['precision@5']:.3f}", f"{mean['MRR']:.3f}", ""]
    print("-" * 86)
    print("  ".join(f"{c:>9}" if i else f"{c}" for i, c in enumerate(mrow)))
    if report["skipped_no_relevant"]:
        print(f"\n  WARN: skipped (no labeled-relevant calls): {report['skipped_no_relevant']}")
    print(f"  evaluated {report['n_evaluated']} profiles")


def main() -> None:
    report = evaluate()
    _print(report)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
