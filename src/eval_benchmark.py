"""Config benchmark + MLflow logging (Day 5, Block D).

Compares two retrieval configs by RE-RANKING the same frozen candidate pool (the dense
top-20 per profile from candidates.json), so the comparison is apples-to-apples and every
ranked item has a human label:
  - dense   : the production config — e5 cosine score (Chroma).
  - hybrid  : dense fused with a BM25 keyword score over call title+description, via
              Reciprocal Rank Fusion (RRF). Tests whether lexical signal sharpens the
              compressed e5 score band noted on Day 3.

Metrics (recall@{5,10,20}, MRR, precision@5) are logged to a local MLflow file store
(./mlruns, experiment "raise-retrieval"), one run per config, with params so they're
comparable in the MLflow UI. If data/gold/metrics_faithfulness.json exists, its numbers
are attached to the dense (main) run.

LIMITATION: this is a re-ranking benchmark over the dense pool — hybrid cannot get credit
for relevant calls outside the labeled top-20. Stated in notes/results.md.

    python src/eval_benchmark.py            # requires data/gold/labeled.csv
"""

from __future__ import annotations

import json
import re

import config
from eval_retrieval import load_candidates, load_labels, _metrics_for_profile, KS

RRF_K = 60          # standard RRF damping constant
_TOK = re.compile(r"[a-z0-9]+", re.I)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text or "")]


def _calls_text() -> dict[str, str]:
    calls = json.loads(config.CALLS_JSON.read_text(encoding="utf-8"))
    return {c["source_id"]: f"{c.get('title','')} {c.get('description','')}" for c in calls}


def _profiles() -> dict[str, "object"]:
    from schema import SMEProfile
    raw = json.loads((config.DATA / "gold" / "profiles.json").read_text(encoding="utf-8"))
    return {r["id"]: SMEProfile(**{k: v for k, v in r.items() if k != "id"}) for r in raw}


def _rerank_dense(pool: list[dict]) -> list[dict]:
    ranked = sorted(pool, key=lambda c: c["score"], reverse=True)
    return [{**c, "rank": i} for i, c in enumerate(ranked, 1)]


def _rerank_hybrid(pool: list[dict], query: str, text_by_id: dict[str, str]) -> list[dict]:
    from rank_bm25 import BM25Okapi
    docs = [text_by_id.get(c["source_id"], c["title"]) for c in pool]
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    bm_scores = bm25.get_scores(_tokenize(query))

    dense_rank = {c["source_id"]: r for r, c in
                  enumerate(sorted(pool, key=lambda c: c["score"], reverse=True), 1)}
    bm_order = sorted(range(len(pool)), key=lambda i: bm_scores[i], reverse=True)
    bm_rank = {pool[i]["source_id"]: r for r, i in enumerate(bm_order, 1)}

    fused = []
    for c in pool:
        sid = c["source_id"]
        rrf = 1.0 / (RRF_K + dense_rank[sid]) + 1.0 / (RRF_K + bm_rank[sid])
        fused.append({**c, "_rrf": rrf})
    fused.sort(key=lambda c: c["_rrf"], reverse=True)
    return [{**c, "rank": i} for i, c in enumerate(fused, 1)]


def _eval_config(name: str, candidates, labels, rerank_fn, profiles, text_by_id) -> dict:
    per_profile, skipped = {}, []
    for pid, pool in candidates.items():
        rel = {c["source_id"] for c in pool if labels.get((pid, c["source_id"]), 0) == 1}
        if not rel:
            skipped.append(pid)
            continue
        if name == "hybrid":
            ranked = rerank_fn(pool, profiles[pid].to_query_text(), text_by_id)
        else:
            ranked = rerank_fn(pool)
        per_profile[pid] = _metrics_for_profile(ranked, rel)
    keys = [f"recall@{k}" for k in KS] + ["precision@5", "MRR"]
    mean = {m: (sum(p[m] for p in per_profile.values()) / len(per_profile)
                if per_profile else 0.0) for m in keys}
    return {"mean": mean, "n_evaluated": len(per_profile), "skipped": skipped}


def main() -> None:
    import mlflow

    candidates = load_candidates()
    labels = load_labels()              # exits cleanly if labeled.csv missing
    profiles = _profiles()
    text_by_id = _calls_text()

    configs = {
        "dense": dict(params={"retriever": "dense", "embedding_model": config.MODEL_NAME,
                              "distance": config.DISTANCE, "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
                              "pool_k": 20},
                      fn=_rerank_dense),
        "hybrid": dict(params={"retriever": "hybrid_dense+bm25_rrf", "embedding_model": config.MODEL_NAME,
                               "distance": config.DISTANCE, "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
                               "pool_k": 20, "rrf_k": RRF_K},
                       fn=_rerank_hybrid),
    }

    faith = None
    fpath = config.DATA / "gold" / "metrics_faithfulness.json"
    if fpath.exists():
        faith = json.loads(fpath.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment("raise-retrieval")

    results = {}
    for name, cfg in configs.items():
        rep = _eval_config(name, candidates, labels, cfg["fn"], profiles, text_by_id)
        results[name] = rep
        with mlflow.start_run(run_name=name):
            mlflow.log_params(cfg["params"])
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in rep["mean"].items()})
            mlflow.log_metric("n_evaluated", rep["n_evaluated"])
            if name == "dense" and faith:
                mlflow.log_metric("faithfulness_code_grounding", faith["code_grounding_pct"])
                mlflow.log_metric("faithfulness_judge_supported", faith["judge_supported_pct"])

    print("=== CONFIG BENCHMARK (re-ranking dense top-20 pool) ===")
    hdr = ["config"] + [f"recall@{k}" for k in KS] + ["prec@5", "MRR"]
    print("  ".join(f"{h:>10}" for h in hdr))
    for name, rep in results.items():
        m = rep["mean"]
        row = [name] + [f"{m[f'recall@{k}']:.3f}" for k in KS] + [f"{m['precision@5']:.3f}", f"{m['MRR']:.3f}"]
        print("  ".join(f"{c:>10}" for c in row))
    print(f"\nLogged to {config.ROOT / 'mlruns'} (experiment: raise-retrieval). "
          f"View: mlflow ui --backend-store-uri file:{config.ROOT / 'mlruns'}")


if __name__ == "__main__":
    main()
