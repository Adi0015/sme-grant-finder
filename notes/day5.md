# Day 5 — evaluation harness

**Goal:** honest numbers — retrieval quality vs a HUMAN-labeled gold set + a faithfulness
check on generated claims, logged to MLflow. No UI (Day 6). Stop at the gate.
See [[results]] for the full results write-up, [[notes]] for design, [[day4]] for the
generation layer being evaluated.

## Integrity rule honored
The gold set is labeled BY THE HUMAN. This scaffold generates the candidate pool and an
EMPTY labeling sheet; **no labels were invented by an LLM**. The faithfulness judge is a
separate, disclosed LLM-as-judge layer (not the gold set) and is backed up by a human
spot-check.

## Log
- [x] Block A — `src/build_gold.py`: 10 SME profiles → dense top-20 candidate pool →
      `data/gold/to_label.csv` (200 rows, `relevant` blank) + frozen `candidates.json`.
- [x] Block B — `src/eval_retrieval.py`: recall@{5,10,20}, MRR, precision@5 (pool-relative).
- [x] Block C — `src/eval_faithfulness.py`: code-grounding + LLM-judge + spot-check CSV. **Ran.**
- [x] Block D — `src/eval_benchmark.py`: dense vs hybrid (BM25+RRF) re-rank, MLflow logging.
- [x] Block E — `notes/results.md`.

## Gate report

### 1. `data/gold/to_label.csv` generated — ✅ (labels NOT filled)
200 rows = 10 profiles × top-20 candidates. Columns `[profile_id, call_source_id,
call_title, source_url, relevant]`; `relevant` left EMPTY for the human. Verified all 200
blank. Frozen ranking in `candidates.json` so labels map exactly to what was retrieved.

### 2. recall@k + MRR — ⏳ PENDING `data/gold/labeled.csv`
Code ready: `venv-index/bin/python src/eval_retrieval.py` → per-profile + mean table,
writes `metrics_retrieval.json`. (Recall is pool-relative; MRR / recall@5,10 are the real
signal — see [[results]] limitations.)

### 3. Faithfulness — ✅ (54 claims, 4 profiles × top-3)
- **Code grounding: 98.1%** (53/54 snippets exact substrings; the 1 miss is a tokenizer
  round-trip artifact in `best_chunk_text`, not a fabrication).
- **LLM-judge: 44.4% `yes`, 85.2% `yes`+`partial`, 8 `no`** (qwen2.5:7b).
- Spot-check CSV: `data/gold/faithfulness_sample.csv` (review ~15 by hand). Most `partial`s
  are faithful-but-low-info eligibility pointer boilerplate; `no`s cluster on cross-domain
  fit tails.

### 4. MLflow — ✅ infra proven (`./mlruns`, experiment `raise-retrieval`)
Faithfulness run logged now (params: retriever/embedding/llm_backend; metrics:
code-grounding, judge-supported). Retrieval + dense-vs-hybrid benchmark log on labeling via
`src/eval_benchmark.py`. (MLflow ≥3 needs `MLFLOW_ALLOW_FILE_STORE=true` for the file
store — set in code.)

### 5. Honest read
- **Grounding holds on unseen profiles** (~100% verbatim) — the Day-4 re-derivation
  generalized; no fabricated eligibility condition survived.
- **Weakest part = fit-reason coverage + retrieval tails.** The 7B fabricates ~half its
  "why it matches" quotes (now dropped, so ~half of calls have no fit citation), and dense
  retrieval drags cross-domain calls into the pool (agri-tech ↔ defense swarms). Both are
  why the gold labels + hybrid benchmark matter.
- **Judge ≠ ground truth** (same qwen family) → the human spot-check is the real check.

### 6. Blockers
- **Human labeling** of `data/gold/to_label.csv` → save as `data/gold/labeled.csv`. That
  unblocks gate items 2 and 4 (benchmark). Nothing else blocking.

## STOP — Day 5 gate reached
Do NOT proceed to Day 6 (Streamlit UI + writeup). After labeling: run
`eval_retrieval.py` + `eval_benchmark.py`, fill [[results]], report. Awaiting go.
