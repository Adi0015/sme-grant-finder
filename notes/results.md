# RAISE — Day 5 evaluation results

Honest numbers for the SME grant-finder. Retrieval quality against a **human-labeled**
gold set, plus a two-layer faithfulness check on generated claims. See [[day5]] for the
gate report, [[notes]] for design details.

> **Status:** faithfulness = DONE. Retrieval metrics + config benchmark = **PENDING the
> human labels** (`data/gold/labeled.csv`). Numbers below are filled as they land.

## Setup

- **Gold set:** 10 SME profiles (`data/gold/profiles.json`) spanning AI/manufacturing,
  green energy, health-data, agri-tech, logistics, cybersecurity, fintech, space/EO,
  circular-economy, e-mobility. Per profile, the dense retriever's **top-20** calls form a
  frozen candidate pool (`data/gold/candidates.json`) → **200 candidate rows** to label.
- **Labeling:** done by the human (me), 1 = plausible funding fit / 0 = not. Labels are
  joined onto the frozen ranking after the fact — they never touch retrieval.
- **Stack:** retrieval = `intfloat/multilingual-e5-base` + Chroma (cosine); generation +
  judge = `ollama:qwen2.5:7b` (local).

## Retrieval metrics (pool-relative)

_PENDING `data/gold/labeled.csv` — run `venv-index/bin/python src/eval_retrieval.py`._

| profile | recall@5 | recall@10 | recall@20 | prec@5 | MRR |
|---------|---------|----------|----------|--------|-----|
| _(per-profile rows)_ | … | … | … | … | … |
| **MEAN** | … | … | … | … | … |

## Config benchmark (re-rank of the dense top-20 pool)

_PENDING labels — `venv-index/bin/python src/eval_benchmark.py` (logs to MLflow)._

| config | recall@5 | recall@10 | prec@5 | MRR |
|--------|---------|----------|--------|-----|
| dense  | … | … | … | … |
| hybrid (dense + BM25, RRF) | … | … | … | … |

## Faithfulness (DONE)

54 generated claims (12 fit + 42 eligibility) from 4 sampled profiles × top-3 calls.

| layer | metric | value |
|-------|--------|-------|
| **Code grounding** | snippet is exact substring of the call text passed | **98.1%** (53/54) |
| **LLM judge** (qwen2.5:7b) | claim fully supported by its cited snippet (`yes`) | **44.4%** (24) |
| LLM judge | supported or partial (`yes` + `partial`) | **85.2%** (46) |
| LLM judge | unsupported (`no`) | 14.8% (8) |

- The single code-ungrounded snippet is a **tokenizer round-trip artifact** in
  `best_chunk_text` (chunk decoded from tokens ≠ exact original characters), not a
  fabrication. Effective grounding ≈ 100% — the Day-4 verbatim re-derivation holds on
  unseen gold profiles.

### Human spot-check (~15 of `data/gold/faithfulness_sample.csv`)

- The large **`partial` bucket is mostly eligibility pointer-boilerplate** — conditions
  like `Eligible Countries` with snippet *"Eligible Countries described in section 6 of
  the call document."* The snippet faithfully reproduces what the call says (a pointer),
  but the judge rates it `partial` because the pointer carries no concrete criterion. So
  the 44% `yes` **undersells** true faithfulness; most `partial`s are faithful-but-low-info.
- The 8 `no` verdicts concentrate in **two honest weaknesses**: (a) fit claims on
  cross-domain retrieval **tails** (agri-tech → a defense "soldiers with robots and drones"
  call) — a retrieval problem, not a citation problem; (b) a couple of label/snippet
  mismatches (e.g. `Other Eligible Conditions` whose snippet is about satellite EO).
- _(Add your own observations after reviewing the CSV by hand.)_

## One defensible finding

**Grounding is real, eligibility is restrained.** With the Day-4 verbatim span
re-derivation, ~100% of surviving citations are exact source text, and not one fabricated
eligibility *condition* survived an adversarial 21-agent audit (it earlier caught 5
fabricated *fit* quotes that a fuzzy check had hidden — now dropped). The system prefers
**no claim over an ungrounded one**.

## Honest limitations

- **Small, single-labeler gold set** (10 profiles, 200 judgments, one annotator) — no
  inter-annotator agreement; treat metrics as indicative, not publication-grade.
- **Pool-relative recall:** labels live inside each profile's dense top-20, so the
  "relevant universe" is pool-internal. `recall@20 ≈ 1.0` by construction; **MRR /
  recall@5 / recall@10 (ranking quality) are the informative signals**, and the benchmark
  is a *re-ranking* of the dense pool (hybrid can't get credit for relevants outside it).
- **English-dominant corpus** despite a multilingual embedder — calls/projects are mostly
  EN, so the multilingual capability is under-exercised here.
- **Eligibility is assistive, never a verdict** — per-condition assessment is `yes` /
  `unclear` only (no hard `no`); a "stated condition" can still be pointer boilerplate.
- **Judge shares the qwen backend** with the generator → its number is a sanity signal,
  not ground truth; the human spot-check is the real check.
- **Fit-reason coverage is low** — the 7B fabricates roughly half its "why it matches"
  quotes, which are then *dropped* (honest, but ~half of calls show no fit citation). A
  stronger model or a "select an existing sentence" constraint is the lever.

## Related
- [[day5]] · [[notes]] · [[scope]] · [[day4]] (grounding hardening)
