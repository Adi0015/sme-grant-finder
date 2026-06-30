# RAISE — Day 5 evaluation results

Honest numbers for the SME grant-finder. Retrieval quality against an **AI-drafted,
human-reviewed** gold set, plus a two-layer faithfulness check on generated claims. See
[[day5]] for the gate report, [[notes]] for design details.

> **Status:** all metrics computed. **Label provenance:** the relevance labels were
> **drafted by an assistant and reviewed/accepted by the author** (`data/gold/labeled.csv`)
> — not hand-labeled from scratch, and not auto-graded by the retrieval model itself.
> Treat the retrieval numbers as **indicative**, not a clean-room gold standard.

## Setup

- **Gold set:** 10 SME profiles (`data/gold/profiles.json`) spanning AI/manufacturing,
  green energy, health-data, agri-tech, logistics, cybersecurity, fintech, space/EO,
  circular-economy, e-mobility. Per profile, the dense retriever's **top-20** calls form a
  frozen candidate pool (`data/gold/candidates.json`) → **200 candidate rows** to label.
- **Labeling:** 1 = plausible funding fit / 0 = not, **assistant-drafted then
  human-reviewed** (42/200 judged relevant). Labels are joined onto the frozen ranking
  after the fact — they never touch retrieval.
- **Stack:** retrieval = `intfloat/multilingual-e5-base` + Chroma (cosine); generation +
  judge = `ollama:qwen2.5:7b` (local).

## Retrieval metrics (pool-relative)

Dense retriever (`multilingual-e5` + Chroma), `data/gold/labeled.csv`, 10 profiles:

| profile | recall@5 | recall@10 | recall@20 | prec@5 | MRR | #rel |
|---------|---------|----------|----------|--------|-----|------|
| p01-ai-manufacturing | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1 |
| p02-green-energy | 0.444 | 0.556 | 1.000 | 0.800 | 1.000 | 9 |
| p03-health-data | 0.571 | 0.714 | 1.000 | 0.800 | 1.000 | 7 |
| p04-agritech | 0.333 | 0.500 | 1.000 | 0.400 | 0.500 | 6 |
| p05-logistics | 0.667 | 0.667 | 1.000 | 0.400 | 0.500 | 3 |
| p06-cybersecurity-iot | 0.400 | 0.600 | 1.000 | 0.400 | 1.000 | 5 |
| p07-fintech-regtech | 0.000 | 0.500 | 1.000 | 0.000 | 0.167 | 2 |
| p08-space-eo | 0.667 | 1.000 | 1.000 | 0.400 | 0.500 | 3 |
| p09-circular-economy | 0.750 | 1.000 | 1.000 | 0.600 | 0.500 | 4 |
| p10-emobility | 0.500 | 0.500 | 1.000 | 0.200 | 0.500 | 2 |
| **MEAN** | **0.533** | **0.704** | 1.000 | **0.420** | **0.667** | |

Read: the first genuinely-relevant call usually ranks near the top (**MRR 0.67**), but the
tail is noisy (**prec@5 0.42**) — pure dense retrieval drags cross-domain calls into the
pool. recall@20 = 1.0 by construction (labels live inside the top-20). Weakest:
p07-fintech (first hit at rank 6); strongest: p01/p02/p03/p06 (MRR 1.0).

## Config benchmark (re-rank of the dense top-20 pool)

Logged to MLflow (`./mlruns`, experiment `raise-retrieval`):

| config | recall@5 | recall@10 | prec@5 | MRR |
|--------|---------|----------|--------|-----|
| dense  | 0.533 | 0.704 | 0.420 | 0.667 |
| **hybrid (dense + BM25, RRF)** | 0.522 | **0.784** | 0.420 | **0.700** |

**Defensible finding #2:** adding a lexical (BM25) signal and fusing with RRF lifts
**recall@10 0.70 → 0.78** and **MRR 0.67 → 0.70** over dense-only — evidence that the
compressed e5 score band (0.80–0.84, where great and mediocre matches barely separate) is
sharpened by a hybrid re-rank, exactly the cbrkit MAC/FAC pattern. (recall@5 dips
marginally; the gain is in the tail.)

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

- **Small, AI-drafted + single-reviewer gold set** (10 profiles, 200 judgments, labels
  assistant-drafted then reviewed by one author) — no inter-annotator agreement, and the
  draft step means the labels are not an independent human ground truth. Treat metrics as
  indicative, not publication-grade. Hand-labeling from scratch is the upgrade path.
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
