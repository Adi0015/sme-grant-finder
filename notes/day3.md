# Day 3 — retrieval module

**Goal:** a clean, importable `retrieve(sme_profile)` → ranked matching calls +
supporting evidence projects, every item citation-ready. NO LLM, eligibility
summaries, or UI today. Stop at the gate.
See [[scope]] for framing, [[notes]] for the Day-3 design details, [[day2]] for Day 2.

## Log

- [x] Refactor shared config → `src/config.py` (model, e5 prefixes, store, score helper); `build_index.py` now imports it.
- [x] Block A — `src/schema.py`: `SMEProfile` dataclass + `to_query_text()`.
- [x] Block B — `src/retrieve.py`: `retrieve_calls`, `retrieve_evidence`, `retrieve()`.
- [x] Block C — score orientation (higher=better) + no hard eligibility filter (assistive, Day-4).
- [x] Block D — `src/run_retrieval.py` CLI smoke test over 3 example profiles.

## Gate report

### 1. `src/schema.py` — ✅
`SMEProfile(description, country=None, org_type=None, budget=None, trl=None, keywords=None)`.
`to_query_text()` puts the free-text description first (dominant signal), then a ` | `-joined
context line of the structured fields. The e5 `"query: "` prefix is applied once in
`config.embed_query()` — not duplicated in the schema.

### 2. `retrieve(profile)` working — ✅ (top hits per profile)

**AI predictive-maintenance SME (DE, TRL 6):**
- Calls: 0.819 Industrial leadership in AI, Data and Robotics · 0.814 Security/Privacy/Robustness of AI Models · 0.810 semiconductors skills
- Evidence: 0.843 Resilient manufacturing lines (smart handling) · 0.834 AI in Manufacturing for Sustainability · 0.834 enabling SMEs to develop AI & DATA solutions

**Green-energy startup (DK, TRL 5):**
- Calls: 0.838 Improve wind energy systems · 0.838 Next-gen renewable energy tech · 0.825 green+digital transformation of energy
- Evidence: 0.839 Data-driven efficient power distribution · 0.832 climate-neutral converting facilities · 0.831 energy-aware swarms

**Health-data startup (FR, TRL 4):**
- Calls: 0.837 Rare Diseases partnership (ERDERA) · 0.834 EEHRxF digital health services · 0.830 AI image screening in medical centres · 0.828 AI uptake in Health
- Evidence: 0.834 AI-ready Data Spaces · 0.823 EDGELM privacy-preserving edge multimodal

Full ranked output (10 calls + 5 evidence each) via `python src/run_retrieval.py`.
Every item carries `source_id` + `source_url` + `title` + `score`.

### 3. Dedupe + score orientation
- **Chunks → parent**: over-fetch `max(k*5, 50)` chunks (Chroma returns best-first), keep the first chunk per `source_id` (its best), rank sources by that best score, take top-k.
- **Scores**: cosine collection → Chroma returns distance `= 1 - cosine_sim`; we invert (`similarity = 1 - distance`) so **higher = better**. All scores are this similarity.

### 4. Honest take
- Retrieval is **sensible**: top-3 per profile are squarely on-domain, and the evidence projects are strong (real manufacturing/energy/health-AI funded work).
- **Misses / caveats**:
  - Tails drift into semantic neighbours (defense AI for manufacturing, satellite for energy, plant-health for human-health) — pure dense retrieval.
  - e5 similarities sit in a **narrow ~0.80–0.84 band** → fixed thresholds unreliable; a hybrid BM25+dense re-rank (cbrkit MAC/FAC) is the lever for Day 5+.
  - Some "Open" calls have past nominal deadlines (rolling calls) — surface but flag.

### 5. Blockers
- **None.** Module imports cleanly, runs on all 3 profiles, output JSON-serializable.

## STOP — Day 3 gate reached
Do NOT proceed to Day 4 (generation: fit + draft eligibility + citations). Awaiting go.
