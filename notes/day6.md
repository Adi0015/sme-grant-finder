# Day 6 — UI + README + writeup + reproducibility

**Goal:** thin UI over `run()`, a 2-minute README, a short honest writeup, and a
cloneable-and-runnable repo. No outreach/CV (Day 7). Stop at the gate.
See [[results]], [[writeup]], [[notes]] (Day 6 section), [[day5]].

## Principle held
The engine is the point. `app.py` imports `pipeline.run()` and only renders — no pipeline
refactor, no frontend gold-plating.

## Gate report

### 1. app.py runs locally — ✅
Launch: `venv-index/bin/streamlit run app.py` (or `make run-app`). Verified serving
`HTTP 200`, `/_stcore/health` = `ok`, `/?demo=1` = `HTTP 200`. Citations (source link +
verbatim snippet) are visible on every fit/eligibility line; disclaimer banner present.
**Screenshot:** ASCII mock in README (no Chromium for headless capture; a full-screen
grab would leak the desktop) — drop a real PNG at `docs/screenshot.png` before publishing.

### 2. README.md — ✅
2-minute structure: pitch + ASCII UI mock → problem → mermaid architecture → data sources
(449 calls / 400 projects) → design choices → results table → RAISE relation → honest
limitations → run (local + Docker).

### 3. notes/writeup.md — ✅
~450 words, builder's voice: what's hard (compressed e5 band, eligibility-as-pointers),
where RAG helped/struggled, the fabrication-hidden-by-fuzzy surprise, what's next
(experience-based config selection à la CBR).

### 4. Docker — ⚠ written, NOT built
`Dockerfile` (python:3.13-slim, installs requirements, copies src+app+samples, healthcheck)
+ `.dockerignore` (excludes chroma/mlruns/venv/large data). **Not built — no docker on this
machine.** Build/run: `make docker-build` / `make docker-run`. App needs `data/` mounted +
`OLLAMA_HOST=host.docker.internal` (both documented); demo mode needs neither.

### 5. Cloneable-and-runnable by a stranger — ✅ (with one prereq)
- `make setup` → `make demo` → open `?demo=1` works with **no index and no LLM** (cached
  `data/samples/demo_shortlist.json`).
- Full pipeline prereqs flagged in README: **`make build-index` first** (Chroma index) +
  **Ollama with `qwen2.5:7b` pulled**. Pinned `requirements.txt` (py3.13).
- Ships `data/samples/` + `data/gold/`; gitignores large corpora, chroma, mlruns, venvs.
- No secrets committed (scanned). Backend/keys read from env.

### 6. Blockers
- **None for Day 6.** External gaps: docker not installed here (build unverified);
  retrieval numbers in README/results stay "labeling in progress" until Day-5
  `data/gold/labeled.csv` exists.

## STOP — Day 6 gate reached
Do NOT proceed to Day 7 (OSS engagement, CV update, email draft). Awaiting go.
