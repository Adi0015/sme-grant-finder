# Day 2 — corpora + vector index

**Goal:** build the full corpora (open calls + CORDIS projects) and a persistent
multilingual vector store with citation-grade metadata. NO retrieval pipeline,
generation, or UI today. Stop at the gate.
See [[scope]] for framing, [[notes]] for the data-source playbooks + Day-2 details, [[day1]] for Day 1.

## Log

- [x] Env: torch/chromadb have no cp314 wheels → new venv `venv-index` on Python 3.13.
- [x] Block A — SEDIA: paginate ALL open grant topics → `data/calls.json` (449).
- [x] Block A — CORDIS: digital/AI/SME slice (score = keywords + CL4 + SME flag), cap 400 → `data/projects.json`.
- [x] Block B — clean/normalize folded into loaders (HTML strip, whitespace, dedupe, source ids).
- [x] Block C — chunk + embed (multilingual-e5-base) + index → Chroma `data/chroma/` (calls, projects).
- [x] Block D — sanity query on the `calls` collection.

## Gate report

### 1. `data/calls.json` + `data/projects.json` saved? — ✅ YES
- `data/calls.json` — **449** distinct OPEN EU grant topics. Sample: `ESC-HUMAID-2021-QUAL-LABEL-FP` "Quality Label Humanitarian Aid - Full Procedure", programme ESC, type_of_action "ESC Quality Label", 8 deadline cut-offs, full description + conditions, `source_url` → topicDetails JSON.
- `data/projects.json` — **400** CORDIS Horizon Europe projects, **392** SME-involved. Top-scored sample: `cordis-101135782` MANOLO "Trustworthy Efficient AI for Cloud-Edge Computing", topics `HORIZON-CL4-2023-HUMAN-01-01`, countries BE/DE/EL/ES/FI/FR/IE/RO, score 19, `source_url` → cordis.europa.eu/project/id/101135782.

### 2. Chroma persisted with `calls` + `projects`? — ✅ YES (verified from a fresh process)
- `data/chroma/` (PersistentClient, `hnsw:space=cosine`).
- `calls` — **1025** chunks (449 records). `projects` — **457** chunks (400 records).
- Every chunk carries `source_id` + `source_url` + `title` + programme/deadline/country metadata → citation-ready.

### 3. Embedding model + chunking
- **`intfloat/multilingual-e5-base`** (768-dim, multilingual; lighter than bge-m3). e5 prefixes: `passage:` on index, `query:` on search; raw text stored for clean citation; embeddings L2-normalized → cosine.
- **Chunking**: token-window, **480 max / 80 overlap**, under e5's 512 cap. Short objectives = single chunk; long call scope splits with overlap.

### 4. Sanity query — ✅ index works
Query: *"A German SME developing AI-based predictive maintenance for manufacturing"* → top-5 `calls`:
1. 0.808 — Industrial leadership in AI, Data and Robotics boosting competitiveness
2. 0.805 — Generative AI for smarter CCAM: enhancing perception, decision-making …
3. 0.803 — Approaches and tools for security in software and hardware development
4. 0.803 — Innovative technologies and solutions to improve wind energy systems
5. 0.802 — Towards a fair and transparent market for cultural and creative content

Top hits are squarely AI/Data/Robotics + industrial — index is retrieving sensibly. (e5 similarities sit in a narrow ~0.80 band by design.)

### 5. Blockers
- **None blocking.** Both corpora + index built end-to-end.
- Env: Python 3.14 can't run the embedding stack (no cp314 torch/chroma wheels) — solved with a 3.13 `venv-index`.
- Minor: SEDIA `totalResults` fluctuates run-to-run (452 today vs 1311 earlier) — caching on the EC index; the paginate-to-`totalResults` loop handles whatever the live count is. Conditions text is captured but intentionally not embedded (feeds Day-3 eligibility, not scope matching).

## STOP — Day 2 gate reached
Do NOT proceed to Day 3 (retrieval module + SME input schema). Awaiting go.
