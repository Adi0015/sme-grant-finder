# Day 1 — prove two data sources

**Goal:** prove I can pull CORDIS + SEDIA. NO RAG pipeline today. Stop at the gate.
See [[scope]] for the project framing, [[notes]] for repo summaries + data-source playbooks.

## Log

- [x] Setup: dirs, git init, venv (Python 3.14.6), `.gitignore` (venv/data/reference/.env)
- [x] Deps: pandas, requests only (embeddings/chroma deferred)
- [x] Block A: read cbrkit / ragold / hivegent → [[notes]]
- [x] C1: CORDIS download → `data/cordis_sample.json` (200 rows) + `src/load_cordis.py`
- [x] C2: SEDIA endpoint live-verified → `data/sedia_sample.json` (20 open topics) + `src/fetch_sedia.py`

## Gate report

### 1. `data/cordis_sample.json` — ✅ YES, 200 rows
- Source: `cordis-HORIZONprojects-csv.zip` (~34 MB, live). Main table `project.csv` (~21.5k rows).
- Required columns confirmed: `id`, `title`, `objective`(=description), `topics`, `frameworkProgramme`(=programme); `organisations` joined from `organization.csv` on `projectID`.
- Sample record: `101069359` "Full spectrum SOLar Direct Air Capture & conversion" (acronym SolDAC), topics `HORIZON-CL5-2021-D2-01-11`, 5+ organisations joined.

### 2. `data/sedia_sample.json` — ✅ YES, 20 distinct OPEN topics
- **Confirmed endpoint** (live 2026-06-23): `POST https://api.tech.ec.europa.eu/search-api/prod/rest/search?apiKey=SEDIA&text=***`, multipart body (`query`/`sort`/`languages` as `application/json` parts).
- Open grants filter → ~454 results today. Each record has all 6 required fields: `identifier`, `title`, `deadline`, `status`, `description`+`keywords`, `url`.
- Sample record: `CERV-2024-CITIZENS-VALUES` / `HORIZON-EIT-2023-25-KIC-EITURBANMOBILITY`, status `31094502` (Open).

### 3. Reference summaries — ✅ in [[notes]]
- **cbrkit**: retrieval engine; a "case" = generic `Mapping[K,V]` value, query same type; MAC/FAC tuple pipelines; `attribute_value` + aggregator for global similarity; BM25 + embedding + rerank.
- **ragold** ⭐: gold annotation format = `{query, queryType, relevantChunks[], distractingChunks[], response, notes}`, chunks grounded to source docs by uuid, loose schema (extensible). Reuse for our eval set.
- **hivegent**: framing = guided LLMs for non-experts; plan-before-execute approval, forced `<cite/>` citations, personality presets; cbrkit-on-pgvector retrieval.

### 4. Blockers
- **None blocking.** Both sources pulled and verified.
- Minor: CORDIS parser skips ~950 rows with embedded unescaped quotes (sample unaffected; revisit for full corpus). Some SEDIA "Open" topics have past nominal deadlines = rolling/continuously-open calls (status flag is authoritative).

## STOP — Day 1 gate reached
Do NOT proceed to Day 2 (corpus building / embeddings). Awaiting go.
