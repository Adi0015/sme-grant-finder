# Notes — sme-grant-finder

External brain for the project. Reference-repo summaries + data-source playbooks.
See [[scope]] for what this tool is (and is not). See [[day1]] for the Day 1 log + gate.

---

## Reference repos (RAISE team code)

_Cloned into `reference/` (gitignored). Read for how the RAISE group builds retrieval, eval, and guided LLM UX._

### cbrkit (wi2trier/cbrkit) — similarity, retrieval, the case model

- **What it is**: Modular, fully-typed Case-Based Reasoning toolkit for Python (`pyproject.toml` v1.5.2, MIT). ICCBR 2024 Best Student Paper. Covers all four CBR phases: retrieve, reuse, revise, retain — plus synthesis (RAG over retrieved cases) and eval.
- **A "case" and "casebase"**: deliberately generic — `type Casebase[K, V] = Mapping[K, V]` (`src/cbrkit/typing.py:235`). A case is the value `V` (str, attribute dict, Pydantic model, dataclass, or graph); keys `K` identify cases. The **query has the same type `V`** as the cases. No dedicated Case class.
- **Unified phase abstraction**: every phase is a `CbrFunc[K,V,S]` (`typing.py:334`): `__call__(batches: Sequence[tuple[Casebase, query]]) -> Sequence[tuple[Casebase, SimMap]]`. `RetrieverFunc`/`ReuserFunc`/`ReviserFunc`/`RetainerFunc` are structurally identical — same build/apply/dropout pattern across phases.
- **Retrieval API** (`src/cbrkit/retrieval/__init__.py`):
  - `cbrkit.retrieval.build(sim_func)` → retriever (optional `multiprocessing=`).
  - `cbrkit.retrieval.dropout(retriever, limit=, min_similarity=, max_similarity=)` → filter/limit wrapper.
  - `apply_query(casebase, query, retriever)` (+ `apply_queries`, `apply_batches`, `*_indexed`, `*_async`).
  - **Pipeline composition**: pass a tuple of retrievers → sequential **MAC/FAC** (cheap pre-filter → expensive measure). Wrappers: `combine`, `distribute`, `transpose`, `persist`, `synced`/`threaded`, `chunk`.
- **Result** (`src/cbrkit/model/result.py`): `.similarities` (`{key: score}`), `.ranking` (keys sorted by score), `.casebase` (retrieved cases only).
- **Similarity measures** (`cbrkit.sim`): a measure is `sim = f(x, y)` (params **must** be named `x`=case, `y`=query — signature-inspected). Built-ins: `strings` (levenshtein, jaro, spacy, ngram), `numbers` (linear, exponential, threshold), `collections` (jaccard, A* mapping, dtw), `taxonomy` (wu_palmer), `graphs` (astar, vf2, ...), and `embed` (sentence-transformers/openai + cosine/dot/angular/euclidean, cached).
- **Global/aggregate similarity**: `cbrkit.sim.attribute_value(attributes={attr: measure}, aggregator=cbrkit.sim.aggregator(pooling="mean"))` — per-attribute local sims aggregated; **nestable**.
- **Advanced retrieval**: BM25 (`retrieval.indexable.bm25`), vector (`retrieval.indexable.embed`), persistent stores (lancedb/chromadb/pgvector/sqlite_vec), async re-rankers (cross_encoder/cohere/voyageai).

**Relevance to us**: this is the retrieval engine the RAISE stack actually uses (hivegent embeds it). Our "case" = a funding call; query = the SME free-text description. The MAC/FAC tuple pattern (cheap BM25 pre-filter → embedding rerank) is the retrieval design to mirror.

Minimal example (from README / `examples/cars_retriever.py`):

```python
import cbrkit
casebase = cbrkit.loaders.file("data/cars-1k.csv")          # Mapping[key -> case]
query = {"year": 2010, "make": "Toyota", "miles": 50000}
retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "year": cbrkit.sim.numbers.linear(max=50),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": cbrkit.sim.numbers.linear(max=1000000),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
    ),
    limit=5,
)
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
print(result.ranking, result.similarities)
```

### ragold (dfki-ebls/ragold) — gold-standard RAG eval annotation format ⭐

> **Most important for us** — reuse this schema for our eval set.

- **What it is**: client-only React 19 + Vite + Zustand web app for hand-building RAG eval datasets. Implements the **"Know Your RAG"** (COLING 2025 Industry) annotation framework. Metadata in localStorage, file bytes in IndexedDB, import/export as a zip.
- **Schema source of truth**: `src/lib/types.ts` (Zod v4 `looseObject`, `SCHEMA_VERSION = 2`), validated by `tests/schema.test.ts`. All objects are **loose** → unknown/future fields survive parsing (so we can add our own without breaking it).

**Gold record (one `Annotation`, keyed by uuid):**

| Field | Type | Meaning |
|-------|------|---------|
| `query` | string (required) | user question as typed into a chat system |
| `queryType` | string, default `fact_single` | `fact_single` \| `summary` \| `reasoning` \| `unanswerable` (free-form, extensible) |
| `relevantChunks` | `Chunk[]` | must-retrieve gold evidence (≥1 unless `unanswerable`) |
| `distractingChunks` | `Chunk[]` | hard negatives / misleading lookalikes (optional) |
| `response` | string (required) | expected/ideal gold answer |
| `notes` | string | annotator caveats |
| `createdAt` / `updatedAt` | ISO 8601 | timestamps |

- **`Chunk`** = `{ content: string, documentId?: string }` — UI forces `documentId` on any non-empty chunk, grounding it to an uploaded source `Document` by uuid.
- **`Document`** (metadata only) = `{ name, size(bytes, ≤10MB), notes, createdAt, updatedAt }`; bytes emitted to `files/<uuid>/<name>` in the export zip.
- **Envelope** (`annotations.json`) = `{ version, author, project, notes, language(en|de), createdAt, updatedAt, annotations{}, documents{} }`. `language` is the only strictly-validated field.

**`queryType` taxonomy** (from Know Your RAG): `fact_single` (answer is one fact in context), `summary` (answer needs summarizing several pieces), `reasoning` (answer derivable but not explicit), `unanswerable` (not present/derivable → `relevantChunks` forced to `[]`).

**Reuse notes for our eval set:**
- Relevance is **binary-by-bucket** (relevant vs distracting), NOT scored/ranked. No confidence or citation-span field — but loose schema means we can add `relevanceScore`, citation offsets, etc., and they survive.
- Two-tier uuid identity: annotation IDs + document IDs; chunks → documents via `documentId`.
- For us: `query` = SME project description; `relevantChunks` = the call-text spans that justify a match/eligibility; `distractingChunks` = lookalike calls; `response` = gold fit-summary + draft eligibility. This is exactly the citation-grounded shape our generator must hit.

### hivegent (dfki-ebls/hivegent) — framing: guided LLMs for non-experts

- **One-liner** (`README.md:3`): "Agentic system for non-expert users to interact with LLMs guided by experience." RAG-over-your-documents chat that **opinionates and scaffolds** the interaction so the user never writes a prompt or wires up tools.
- **Guidance is server-composed**, assembled per request in `server/routes/conversations.py` (~L390-420) from reusable blocks in `backend/src/hivegent/prompts.py`:
  - **Personalities** (`Personality` enum → `PERSONALITY_TEMPLATES`): user picks a behavior label (concise/detailed/structured) → backend supplies a full system prompt.
  - **Plan vs Execute mode** (`agents/registry.py`): plan mode is read-only, ends in a `create_plan` call ("Do not attempt any write operations") so the user **approves a plan before any mutation** — an explicit non-expert guardrail.
  - **Auto-injected rules** every turn: cite-your-sources (`<cite/>` with exact file + line), answer in the user's language, inline images, math. Grounded, traceable answers for free.
  - **Persistent memory** (`save_memory`): adapts across conversations = the "guided by experience" claim.
- **Agent architecture**: single Pydantic AI agent per request with user-scoped `UserDeps`, given a *filtered* set of toolset groups (`explore`, `subagent`, `write`, `memory`, `web`, `conversation`, `plan`). Subagents are shallow focused-exploration runs streaming back live. Backend is both an MCP client and an MCP server (FastMCP at `/mcp`).
- **Backend/frontend split** — thin client / fat server. Frontend: React 19 + Vite SPA, TanStack Router, Vercel AI SDK streaming. Backend: FastAPI + Pydantic AI = composition root for auth, retrieval, tools, model exec. **Retrieval = PostgreSQL + pgvector + cbrkit (dense/sparse/hybrid)**; filesystem is source of truth, SQL `documents`/`chunks` a derived index.

**Relevance to us**: the whole product is a guard-railed LLM layer for non-experts — forced citations, plan-before-execute approval, personality presets. Our "draft eligibility, not legal verdict" framing ([[scope]]) maps directly to its plan/approve + cite-everything model. And it stacks **cbrkit on pgvector** — confirming the retrieval engine choice.

---

## Data sources

### CORDIS — Horizon Europe projects (the reference corpus)

- **Bulk download** (live, no auth): `https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip` (~34 MB zip). From cordis.europa.eu/projects/en → "Download all Horizon Europe projects".
- **Format**: set of **`;`-separated** CSVs. Main table = `project.csv` (~21.5k rows, 21 cols). Organisations/topics are **relational** in sibling CSVs joined on `projectID`.
- **Column mapping** (spec name → real CORDIS column):
  - `id` → `id`, `title` → `title`, `description` → **`objective`**, `topics` → `topics`, `programme` → **`frameworkProgramme`**.
  - `organisations` → join `organization.csv` on `projectID` (list of participant names).
- **Loader**: `src/load_cordis.py` → confirms columns, joins orgs, writes `data/cordis_sample.json` (200 rows).
- **Gotcha**: ~950 rows skipped on parse (embedded unescaped quotes); use `engine="python"`, `on_bad_lines="warn"`. Fine for a sample; revisit if we need the full corpus.

### SEDIA — EU Funding & Tenders open calls (the live retrieval target) ⭐

- **Endpoint** (verified live 2026-06-23, HTTP 200): `POST https://api.tech.ec.europa.eu/search-api/prod/rest/search?apiKey=SEDIA&text=***`
  - `apiKey=SEDIA` is the **public** portal key (not a secret); `text=***` = wildcard match-all.
  - Body = **multipart/form-data** with three JSON parts: `query` (bool/must DSL), `sort`, `languages`.
- **⚠️ THE gotcha**: every multipart JSON part **must** carry `Content-Type: application/json`. A plain string part → octet-stream → **HTTP 500** "Content-Type 'application/octet-stream' is not supported". In `requests`: `files={'query': ('blob', json.dumps(q), 'application/json'), ...}`.
- **Status codes**: `31094501` = Forthcoming, **`31094502` = Open**, `31094503` = Closed. **Type codes**: `0` = tenders, `1`/`2`/`8` = grants (calls for proposals).
- **Open grants query**: `{"bool":{"must":[{"terms":{"type":["1","2","8"]}},{"terms":{"status":["31094502"]}}]}}` → ~454 open results today.
- **Field mapping** (metadata values are single-element arrays — index `[0]`):
  - `identifier` → `result.metadata.identifier[0]`
  - `title` → `result.metadata.title[0]` (fallback `callTitle[0]` / `summary`)
  - `deadline` → `result.metadata.deadlineDate[0]` (ISO-8601)
  - `status` → `result.metadata.status[0]`
  - `description/keywords` → `metadata.keywords` (string array) and/or `metadata.description[0]` (HTML — strip tags)
  - `url` → `result.url` (≡ `metadata.url[0]`)
- **Fetcher**: `src/fetch_sedia.py` → over-fetches, dedupes by `identifier` (multi-deadline topics repeat), writes `data/sedia_sample.json` (20 open topics).
- **Note**: some Open topics have **past nominal deadlines** — they're continuously/rolling open calls. Status flag is authoritative, not the date.
- **Reference impl** that pinned the request shape: `github.com/ajruben/sedia-api-fetchers`. A sibling `/facet` endpoint enumerates the status/type/programme code values.

---

## Day 2 — corpora + vector index

Built the full corpora and a persistent multilingual vector store. Code: `src/fetch_sedia.py` (calls), `src/load_cordis.py` (projects), `src/build_index.py` (chunk + embed + index). See [[day2]] for the gate report.

### Final counts
- `data/calls.json` — **449** distinct OPEN EU grant topics (SEDIA, all open grants paginated to exhaustion).
- `data/projects.json` — **400** CORDIS Horizon Europe projects (digital/AI/SME slice; **392** involve an SME).
- Chroma `data/chroma/` (persistent, cosine):
  - `calls` collection — **1025** chunks
  - `projects` collection — **457** chunks

### Embedding model — `intfloat/multilingual-e5-base` (why)
- **Multilingual**: inputs are German SME text + multilingual EU call text. e5-multilingual scores well on MTEB at modest size (~278M params, **768-dim**) vs the heavier `BAAI/bge-m3` (~2.2 GB).
- **Prefix handling (mandatory for e5)**: passages embedded as `"passage: <text>"`, queries as `"query: <text>"`. Raw text is stored in the Chroma documents (so we cite clean source); the prefix is added only at encode time.
- Embeddings **L2-normalized** → cosine space (`hnsw:space=cosine`).
- e5-base hard window = **512 tokens**; chunks stay under it (no silent truncation).

### Chunking params
- Token-window over the model tokenizer: **480 max tokens, 80 overlap** (under e5's 512 incl. the `passage:` prefix + special tokens).
- Short CORDIS objectives fall through as a single chunk; long call scope text splits with overlap (449 calls → 1025 chunks ≈ 2.3/call).
- Body embedded = `title. description` (call scope / project objective). Call **eligibility `conditions`** is kept in `calls.json` but **not indexed** — scope text drives matching; conditions feed Day-3 eligibility drafting.

### Collection schema (Chroma metadata is flat str/int/float/bool only)
- **Common**: `source_id`, `source_type`, `source_url`, `title`, `chunk_index`, `n_chunks`.
- **calls**: `identifier`, `programme`, `programme_name`, `programme_period`, `type_of_action`, `status_label`, `deadline`, `deadline_model`, `keywords` (csv).
- **projects**: `acronym`, `programme`, `topics`, `countries` (csv), `sme_involved` (bool), `relevance_score` (int).
- Chunk id = `f"{source_id}::{chunk_index}"`; document = raw chunk text (the citation source).

### Block B (clean/normalize) — folded into the loaders
- SEDIA: strip HTML tags + `html.unescape` + collapse whitespace; dedupe by identifier; drop empty.
- CORDIS: whitespace-normalize objective; drop stubs (<80 chars); dedupe by id.
- Every record stamped `source_id` / `source_url` / `source_type`; raw text kept intact (no aggressive truncation).

### Env note ⚠️
- torch / sentence-transformers / chromadb have **no cp314 wheels** → the Day-1 venv (Python **3.14**) can't run Block C. Built a second venv `venv-index` on **Python 3.13** (torch 2.12.1, sentence-transformers 5.6.0, chromadb 1.5.9, onnxruntime 1.27.0). All Day-2 code runs from it; `venv-index/` is gitignored.

### Corrections to the Day-1 SEDIA field map
- Description lives in `metadata.descriptionByte[0]` (HTML) — **not** `metadata.description`. `deadlineDate` is an **array** (one entry per cut-off). Programme is an EC id in `frameworkProgramme` + readable `programmePeriod`; readable family derived from the identifier prefix. Type of action = `typesOfAction[0]`.

### Harmless warning
- `transformers` prints "Token indices sequence length is longer than 512" while the tokenizer length-checks a long doc inside `chunk_text`; the doc is then windowed to ≤480 tokens before embedding. Expected, not an error.
