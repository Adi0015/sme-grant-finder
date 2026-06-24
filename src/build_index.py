"""Build the persistent vector index for the SME grant finder (Day 2, Block C+D).

Reads the two cleaned corpora (data/calls.json, data/projects.json), chunks the
text, embeds it with a MULTILINGUAL model, and writes two Chroma collections that
persist to data/chroma/. Every chunk keeps enough metadata (source_id + source_url
+ title + programme/deadline/country …) to be CITED back to its source later.

This file is the index BUILDER only. It deliberately does NOT implement the Day-3
retrieval module — `sanity_check()` is a one-off smoke test, not a reusable API.

EMBEDDING MODEL — intfloat/multilingual-e5-base
  Chosen because the inputs are bilingual+ (German SME descriptions, EU call text in
  many languages) and e5 has strong multilingual MTEB scores at a modest size
  (~278M params, 768-dim) vs the heavier BAAI/bge-m3 (~2.2GB).
  *** e5 REQUIRES task prefixes ***: every passage is embedded as "passage: <text>"
  and every query as "query: <text>". We store the RAW chunk text in Chroma (so we
  cite clean source text) and add the prefix only at encode time. e5-base also has a
  hard 512-token window, so CHUNK_MAX_TOKENS stays under it (no silent truncation).

Usage:
    python src/build_index.py            # (re)builds data/chroma/, prints counts + sanity
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CALLS_JSON = DATA / "calls.json"
PROJECTS_JSON = DATA / "projects.json"
CHROMA_DIR = DATA / "chroma"

MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "   # e5: index side
QUERY_PREFIX = "query: "       # e5: query side
EMBED_BATCH = 64

# e5-base max sequence length is 512 tokens; stay under it (leave room for the
# "passage: " prefix + special tokens) so nothing is silently truncated.
CHUNK_MAX_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 80

COLLECTION_CALLS = "calls"
COLLECTION_PROJECTS = "projects"
DISTANCE = "cosine"            # hnsw:space; pairs with normalized embeddings

TEST_SME = "A German SME developing AI-based predictive maintenance for manufacturing"


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """Sliding-window chunks over model tokens, with overlap. Short text -> [text]."""
    text = (text or "").strip()
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    step = max_tokens - overlap
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start:start + max_tokens]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True).strip())
        if start + max_tokens >= len(ids):
            break
    return [c for c in chunks if c]


# ── Metadata (Chroma allows only str/int/float/bool — flatten everything) ──────
def _csv(values) -> str:
    if isinstance(values, list):
        return ", ".join(str(v) for v in values if v not in (None, ""))
    return str(values or "")


def call_body(rec: dict) -> str:
    """Text we embed for a call: title + scope description (conditions live in
    calls.json for Day-3 eligibility drafting; we index scope for matching)."""
    title = rec.get("title", "").strip()
    desc = rec.get("description", "").strip()
    return f"{title}. {desc}".strip(". ").strip() if title else desc


def call_meta(rec: dict, idx: int, n: int) -> dict:
    return {
        "source_id": rec.get("source_id", ""),
        "source_type": "call",
        "source_url": rec.get("source_url", ""),
        "title": rec.get("title", ""),
        "chunk_index": idx,
        "n_chunks": n,
        "identifier": rec.get("identifier", ""),
        "programme": rec.get("programme", ""),
        "programme_name": rec.get("programme_name", ""),
        "programme_period": rec.get("programme_period", ""),
        "type_of_action": rec.get("type_of_action", ""),
        "status_label": rec.get("status_label", ""),
        "deadline": rec.get("deadline", ""),
        "deadline_model": rec.get("deadline_model", ""),
        "keywords": _csv(rec.get("keywords")),
    }


def project_body(rec: dict) -> str:
    title = rec.get("title", "").strip()
    desc = rec.get("description", "").strip()
    return f"{title}. {desc}".strip(". ").strip() if title else desc


def project_meta(rec: dict, idx: int, n: int) -> dict:
    return {
        "source_id": rec.get("source_id", ""),
        "source_type": "project",
        "source_url": rec.get("source_url", ""),
        "title": rec.get("title", ""),
        "chunk_index": idx,
        "n_chunks": n,
        "acronym": rec.get("acronym", ""),
        "programme": rec.get("programme", ""),
        "topics": rec.get("topics", ""),
        "countries": _csv(rec.get("countries")),
        "sme_involved": bool(rec.get("sme_involved", False)),
        "relevance_score": int(rec.get("relevance_score", 0)),
    }


# ── Build one collection ──────────────────────────────────────────────────────
def build_collection(client, model, name: str, records: list[dict],
                     body_fn, meta_fn) -> int:
    tokenizer = model.tokenizer
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for rec in records:
        chunks = chunk_text(body_fn(rec), tokenizer, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
        n = len(chunks)
        for i, ch in enumerate(chunks):
            ids.append(f"{rec['source_id']}::{i}")
            docs.append(ch)            # RAW text — what we cite from
            metas.append(meta_fn(rec, i, n))

    if not ids:
        print(f"  [{name}] no chunks — skipped")
        return 0

    # e5 passage prefix only at encode time; documents stay raw.
    embeddings = model.encode(
        [PASSAGE_PREFIX + d for d in docs],
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).tolist()

    # Idempotent: drop & recreate so re-runs don't duplicate.
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name, metadata={"hnsw:space": DISTANCE})
    coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"  [{name}] {len(records)} records -> {len(ids)} chunks indexed")
    return len(ids)


# ── Block D: sanity smoke test (NOT the Day-3 retrieval module) ────────────────
def sanity_check(client, model) -> None:
    print(f"\nSanity query (calls): {TEST_SME!r}")
    emb = model.encode([QUERY_PREFIX + TEST_SME], normalize_embeddings=True,
                       convert_to_numpy=True)[0].tolist()
    coll = client.get_collection(COLLECTION_CALLS)
    res = coll.query(query_embeddings=[emb], n_results=5,
                     include=["metadatas", "distances"])
    for rank, (md, dist) in enumerate(zip(res["metadatas"][0], res["distances"][0]), 1):
        print(f"  {rank}. sim={1 - dist:.3f}  {md['title'][:66]}")
        print(f"     [{md.get('type_of_action') or md.get('programme_name')}] {md['source_url']}")


def main() -> None:
    calls = json.loads(CALLS_JSON.read_text(encoding="utf-8"))
    projects = json.loads(PROJECTS_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(calls)} calls, {len(projects)} projects.")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  max_seq_length={model.max_seq_length}, dim={model.get_sentence_embedding_dimension()}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    print("Building collections ...")
    n_calls = build_collection(client, model, COLLECTION_CALLS, calls, call_body, call_meta)
    n_proj = build_collection(client, model, COLLECTION_PROJECTS, projects, project_body, project_meta)

    print(f"\nPersisted to {CHROMA_DIR.relative_to(ROOT)}/  "
          f"| calls={n_calls} chunks, projects={n_proj} chunks")
    sanity_check(client, model)


if __name__ == "__main__":
    main()
