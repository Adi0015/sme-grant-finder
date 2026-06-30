"""Shared config + embedding/store helpers for the SME grant finder.

Single source of truth for the embedding model, the e5 task prefixes, the Chroma
location, collection names, and score orientation. Both the index builder
(build_index.py) and the retrieval module (retrieve.py) import from here so the
two sides can never drift (e.g. embed passages with one prefix, query with another).
"""

from __future__ import annotations

import os

# Set BEFORE torch / transformers / tokenizers import (config is imported by every
# entry point, and load_model() imports them lazily later, so this runs first).
# Guards the macOS "[Errno 1] Operation not permitted" (EPERM) class that hits when
# HuggingFace tokenizers / OpenMP fork after threads already exist — which is exactly
# what happens running the model inside Streamlit's ScriptRunner thread (the CLI works
# because it's single-threaded; Streamlit isn't). setdefault so the user can override.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # no fork-after-thread in tokenizers
os.environ.setdefault("OMP_NUM_THREADS", "1")             # don't spawn an OpenMP pool
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # tolerate duplicate libomp on macOS

from functools import lru_cache
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CALLS_JSON = DATA / "calls.json"
PROJECTS_JSON = DATA / "projects.json"
CHROMA_DIR = DATA / "chroma"

# ── Embedding model ─────────────────────────────────────────────────────────────
# intfloat/multilingual-e5-base: multilingual (DE SME text + EU call text), 768-dim.
# e5 REQUIRES task prefixes — "passage: " on the index side, "query: " on the search
# side. Applied centrally here (embed_passages / embed_query) so the convention is
# defined exactly once.
MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
EMBED_BATCH = 64

# Chunking (used by build_index.py). e5-base hard window = 512 tokens; stay under it.
CHUNK_MAX_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 80

# ── Vector store ────────────────────────────────────────────────────────────────
COLLECTION_CALLS = "calls"
COLLECTION_PROJECTS = "projects"
DISTANCE = "cosine"   # hnsw:space; pairs with L2-normalized embeddings


@lru_cache(maxsize=1)
def load_model():
    """Load the SentenceTransformer once per process (cached)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_passages(texts: list[str]):
    """Embed index-side text with the e5 'passage:' prefix. Returns numpy array."""
    model = load_model()
    return model.encode(
        [PASSAGE_PREFIX + t for t in texts],
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def embed_query(text: str):
    """Embed a query with the e5 'query:' prefix. Returns a 1-D numpy vector."""
    model = load_model()
    return model.encode(
        [QUERY_PREFIX + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


@lru_cache(maxsize=1)
def get_client():
    """Persistent Chroma client at CHROMA_DIR (cached)."""
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def similarity_from_distance(distance: float) -> float:
    """Chroma returns a cosine *distance* (= 1 - cosine_similarity) when the
    collection space is 'cosine'. Invert it so higher = more similar = better.
    With L2-normalized vectors the result is the cosine similarity in [-1, 1]."""
    return 1.0 - float(distance)


# ── LLM (Day 4 generation layer) ────────────────────────────────────────────────
# Backend is swappable by ONE constant. Default is local Ollama (no API key, no
# paid hosted service). "anthropic" is available but is a PAID hosted API — only
# selected deliberately, never silently. llm.py reads these.
LLM_BACKEND = "ollama"               # "ollama" | "anthropic"

# Ollama (local server proxies any pulled model; cloud-tagged models need a paid
# subscription, so prefer a locally pulled model e.g. "qwen2.5:7b", "llama3.1:8b").
# Host/model are env-overridable so a container can point at the host's Ollama
# (e.g. OLLAMA_HOST=http://host.docker.internal:11434) without code changes.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Anthropic (paid hosted API — requires ANTHROPIC_API_KEY in the environment).
ANTHROPIC_MODEL = "claude-sonnet-4-6"

LLM_TEMPERATURE = 0.0                # deterministic extraction, not creative writing
LLM_TIMEOUT = 240                    # seconds per call
MAX_CONDITIONS_CHARS = 4500          # cap the call's conditions text passed to the LLM
