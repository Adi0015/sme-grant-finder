"""Shared config + embedding/store helpers for the SME grant finder.

Single source of truth for the embedding model, the e5 task prefixes, the Chroma
location, collection names, and score orientation. Both the index builder
(build_index.py) and the retrieval module (retrieve.py) import from here so the
two sides can never drift (e.g. embed passages with one prefix, query with another).
"""

from __future__ import annotations

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
