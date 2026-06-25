"""Retrieval core: SME profile in -> ranked matching calls + supporting evidence.

    retrieve(profile) -> {"matched_calls": [...], "evidence_projects": [...]}

Pure and importable: no LLM, no I/O side effects beyond reading the persisted
Chroma store. Every returned item keeps source_id + source_url + title so the
Day-4 generator can cite it. Scores are oriented so HIGHER = better (cosine
similarity; see config.similarity_from_distance).

Chunk -> parent dedupe: calls (and occasionally projects) are stored as several
chunks. We over-fetch chunks, then collapse to one entry per source_id keeping
that source's BEST-scoring chunk, and rank sources by that best score.
"""

from __future__ import annotations

import config
from schema import SMEProfile

# Over-fetch factor: query this many chunks per requested result so that, after
# collapsing chunks -> parents, we still have >= k distinct sources to rank.
OVERSAMPLE = 5
MIN_FETCH = 50


def _query_chunks(collection_name: str, query_vec, n: int) -> list[tuple]:
    """Return [(id, document, metadata, distance), ...] sorted best-first."""
    coll = config.get_client().get_collection(collection_name)
    res = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ))


def _dedupe_to_parents(rows: list[tuple]) -> dict[str, dict]:
    """Keep the best chunk per source_id. rows are already sorted best-first, so the
    first time we see a source_id is its best chunk."""
    best: dict[str, dict] = {}
    for cid, doc, md, dist in rows:
        sid = md.get("source_id", cid)
        if sid in best:
            continue
        best[sid] = {"doc": doc, "md": md, "score": config.similarity_from_distance(dist)}
    return best


def retrieve_calls(profile: SMEProfile, k: int = 10) -> list[dict]:
    """Top-k OPEN calls matching the SME profile, one entry per call."""
    qv = config.embed_query(profile.to_query_text())
    rows = _query_chunks(config.COLLECTION_CALLS, qv, n=max(k * OVERSAMPLE, MIN_FETCH))
    best = _dedupe_to_parents(rows)

    # NOTE: we deliberately do NOT hard-filter on profile.country / trl / budget.
    # Eligibility is Day-4's job and stays assistive — structured fields ride along
    # in the payload (and weighted the query text) but never drop a semantic match.
    entries = []
    for sid, b in best.items():
        md = b["md"]
        entries.append({
            "source_id": sid,
            "title": md.get("title", ""),
            "source_url": md.get("source_url", ""),
            "deadline": md.get("deadline", ""),
            "programme": md.get("programme_name") or md.get("programme", ""),
            "type_of_action": md.get("type_of_action", ""),
            "best_score": round(b["score"], 4),
            "best_chunk_text": b["doc"],
        })
    entries.sort(key=lambda e: e["best_score"], reverse=True)
    return entries[:k]


def retrieve_evidence(profile: SMEProfile, k: int = 5) -> list[dict]:
    """Top-k past funded projects similar to the SME — supporting context, not
    match targets ('projects like yours that were funded')."""
    qv = config.embed_query(profile.to_query_text())
    rows = _query_chunks(config.COLLECTION_PROJECTS, qv, n=max(k * OVERSAMPLE, MIN_FETCH))
    best = _dedupe_to_parents(rows)

    entries = []
    for sid, b in best.items():
        md = b["md"]
        entries.append({
            "source_id": sid,
            "title": md.get("title", ""),
            "source_url": md.get("source_url", ""),
            "programme": md.get("programme", ""),
            "country": md.get("countries", ""),
            "score": round(b["score"], 4),
            "chunk_text": b["doc"],
        })
    entries.sort(key=lambda e: e["score"], reverse=True)
    return entries[:k]


def retrieve(profile: SMEProfile, k_calls: int = 10, k_evidence: int = 5) -> dict:
    """Top-level entry point. Returns JSON-serializable dicts."""
    return {
        "query": profile.to_dict(),
        "matched_calls": retrieve_calls(profile, k=k_calls),
        "evidence_projects": retrieve_evidence(profile, k=k_evidence),
    }
