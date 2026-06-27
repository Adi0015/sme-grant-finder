"""Orchestration: SME profile -> ranked, grounded shortlist (Day 4, Block C).

    run(profile, top_n=5) -> {
        "query": {...},
        "shortlist": [ <generate_for_call object>, ... ],   # ranked by retrieval score
        "evidence_projects": [ ... ],
    }

Side-effect-free and importable: Day-5 eval and Day-6 UI call run() directly. It
chains the Day-3 retriever (retrieve) with the Day-4 generator (generate_for_call),
passing each call the top evidence projects as optional context.
"""

from __future__ import annotations

import retrieve as retrieve_mod
from generate import generate_for_call
from schema import SMEProfile


def run(profile: SMEProfile, top_n: int = 5, k_evidence: int = 3,
        model: str | None = None) -> dict:
    retrieved = retrieve_mod.retrieve(profile, k_calls=top_n, k_evidence=k_evidence)
    evidence = retrieved["evidence_projects"]

    shortlist = []
    for call in retrieved["matched_calls"][:top_n]:
        shortlist.append(generate_for_call(profile, call, evidence=evidence[:2], model=model))

    # Calls arrive already ranked by retrieval score; keep that order (None-safe).
    shortlist.sort(key=lambda c: (c.get("score") is not None, c.get("score") or 0.0),
                   reverse=True)
    return {
        "query": profile.to_dict(),
        "shortlist": shortlist,
        "evidence_projects": evidence,
    }
