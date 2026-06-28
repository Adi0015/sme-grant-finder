"""RAISE — SME grant finder: thin Streamlit UI over pipeline.run() (Day 6, Block A).

A single page. It does NOT reimplement any logic — it imports run() and renders the
ranked shortlist, making every fit/eligibility CITATION visible (source link + the exact
quoted snippet). That visible grounding is the whole pitch.

Run:
    venv-index/bin/streamlit run app.py
Demo (no LLM / no index needed — renders a cached sample shortlist):
    open "http://localhost:8501/?demo=1"   or click "Load demo result" in the sidebar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from schema import SMEProfile          # noqa: E402

DEMO_FILE = ROOT / "data" / "samples" / "demo_shortlist.json"

st.set_page_config(page_title="RAISE — SME Grant Finder", page_icon="🔎", layout="centered")

_MEET = {
    "yes": ("✅", "appears to match"),
    "unclear": ("❔", "unclear — verify"),
    "not_stated": ("➖", "not stated in call"),
}


def _profile_from_inputs() -> SMEProfile:
    kws = [k.strip() for k in (st.session_state.get("keywords") or "").split(",") if k.strip()]
    return SMEProfile(
        description=st.session_state.get("description", "").strip(),
        country=(st.session_state.get("country") or "").strip() or None,
        org_type=(st.session_state.get("org_type") or "").strip() or None,
        budget=st.session_state.get("budget") or None,
        trl=st.session_state.get("trl") or None,
        keywords=kws or None,
    )


def _render_call(i: int, c: dict) -> None:
    score = c.get("score")
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
    with st.container(border=True):
        st.markdown(f"### {i}. {c.get('title', '(untitled)')}")
        meta = " · ".join(x for x in [
            f"**match** {score_s}",
            f"**deadline** {(c.get('deadline') or 'rolling/none')[:10]}",
            (c.get("type_of_action") or c.get("programme") or ""),
        ] if x)
        st.caption(meta)
        if c.get("source_url"):
            st.markdown(f"[Open the official call ↗]({c['source_url']})")

        if c.get("fit_summary"):
            st.markdown(f"**Why it may fit:** {c['fit_summary']}")
        for r in c.get("fit_reasons", []):
            with st.expander(f"↳ fit evidence: {r.get('claim', '')[:80]}"):
                st.markdown(f"> {r.get('snippet', '')}")
                if r.get("source_url"):
                    st.markdown(f"[source ↗]({r['source_url']})  `{r.get('source_id','')}`")

        elig = c.get("eligibility", [])
        st.markdown(f"**Eligibility (draft — {len(elig)} stated condition(s)):**")
        for e in elig:
            icon, label = _MEET.get(e.get("applicant_appears_to_meet", "not_stated"), ("➖", ""))
            with st.expander(f"{icon} {e.get('condition', '')[:80]}  —  _{label}_"):
                st.markdown(f"> {e.get('snippet', '')}")
                if e.get("source_url"):
                    st.markdown(f"[source ↗]({e['source_url']})  `{e.get('source_id','')}`")

        mi = c.get("missing_info", [])
        if mi:
            st.markdown("**To verify in the official documents:**")
            st.markdown("\n".join(f"- {m}" for m in mi))


def _render_results(result: dict) -> None:
    shortlist = result.get("shortlist", [])
    if not shortlist:
        st.info("No matching open calls found for this description. Try broader wording.")
        return
    st.success(f"{len(shortlist)} candidate calls (ranked by semantic match).")
    for i, c in enumerate(shortlist, 1):
        _render_call(i, c)

    ev = result.get("evidence_projects", [])
    if ev:
        with st.expander(f"Similar funded projects ({len(ev)}) — evidence this kind of work gets funded"):
            for e in ev:
                st.markdown(f"- [{e.get('title','')[:90]}]({e.get('source_url','')}) "
                            f"· {e.get('country','')} · match {e.get('score','')}")


# ── Page ──────────────────────────────────────────────────────────────────────────
st.title("🔎 RAISE — SME Grant Finder")
st.caption("Describe your project → matching **open EU funding calls** with grounded, "
           "cited fit + draft eligibility.")
st.warning("Decision-support draft — eligibility is **indicative**, verify against the "
           "official call documents. Not a legal eligibility verdict.")

with st.sidebar:
    st.header("Your project")
    st.text_area("Project description", key="description", height=160,
                 placeholder="e.g. A German SME building AI predictive-maintenance "
                             "software for manufacturing lines …")
    st.text_input("Country (ISO/name)", key="country", placeholder="DE")
    st.text_input("Organisation type", key="org_type", placeholder="SME / startup")
    st.number_input("Indicative budget (EUR)", key="budget", min_value=0.0, step=10000.0, value=0.0)
    st.number_input("TRL (1–9)", key="trl", min_value=0, max_value=9, step=1, value=0)
    st.text_input("Keywords (comma-separated)", key="keywords",
                  placeholder="predictive maintenance, machine learning")
    go = st.button("🔎 Find calls", type="primary", use_container_width=True)
    demo = st.button("▶︎ Load demo result (no LLM)", use_container_width=True)

# Demo mode: ?demo=1 in the URL, or the sidebar button — renders a cached sample shortlist
# so the UI works with no Ollama / no built index (great for a first look or a screenshot).
if demo or st.query_params.get("demo"):
    if DEMO_FILE.exists():
        _render_results(json.loads(DEMO_FILE.read_text(encoding="utf-8")))
    else:
        st.error(f"Demo file missing: {DEMO_FILE.relative_to(ROOT)}")
elif go:
    if not st.session_state.get("description", "").strip():
        st.error("Please enter a project description.")
    else:
        from pipeline import run          # imported lazily so demo mode needs no ML deps
        try:
            with st.spinner("Retrieving calls and generating grounded fit + eligibility …"):
                result = run(_profile_from_inputs(), top_n=5)
            _render_results(result)
        except Exception as exc:                                  # noqa: BLE001
            st.error(f"Generation failed: {exc}")
            st.caption("Is the index built (`python src/build_index.py`) and Ollama "
                       "running with the model pulled? See the README.")
else:
    st.info("Fill in your project on the left and hit **Find calls** — or **Load demo "
            "result** to see a cached example with no setup.")
