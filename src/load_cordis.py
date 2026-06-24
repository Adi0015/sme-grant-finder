"""Build the CORDIS Horizon Europe evidence corpus (data/projects.json).

Source: https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip
A set of ';'-separated CSVs. project.csv is the main table (~21.5k rows);
organisations are relational (organization.csv, joined on projectID).

Day 2: from the full ~21.5k projects we select a manageable, digital/AI/SME-relevant
slice (cap N_PROJECTS) so the evidence corpus mirrors what an SME grant-seeker would
care about. Relevance score per project:

    score = (# AI/digital keyword hits in title+objective+topics+keywords)
          + 3 if the project sits in the HORIZON CL4 cluster (Digital, Industry & Space)
          + 2 if at least one participating organisation is flagged SME in CORDIS

Records carry source_id / source_url / source_type plus organisations + countries so
chunks can be cited back to a real CORDIS project page.

Usage:
    python src/load_cordis.py            # writes data/projects.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cordis-horizon"
OUT = ROOT / "data" / "projects.json"

# Size cap for the evidence slice.
N_PROJECTS = 400
MIN_OBJECTIVE_LEN = 80          # drop stubs with no real text to embed/cite
MAX_ORGS_PER_PROJECT = 30       # bound record size
PROJECT_URL = "https://cordis.europa.eu/project/id/{id}"

# spec name -> real CORDIS column
REQUIRED = {
    "id": "id",
    "title": "title",
    "description": "objective",   # CORDIS calls the description "objective"
    "topics": "topics",
    "programme": "frameworkProgramme",
}

# Lowercase substrings marking digital / AI / SME-relevant work. Broad on purpose:
# they feed a score, and the cap keeps only the highest-scoring (most digital) slice.
DIGITAL_AI_KEYWORDS = [
    "artificial intelligence", " ai ", "ai-", "machine learning", "deep learning",
    "neural", "big data", "data-driven", "data driven", "digital", "software",
    "algorithm", "robot", "automation", "autonomous", "cyber", "iot",
    "internet of things", "cloud", "edge computing", "computing", "computer vision",
    "nlp", "natural language", "predictive", "analytics", "blockchain", "5g", "6g",
    "semiconductor", "sensor", "simulation", "digital twin", "small and medium",
    "manufacturing", "industry 4.0", "smart", "optimization", "optimisation",
    "high performance computing", "quantum",
]


def load_projects(raw_dir: Path = RAW) -> pd.DataFrame:
    """Read project.csv with the right separator/quoting, everything as str."""
    return pd.read_csv(
        raw_dir / "project.csv",
        sep=";",
        quotechar='"',
        dtype=str,
        keep_default_na=False,
        on_bad_lines="warn",
        engine="python",
    )


def load_organisations(raw_dir: Path = RAW):
    """Return (orgs_by_project, sme_project_ids).

    orgs_by_project: projectID -> list[{name, country, sme}]
    sme_project_ids: set of projectIDs with >=1 organisation flagged SME.
    """
    org = pd.read_csv(
        raw_dir / "organization.csv",
        sep=";",
        quotechar='"',
        dtype=str,
        keep_default_na=False,
        usecols=["projectID", "name", "country", "SME"],
        on_bad_lines="skip",
        engine="python",
    )
    orgs_by_project: dict[str, list[dict]] = {}
    sme_projects: set[str] = set()
    for row in org.itertuples(index=False):
        pid = row.projectID
        if not pid:
            continue
        is_sme = str(row.SME).strip().lower() == "true"
        orgs_by_project.setdefault(pid, []).append(
            {"name": row.name, "country": row.country, "sme": is_sme}
        )
        if is_sme:
            sme_projects.add(pid)
    return orgs_by_project, sme_projects


def confirm_columns(df: pd.DataFrame) -> None:
    missing = [src for src in REQUIRED.values() if src not in df.columns]
    if missing:
        raise SystemExit(f"MISSING required columns: {missing}")
    print(f"project.csv: {len(df)} rows; required columns confirmed.")


def relevance_score(row, sme_projects: set[str]) -> int:
    hay = " ".join([
        row.get("title", ""), row.get("objective", ""),
        row.get("topics", ""), row.get("keywords", ""),
    ]).lower()
    hits = sum(1 for kw in DIGITAL_AI_KEYWORDS if kw in hay)
    score = hits
    if "cl4" in (row.get("topics", "") + row.get("frameworkProgramme", "") +
                 row.get("masterCall", "") + row.get("subCall", "")).lower():
        score += 3
    if row["id"] in sme_projects:
        score += 2
    return score


def build_corpus(n: int = N_PROJECTS) -> list[dict]:
    df = load_projects()
    confirm_columns(df)
    orgs_by_project, sme_projects = load_organisations()
    print(f"organisations: {len(orgs_by_project)} projects mapped, "
          f"{len(sme_projects)} involve an SME.")

    scored: list[tuple[int, int, dict]] = []
    for _, row in df.iterrows():
        objective = row.get("objective", "")
        if len(objective) < MIN_OBJECTIVE_LEN:
            continue
        score = relevance_score(row, sme_projects)
        if score <= 0:
            continue
        scored.append((score, len(objective), row))

    # Highest score first; tie-break by longer (more substantive) objective. Stable.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    selected = scored[:n]

    records: list[dict] = []
    for score, _, row in selected:
        pid = row["id"]
        objective = row.get("objective", "")   # recompute per row — do NOT reuse loop-leaked var
        orgs = orgs_by_project.get(pid, [])
        names = [o["name"] for o in orgs if o["name"]][:MAX_ORGS_PER_PROJECT]
        countries = sorted({o["country"] for o in orgs if o["country"]})
        records.append({
            "source_id": f"cordis-{pid}",
            "source_url": PROJECT_URL.format(id=pid),
            "source_type": "project",
            "id": pid,
            "acronym": row.get("acronym", ""),
            "title": row.get("title", ""),
            "description": objective_clean(objective),
            "topics": row.get("topics", ""),
            "programme": row.get("frameworkProgramme", ""),
            "keywords": row.get("keywords", ""),
            "organisations": names,
            "countries": countries,
            "sme_involved": pid in sme_projects,
            "relevance_score": score,
        })
    return records


def objective_clean(text: str) -> str:
    """CORDIS objectives are plain text; just normalize whitespace."""
    return " ".join((text or "").split())


def main() -> None:
    records = build_corpus()
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    sme_n = sum(1 for r in records if r["sme_involved"])
    print(f"\nWrote {len(records)} projects -> {OUT.relative_to(ROOT)}  "
          f"({sme_n} involve an SME)")

    if records:
        sample = dict(records[0])
        sample["description"] = sample["description"][:300] + " ..."
        sample["organisations"] = sample["organisations"][:5]
        print("\nTop-scored sample record:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
