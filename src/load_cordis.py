"""Load the CORDIS Horizon Europe projects bulk CSV and emit a small JSON sample.

Source: https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip
The bulk download is a set of ';'-separated CSVs. project.csv is the main table;
organisations are relational (organization.csv, joined on projectID).

Usage:
    python src/load_cordis.py            # prints columns, writes data/cordis_sample.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cordis-horizon"
OUT = ROOT / "data" / "cordis_sample.json"

# Columns we promise downstream consumers. Maps the spec's generic names to the
# real CORDIS column names.
REQUIRED = {
    "id": "id",
    "title": "title",
    "description": "objective",   # CORDIS calls the description "objective"
    "topics": "topics",
    "programme": "frameworkProgramme",
}

SAMPLE_FIELDS = [
    "id", "acronym", "status", "title", "objective", "topics",
    "frameworkProgramme", "fundingScheme", "startDate", "endDate", "keywords",
]


def load_projects(raw_dir: Path = RAW) -> pd.DataFrame:
    """Read project.csv with the right separator/quoting, everything as str."""
    df = pd.read_csv(
        raw_dir / "project.csv",
        sep=";",
        quotechar='"',
        dtype=str,
        keep_default_na=False,
        on_bad_lines="warn",
        engine="python",
    )
    return df


def organisations_by_project(raw_dir: Path = RAW) -> dict[str, list[str]]:
    """Map projectID -> list of participating organisation names."""
    org = pd.read_csv(
        raw_dir / "organization.csv",
        sep=";",
        quotechar='"',
        dtype=str,
        keep_default_na=False,
        usecols=["projectID", "name", "country", "role"],
        on_bad_lines="skip",
        engine="python",
    )
    out: dict[str, list[str]] = {}
    for pid, grp in org.groupby("projectID"):
        names = [n for n in grp["name"].tolist() if n]
        out[pid] = names
    return out


def confirm_columns(df: pd.DataFrame) -> None:
    print(f"project.csv: {len(df)} rows, {len(df.columns)} columns")
    print("columns:", list(df.columns))
    missing = [src for src in REQUIRED.values() if src not in df.columns]
    if missing:
        raise SystemExit(f"MISSING required columns: {missing}")
    print("\nRequired columns confirmed:")
    for generic, real in REQUIRED.items():
        print(f"  {generic:12s} -> '{real}'  OK")
    print("  organisations -> organization.csv (relational join on projectID)  OK")


def build_sample(n: int = 200) -> list[dict]:
    df = load_projects()
    confirm_columns(df)
    orgs = organisations_by_project()

    records: list[dict] = []
    for _, row in df.head(n).iterrows():
        rec = {f: row.get(f, "") for f in SAMPLE_FIELDS}
        rec["organisations"] = orgs.get(row["id"], [])
        records.append(rec)
    return records


def main() -> None:
    records = build_sample(200)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} records -> {OUT.relative_to(ROOT)}")
    print("\nSample record (truncated objective):")
    sample = dict(records[0])
    if sample.get("objective"):
        sample["objective"] = sample["objective"][:300] + " ..."
    sample["organisations"] = sample["organisations"][:5]
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
