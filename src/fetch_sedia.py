"""Fetch OPEN EU funding calls/topics from the SEDIA search API.

The EU Funding & Tenders Portal exposes its open calls through the EC "search-api"
(SEDIA index). No auth beyond the public `apiKey=SEDIA` is required.

Verified live 2026-06-23. Endpoint + request shape confirmed against working code at
github.com/ajruben/sedia-api-fetchers and by hitting it directly.

    POST https://api.tech.ec.europa.eu/search-api/prod/rest/search
         ?apiKey=SEDIA&text=***&pageSize=<n>&pageNumber=<p>
    body: multipart/form-data with three JSON blob parts: query, sort, languages

GOTCHA: every multipart JSON part MUST be sent with Content-Type 'application/json'.
A plain string part defaults to octet-stream and the API answers HTTP 500
("Content-Type 'application/octet-stream' is not supported"). With `requests` this
means each part is a (filename, body, content_type) tuple — see PART_CT below.

Status codes (verified from live facet data):
    31094501 = Forthcoming    31094502 = Open    31094503 = Closed
Type codes:
    "0" = tenders            "1","2","8" = grants (calls for proposals)

Usage:
    python src/fetch_sedia.py            # writes ~20 OPEN topics to data/sedia_sample.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sedia_sample.json"

SEARCH_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
API_KEY = "SEDIA"  # public portal key, not a secret

STATUS_OPEN = "31094502"
STATUS_LABELS = {
    "31094501": "Forthcoming",
    "31094502": "Open",
    "31094503": "Closed",
}
GRANT_TYPES = ["1", "2", "8"]  # calls for proposals

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")


def _first(value):
    """SEDIA wraps most metadata values in a single-element list — unwrap it."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value if value is not None else ""


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()


def fetch_open_topics(page_size: int = 20, types: list[str] | None = None) -> list[dict]:
    """Return raw SEDIA result records for OPEN grant calls."""
    types = types or GRANT_TYPES
    query = {
        "bool": {
            "must": [
                {"terms": {"type": types}},
                {"terms": {"status": [STATUS_OPEN]}},
            ]
        }
    }
    sort = {"field": "deadlineDate", "order": "ASC"}
    languages = ["en"]

    # Each JSON part must be ('blob', <json>, 'application/json') or the API 500s.
    files = {
        "query": ("blob", json.dumps(query), "application/json"),
        "sort": ("blob", json.dumps(sort), "application/json"),
        "languages": ("blob", json.dumps(languages), "application/json"),
    }
    params = {
        "apiKey": API_KEY,
        "text": "***",          # wildcard match-all
        "pageSize": str(page_size),
        "pageNumber": "1",
    }
    resp = requests.post(
        SEARCH_URL,
        params=params,
        files=files,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    print(f"totalResults (open grants): {payload.get('totalResults')}")
    return payload.get("results", [])


def normalize(result: dict) -> dict:
    """Map a raw SEDIA result to the 6 required fields (+ a couple of useful extras)."""
    meta = result.get("metadata", {}) or {}

    status_code = _first(meta.get("status"))
    # description/keywords: prefer the keywords array; fall back to HTML description.
    keywords = meta.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    description = _strip_html(_first(meta.get("description")) or _first(meta.get("descriptionByte")))
    if not description:
        description = _strip_html(result.get("summary") or result.get("content") or "")

    return {
        "identifier": _first(meta.get("identifier")) or result.get("reference", ""),
        "title": _first(meta.get("title")) or _first(meta.get("callTitle")) or result.get("summary", ""),
        "deadline": _first(meta.get("deadlineDate")),
        "status": status_code,
        "status_label": STATUS_LABELS.get(status_code, status_code),
        "keywords": keywords,
        "description": description[:1000],
        "url": result.get("url") or _first(meta.get("url")),
    }


def dedupe(records: list[dict]) -> list[dict]:
    """A topic with several deadline cut-offs returns one row per deadline; keep one."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = r["identifier"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    # Over-fetch then dedupe so we still land ~20 DISTINCT open topics.
    results = fetch_open_topics(page_size=40)
    records = dedupe([normalize(r) for r in results])[:20]
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} OPEN topics -> {OUT.relative_to(ROOT)}")

    if records:
        sample = dict(records[0])
        sample["description"] = (sample["description"][:300] + " ...") if sample["description"] else ""
        print("\nSample record:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
