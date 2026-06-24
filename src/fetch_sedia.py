"""Fetch ALL currently OPEN EU funding calls/topics from the SEDIA search API.

The EU Funding & Tenders Portal exposes its open calls through the EC "search-api"
(SEDIA index). No auth beyond the public `apiKey=SEDIA` is required.

Verified live 2026-06-23/24. Endpoint + request shape confirmed against working code
at github.com/ajruben/sedia-api-fetchers and by hitting it directly.

    POST https://api.tech.ec.europa.eu/search-api/prod/rest/search
         ?apiKey=SEDIA&text=***&pageSize=<n>&pageNumber=<p>
    body: multipart/form-data with three JSON blob parts: query, sort, languages

GOTCHA: every multipart JSON part MUST be sent with Content-Type 'application/json'.
A plain string part defaults to octet-stream and the API answers HTTP 500
("Content-Type 'application/octet-stream' is not supported"). With `requests` this
means each part is a (filename, body, content_type) tuple — see the `files` dict.

Status codes (verified from live facet data):
    31094501 = Forthcoming    31094502 = Open    31094503 = Closed
Type codes:
    "0" = tenders            "1","2","8" = grants (calls for proposals)

Metadata key map (learned by probing a live record, Day 2):
    identifier      -> metadata.identifier[0]
    title           -> metadata.title[0]  (fallback callTitle[0] / summary)
    description      -> metadata.descriptionByte[0]   (HTML — strip + unescape)
    conditions      -> metadata.topicConditions[0]    (HTML eligibility text)
    keywords/tags   -> metadata.keywords[], metadata.tags[], metadata.cenTagsA[]
    deadline(s)     -> metadata.deadlineDate[]  (ARRAY — one per cut-off)
    deadline model  -> metadata.deadlineModel[0]
    status          -> metadata.status[0]
    programme       -> metadata.frameworkProgramme[0] (an EC id) + programmePeriod[0]
    type of action  -> metadata.typesOfAction[0]
    canonical url   -> result.url

Day 2: pulls every OPEN grant topic (paginates to exhaustion), keeps the FULL
description (no truncation — we cite from it later), strips HTML, dedupes by
identifier, and stamps each record with source_id / source_url / source_type.

Usage:
    python src/fetch_sedia.py            # writes ALL open topics to data/calls.json
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calls.json"

SEARCH_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
API_KEY = "SEDIA"  # public portal key, not a secret

STATUS_OPEN = "31094502"
STATUS_LABELS = {
    "31094501": "Forthcoming",
    "31094502": "Open",
    "31094503": "Closed",
}
GRANT_TYPES = ["1", "2", "8"]  # calls for proposals (exclude tenders / type 0)

PAGE_SIZE = 100
MAX_PAGES = 60  # safety cap (~6000 topics) so a bad totalResults can't loop forever

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _first(value):
    """SEDIA wraps most metadata values in a single-element list — unwrap it."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value if value is not None else ""


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)] if value != "" else []


def _strip_html(text: str) -> str:
    """Remove tags, decode HTML entities, normalize whitespace."""
    text = _TAG_RE.sub(" ", text or "")
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _fetch_page(page_number: int, page_size: int, types: list[str]) -> dict:
    query = {
        "bool": {
            "must": [
                {"terms": {"type": types}},
                {"terms": {"status": [STATUS_OPEN]}},
            ]
        }
    }
    sort = {"field": "deadlineDate", "order": "ASC"}
    # Each JSON part must be ('blob', <json>, 'application/json') or the API 500s.
    files = {
        "query": ("blob", json.dumps(query), "application/json"),
        "sort": ("blob", json.dumps(sort), "application/json"),
        "languages": ("blob", json.dumps(["en"]), "application/json"),
    }
    params = {
        "apiKey": API_KEY,
        "text": "***",  # wildcard match-all
        "pageSize": str(page_size),
        "pageNumber": str(page_number),
    }
    resp = requests.post(
        SEARCH_URL,
        params=params,
        files=files,
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_open_topics(page_size: int = PAGE_SIZE, types: list[str] | None = None) -> list[dict]:
    """Paginate the SEDIA search until every OPEN grant topic is collected.

    Stop condition is collected >= totalResults (robust to the API silently
    capping page_size lower than requested), with a hard MAX_PAGES backstop.
    """
    types = types or GRANT_TYPES
    first = _fetch_page(1, page_size, types)
    total = int(first.get("totalResults") or 0)
    results = list(first.get("results", []))
    print(f"totalResults (open grants): {total}")

    page = 2
    while len(results) < total and page <= MAX_PAGES:
        payload = _fetch_page(page, page_size, types)
        batch = payload.get("results", [])
        if not batch:
            break
        results.extend(batch)
        print(f"  page {page}: +{len(batch)} (collected {len(results)}/{total})")
        page += 1
    return results


def normalize(result: dict) -> dict:
    """Map a raw SEDIA result to a clean, citation-ready call record."""
    meta = result.get("metadata", {}) or {}

    identifier = _first(meta.get("identifier")) or result.get("reference", "")
    status_code = _first(meta.get("status"))
    url = result.get("url") or _first(meta.get("url"))

    # Full description (no truncation) + eligibility conditions, both HTML -> text.
    description = _strip_html(_first(meta.get("descriptionByte")))
    if not description:
        description = _strip_html(result.get("summary") or result.get("content") or "")
    conditions = _strip_html(_first(meta.get("topicConditions")))

    # keywords / tags from the several places SEDIA scatters them.
    keywords = _as_list(meta.get("keywords"))
    tags = _as_list(meta.get("tags")) + _as_list(meta.get("cenTagsA"))

    deadlines = _as_list(meta.get("deadlineDate"))
    programme_name = identifier.split("-")[0] if identifier else ""

    return {
        "source_id": identifier,
        "source_url": url,
        "source_type": "call",
        "identifier": identifier,
        "title": _first(meta.get("title")) or _first(meta.get("callTitle")) or result.get("summary", ""),
        "description": description,
        "conditions": conditions,
        "keywords": keywords,
        "tags": tags,
        "deadline": deadlines[0] if deadlines else "",
        "deadlines": deadlines,
        "deadline_model": _first(meta.get("deadlineModel")),
        "status": status_code,
        "status_label": STATUS_LABELS.get(status_code, status_code),
        "programme": _first(meta.get("frameworkProgramme")),
        "programme_name": programme_name,
        "programme_period": _first(meta.get("programmePeriod")),
        "type_of_action": _first(meta.get("typesOfAction")),
        "type": _first(meta.get("type")),
        "call_identifier": _first(meta.get("callIdentifier")),
    }


def dedupe(records: list[dict]) -> list[dict]:
    """A topic with several deadline cut-offs can return >1 row; keep the first."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = r["source_id"]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def drop_empty(records: list[dict]) -> list[dict]:
    """Drop records with no usable text to embed/cite."""
    return [r for r in records if (r.get("title") or r.get("description"))]


def main() -> None:
    raw = fetch_all_open_topics()
    records = drop_empty(dedupe([normalize(r) for r in raw]))
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} distinct OPEN call topics -> {OUT.relative_to(ROOT)}")

    if records:
        sample = dict(records[0])
        sample["description"] = (sample["description"][:300] + " ...") if sample["description"] else ""
        sample["conditions"] = (sample["conditions"][:120] + " ...") if sample["conditions"] else ""
        print("\nSample record (description/conditions truncated for display):")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
