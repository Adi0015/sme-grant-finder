"""SME input schema for the grant finder.

An SMEProfile is the single input to retrieve(). to_query_text() composes its
fields into ONE retrieval string: the free-text description dominates (it carries
the real semantic signal), with the structured fields appended as a compact
context line.

The e5 'query:' prefix is NOT added here — it is applied once, centrally, by
config.embed_query(), so the index side and the query side can't drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SMEProfile:
    description: str                       # required free-text project description
    country: str | None = None            # ISO code or name, e.g. "DE"
    org_type: str | None = None           # e.g. "SME", "startup", "research org"
    budget: float | None = None           # indicative EUR
    trl: int | None = None                # Technology Readiness Level 1–9
    keywords: list[str] | None = None     # optional domain keywords

    def to_query_text(self) -> str:
        """Compose fields into a single retrieval string (no embedding prefix —
        config.embed_query() adds it). Description first so it carries the most
        weight; structured fields trail as context."""
        parts: list[str] = [self.description.strip()]
        ctx: list[str] = []
        if self.keywords:
            ctx.append("Keywords: " + ", ".join(self.keywords))
        if self.org_type:
            ctx.append(f"Organisation type: {self.org_type}")
        if self.country:
            ctx.append(f"Country: {self.country}")
        if self.trl is not None:
            ctx.append(f"Technology readiness level: TRL {self.trl}")
        if self.budget is not None:
            ctx.append(f"Indicative budget: EUR {self.budget:,.0f}")
        if ctx:
            parts.append(" | ".join(ctx))
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """JSON-serializable view (for echoing the query back in payloads)."""
        return asdict(self)
