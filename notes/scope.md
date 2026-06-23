# Scope — sme-grant-finder

**Decision-support tool, not a legal verdict.**

An SME describes its project in free text → retrieve matching OPEN EU funding calls → generate a fit summary + DRAFT eligibility assessment per call, with citations to source text.

## What it is
- Retrieval over a corpus of EU funding calls (SEDIA) + reference projects (CORDIS).
- Source-grounded generation: every claim cites the call/topic text it came from.
- Draft eligibility & fit summary — a starting point for the SME to investigate.

## What it is NOT
- NOT a legal verdict on eligibility.
- NOT a guarantee of funding or compliance.
- NOT a substitute for reading the official call documents or consulting an advisor.

Output is framed as **draft / decision-support**. The human decides.

## Target
DFKI RAISE group — RAG for German SMEs. Mirror their work: retrieval, source-grounded generation, rigorous evaluation.
