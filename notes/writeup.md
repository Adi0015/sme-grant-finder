# RAISE grant-finder — a short writeup

I built this to see whether retrieval-augmented generation could do something genuinely
useful for a non-expert: take an SME's plain-language project description and hand back the
EU funding calls worth their time, with enough grounding that they can trust it. Here's what
I learned.

**What's actually hard.** The matching problem looks like search but isn't. An SME writes
"AI predictive maintenance for factories"; the relevant call is titled "Industrial
leadership in AI, Data and Robotics." There's little lexical overlap, so dense embeddings
are the right tool — but EU calls cluster tightly in embedding space (everything is
"AI for X"), so the similarity scores for great and mediocre matches sit in a narrow
0.80–0.84 band. A fixed threshold is useless; ranking is everything. The harder half is
*eligibility*: it lives in dense legal boilerplate ("Eligible countries described in section
6 of the call document") that points elsewhere rather than stating the rule. You can extract
the structure but rarely the actual criterion.

**Where RAG helped.** Retrieval plus chunk→call dedupe gave consistently on-domain top-3s
across ten test profiles. The generation layer is where it got interesting. My first
instinct — let the model write the "why it matches" quote and check it with fuzzy
word-overlap — was quietly wrong. A 7B model fabricated roughly half its fit quotes, and the
fuzzy matcher *hid* them: it snapped each invented sentence onto a thematically adjacent real
fragment and passed it. My code proudly reported "0 fabricated." An adversarial multi-agent
audit found five. That was the most useful failure of the project.

**Where it struggled.** Two places. First, faithfulness of *fit* claims is model-bound — the
local 7B simply isn't reliable at quoting, so I now re-derive every snippet to a verbatim
source span and drop anything that doesn't align. Honest, but it means half the calls show
no fit citation. Second, eligibility resists extraction: the faithful answer is usually "the
call states a condition here, but you must read it." I leaned into that rather than
papering over it — the tool never says "you are eligible," only "appears to match / unclear,"
and routes unknowns to a to-verify list. Multilingual capability is built in but
under-tested, since the corpus turned out English-dominant.

**What surprised me.** That the *evaluation* caught more than the build did. The deterministic
grounding check gave false confidence; it took an independent, adversarial pass to see the
fabrications. And the LLM-judge's modest 44% "fully supported" was misleading in the other
direction — most of the gap was faithful-but-low-information eligibility boilerplate, which a
human spot-check sorts out in minutes. Numbers need a human reading them.

**What I'd build next.** Better eligibility extraction (pull the actual country/TRL/budget
criteria, not the pointer), a stronger generator for fit quotes, and — closest to the RAISE
group's CBR work — experience-based config selection: log which retriever (dense vs hybrid)
wins per query type and pick it adaptively, instead of one fixed pipeline for every SME.
