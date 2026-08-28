# Prompt Log

A record of the prompts used with Claude Code (and any other AI assistance)
while building this project, for transparency in the placement writeup.
Add a new entry every time you use an AI tool to write or meaningfully
modify code -- include what you asked for, roughly what you got, and what
(if anything) you changed afterward.

---

## Entry 1 -- 2026-08-28 -- Initial project scaffold

**Tool:** Claude Code

**Prompt (summarized):** Build a full end-to-end system for an AI/ML
placement assignment: read `Facets_Assignment.csv` (399 personality/
behavioural facets), build a pipeline that scores relevant facets from a
short conversation using a *local* Ollama `llama3.1` model, on a 1-5 integer
scale, with a hard requirement to retrieve relevant facets first (FAISS +
sentence-transformers) rather than ever sending all 399 facets in one
prompt, batch LLM scoring at 10 facets per call, and abstain
(`insufficient_evidence`) when the conversation doesn't support a judgment.
Specifically requested:
- Project structure (`data/`, `outputs/`, `notebooks/`, `src/`, `docs/`,
  `requirements.txt`, `main.py`)
- `src/audit.py` -- clean/classify the raw CSV into 7 categories, flag
  conversation-observability, sensitivity, abstention reasons, scoring
  anchors; save `outputs/enriched_facets.csv`
- `src/embeddings.py` -- FAISS index over observable facets only,
  `retrieve_relevant_facets(conversation, top_k=25)`
- `src/scorer.py` -- batched Ollama scoring, robust JSON parsing, hard
  block on medical facet hallucination
- `src/pipeline.py` -- wires retrieval -> batching -> scoring -> dedupe ->
  saved JSON output
- `src/benchmark.py` -- 10 hand-written conversations (clear, ambiguous,
  contradictory, sarcastic, low-evidence, code-switched, medical trap,
  spiritual trap, high-emotion, professional) with >=3 hand-labeled
  reference facets each, compared against actual pipeline output
- `main.py` -- CLI with `--audit`, `--embed`, `--score`, `--benchmark`,
  `--setup`
- Starter docs (this file, README, DECISIONS, DEBUGGING)

**What was produced:** the full scaffold above, all files populated with
working (not stub) implementations, run against the actual
`data/Facets_Assignment.csv` on disk (399 rows, confirmed facet names like
`Risktaking`, `FSH level`, `Basophil count`, `I Ching hexagram <n>
resonance level`, `Compassion`, `Patience: Resistance to anger` used
directly in the benchmark reference set after grepping the real CSV to
confirm they exist verbatim).

**What was changed afterward:** See Entry 2 -- ran the actual pipeline
against my real CSV/Ollama setup, found a real retrieval issue via the
benchmark, and made one concrete change (`top_k` 25 -> 40) based on that
evidence rather than tweaking anything blind.

---

## Entry 2 -- 2026-08-28 -- Diagnosing and tuning the retrieval step

**Tool:** Claude (used both claude.ai to originally draft the assignment
prompt in Entry 1, and Claude Code to build/run it)

**What I did:** Went through the code Claude Code generated file by file
before trusting it -- didn't just take "it works" on faith. Ran the full
pipeline myself in order: `--setup` (confirmed Ollama + llama3.1 were
already up), `--audit` (checked `outputs/enriched_facets.csv` and confirmed
`FSH level` / `Basophil count` landed in `medical_biological` and `Types of
Mindfulness Techniques Used` landed in `header_or_malformed`, both correctly
marked not observable), `--embed`, then `--benchmark`.

First benchmark run (`top_k=25`) came back 17/30, with `retrieval_miss: 13`
as the dominant failure mode and, importantly, 0 safety violations -- no
medical/spiritual/malformed facet ever got scored. I pulled the actual
missed facet list out of `outputs/benchmark_report.json` myself (`Emotionalism`,
`Self-improvement`, `Peacefulness`, `Decency`, `Doggedness`, `Common-sense`,
etc.) instead of just reading the summary counts, to understand *what kind*
of facets were being missed.

I asked Claude Code to bump `top_k` to 40 in `src/pipeline.py` and re-run
the benchmark to see if a wider retrieval window would fix it. It only
recovered 1 of the 13 misses (18/30 total) -- the other 12 were still
missing even with 15 extra slots out of 322 candidates. That told me the
real problem is the embedding text quality (generic templated scoring
anchors don't read like natural conversation), not the window size.

**Decision I made:** Given 24 hours total, I chose to keep `top_k=40` (free,
small improvement) and document the real root cause and the fact that I
diagnosed-but-didn't-fully-fix it, rather than spend remaining time
rebuilding the anchor-generation logic in `audit.py` under time pressure.
That reasoning is written up in full in `DECISIONS.md` (Decision 2) and
`DEBUGGING.md` (#1) -- I'd rather hand in a system with an honestly
diagnosed, understood limitation than one where I silently patched the
symptom and hoped nobody asked why `Emotionalism` still gets missed
sometimes.
