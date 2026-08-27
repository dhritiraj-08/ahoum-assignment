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

**What was changed afterward:** [FILL IN -- once you've run `--audit`,
`--embed`, `--score`, and `--benchmark` yourself, note anything you had to
tweak: reclassified facets, adjusted keyword lists, changed batch size,
fixed a bug (cross-reference the matching `docs/DEBUGGING.md` entry), etc.]
