# Prompt Log

A record of the prompts used with AI tools while building this project, for
transparency in the placement writeup. One entry per session/prompt that
produced or changed real code, including the actual prompt text (not a
paraphrase), what I kept vs. changed vs. rejected from what came back, and
-- separately -- concrete examples of where the AI's output was actually
wrong and I had to catch it myself.

---

## Entry 1 -- 2026-08-28 -- Initial project build

**Tool:** Claude Code (Anthropic's CLI-based coding agent)

**The actual prompt** (this is the real text I gave it, not a summary):

> I am a student building an AI/ML assignment for a company placement. I
> have 24 hours. Here is the full assignment:
>
> ASSIGNMENT SUMMARY:
> - I have a CSV file called "Facets_Assignment.csv" with 399
>   personality/behavioural facets (traits like Risktaking, Compassion, FSH
>   level, I Ching hexagram scores, etc.)
> - I need to build a system that reads a short conversation and scores
>   relevant personality facets from it
> - The system must NOT send all 399 facets in one LLM prompt -- it must
>   retrieve only relevant ones first, then score in batches
> - Use Ollama running locally with llama3.1 model (http://localhost:11434)
> - Score scale: 1 to 5 (ordered integer levels)
> - System must abstain (say "insufficient_evidence") when a facet cannot
>   be judged from conversation alone
>
> MY SETUP:
> - Acer Predator laptop, RTX 4050 6GB VRAM, 16GB RAM, i5 13th gen
> - Ollama installed locally, will pull llama3.1
> - Python project
> - Facets CSV is at: Facets_Assignment.csv (one column called "Facets",
>   399 rows)
>
> WHAT I NEED YOU TO BUILD -- in this order: [project folder structure;
> requirements.txt with pandas/numpy/sentence-transformers/faiss-cpu/
> ollama/pydantic/rich/tqdm/scikit-learn/jupyter; src/audit.py to clean and
> classify the 399 facets into 7 categories with observability/sensitivity/
> abstention_reason/scoring_anchors columns; src/embeddings.py to build a
> FAISS index over observable facets only and expose
> retrieve_relevant_facets(conversation_text, top_k=25); src/scorer.py to
> batch-score at most 10 facets per Ollama call with strict JSON output and
> a hard block on medical-facet hallucination; src/pipeline.py to wire
> retrieval -> batching -> scoring -> dedupe -> saved JSON; src/benchmark.py
> with 10 hand-written conversations (clear, ambiguous, contradictory,
> sarcastic, low-evidence, code-switched, medical trap, spiritual trap,
> high-emotion, professional) each with >=3 hand-labeled reference facets,
> compared against actual pipeline output; main.py CLI with --audit,
> --embed, --score, --benchmark, --setup; and starter docs (README,
> DECISIONS, DEBUGGING, PROMPT_LOG)]
>
> IMPORTANT INSTRUCTIONS FOR YOU (Claude Code): Build everything in one go,
> file by file. Add clear comments in every function explaining what it
> does and why. Make the code robust -- try/except everywhere the LLM or
> file IO is involved. After building, tell me exactly what commands to run
> in what order to test it. Flag any place where I need to manually fill
> something in.

**What I used from it:** Essentially the entire structure as specified --
the folder layout, all five `src/` modules with the exact responsibilities
listed, the CLI flags, and the 7-category classification scheme. I didn't
rewrite the architecture; the brief was detailed enough that there wasn't
much ambiguity at the structural level.

**What I changed or rejected:**
- The raw CSV file on disk was actually named `Facets_Assignment.csv.csv`
  (a double extension, probably from how I originally downloaded/saved it)
  sitting in `data/` instead of the exact path the brief assumed --
  Claude Code found this itself and renamed it before running anything, but
  I want to record that the CSV wasn't where the spec said it would be.
- Kept `top_k=25` as the starting value (matching the brief's example
  signature) but changed it to `top_k=40` after benchmarking showed it was
  undersized -- see Entry 2 and `DECISIONS.md` #1.
- Rejected the idea of fixing the embedding-text quality issue that the
  benchmark surfaced (richer per-facet example phrasings instead of the
  generic anchor template) -- diagnosed it, decided not to implement it
  given remaining time, and documented it as a known limitation instead of
  quietly leaving it unexplained.
- Kept the double-layer safety architecture (structural exclusion from the
  FAISS index *and* a hard block in `scorer.py`) that Claude Code built
  beyond the brief's minimum ask of "force abstention if any slip through"
  -- this was already more than what was strictly required, and I verified
  it actually holds (0 safety violations across two full benchmark runs)
  rather than just trusting that it would.

### What AI got wrong / what I corrected

1. **The `top_k=25` default was undersized for the actual data, and this
   wasn't obvious until I benchmarked it.** Claude Code picked 25 as the
   starting value (matching the brief's own example signature), but with
   322 observable facets in my actual CSV, running the benchmark showed 13
   out of 30 hand-labeled reference facets were never even retrieved. The
   AI's code was working exactly as written -- this wasn't a bug -- but the
   *parameter choice* it (and the brief) suggested turned out to be wrong
   for this dataset once I actually measured it instead of trusting it by
   inspection. I corrected it to `top_k=40` after confirming the improvement
   with a second full benchmark run.

2. **The generic templated scoring-anchor text it generates in `audit.py`
   hurts retrieval quality, and increasing `top_k` doesn't fix it.** Claude
   Code's `generate_scoring_anchor()` produces boilerplate rubric sentences
   like `"1=Very low X; 3=Moderate X; 5=Very high X clearly expressed..."`
   for every personality-trait facet, and these get embedded alongside the
   facet name for retrieval. I found, by comparing the retrieval-miss lists
   from my `top_k=25` and `top_k=40` benchmark runs, that 12 of the same 13
   facets missed retrieval at *both* settings -- meaning the problem isn't
   how many candidates get considered, it's that these facets rank low
   *regardless* of the cutoff, because the embedded anchor text doesn't
   resemble natural conversational language. This is a real design flaw in
   what the AI chose to embed, not something I would have caught without
   actually running the benchmark twice and diffing the results myself.

3. **The malformed-entry heuristic in `audit.py` is more aggressive than it
   should be, and I only found this by querying the output directly.** The
   rule Claude Code wrote flags any facet whose raw text ends in `:` as
   `header_or_malformed`. When I checked `outputs/enriched_facets.csv` for
   specific entries, I found this swept up things like `Achievement
   Motivation:`, `Leadership Potential:`, and `HonestyHumility:` -- names
   that read like genuine personality/behavioral traits, not spreadsheet
   section headers, and got excluded from scoring entirely as a result. I
   decided this was an acceptable conservative trade-off rather than a bug
   to fix outright (see `DECISIONS.md` #2 for the full reasoning), but it's
   a concrete case where the AI's rule was too blunt on close inspection,
   and I had to go find that myself by spot-checking real rows instead of
   trusting the audit summary counts at face value.

---

## Entry 2 -- 2026-08-28 -- Diagnosing and tuning the retrieval step

**Tool:** Claude Code

**The actual prompt:**

> In src/pipeline.py, change top_k from 25 to 40. Then re-run the benchmark
> with python main.py --benchmark and tell me the new scores.

**What I used from it:** The one-line change and the re-run, exactly as
asked -- this was a small, targeted follow-up after I'd already looked at
the first benchmark report myself and decided widening `top_k` was worth
testing as a first hypothesis.

**What I changed or rejected:** Nothing about the change itself -- it's a
one-line edit. What I want to record here is the reasoning *behind* asking
for it: I had already pulled the specific `retrieval_miss` facet list out
of `outputs/benchmark_report.json` before asking for this change, so I knew
going in exactly which facets I was testing whether the fix would recover.
After seeing the result (1 of 13 recovered), I explicitly chose *not* to
ask for a bigger `top_k` or to chase a reranker -- the data already told me
the bottleneck was elsewhere (embedding text quality, not window size), so
throwing more `top_k` at it wouldn't have been worth the extra Ollama calls.

### What AI got wrong / what I corrected

Nothing about this specific one-line change was wrong -- it did exactly
what I asked. What's worth recording is that I didn't let the small
improvement (17/30 -> 18/30) get reported as if it solved the problem: I
kept the change (it's a real, free improvement) but insisted the docs
reflect that it's a partial mitigation with a diagnosed, unresolved root
cause, not a fix. That framing is mine, not something the AI proposed on
its own -- left to its own devices after a "better" benchmark score, the
natural next step would have been to declare success.
