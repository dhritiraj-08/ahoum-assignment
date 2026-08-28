# Conversation -> Personality Facet Scorer

![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![Model: Llama 3.1 8B](https://img.shields.io/badge/model-Llama%203.1%208B-green.svg)
![Tests: 42 passing](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)
![Scoring accuracy: 59%](https://img.shields.io/badge/scoring%20accuracy-59%25-yellow.svg)
![Retrieval recall: 26%](https://img.shields.io/badge/retrieval%20recall-26%25-red.svg)
![Hallucinations: 0](https://img.shields.io/badge/hallucinations-0%2F3-brightgreen.svg)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

**Scoring accuracy (59%) and retrieval recall (26%) are reported as two
separate numbers, not blended into one pass rate** -- see "Benchmark
methodology" below for why, and `outputs/benchmark_report.json` for the
real run these numbers come from.

Reads a short conversation and scores the personality/behavioural facets it
actually gives evidence for -- out of a 399-facet taxonomy -- using a local
LLM (Ollama + llama3.1), retrieval-first so the model never sees all 399
facets in one prompt, and abstaining explicitly ("insufficient_evidence")
whenever the conversation doesn't support a judgment.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1
python main.py --setup
```

`--setup` checks that the Ollama server is reachable at
`http://localhost:11434` and that `llama3.1` is pulled; it will attempt to
pull it for you if missing. On my machine (RTX 4050, 6GB VRAM) Ollama picks
up the GPU automatically -- no extra config needed -- and a full 10-batch
benchmark run finishes in a few minutes rather than being CPU-bound-slow.

## LLM Backend Options

`src/scorer.py` auto-detects which backend to use, in this order, every
time a conversation is scored (cached per-process after the first check):

- **Option A -- Local Ollama (default).** Private (nothing leaves your
  machine), no API key, no cost, but requires a GPU-capable machine with
  Ollama installed and `llama3.1` pulled. This is what every score/
  benchmark result in this repo's docs was produced with.
- **Option B -- Groq API (automatic cloud fallback).** Used automatically
  whenever Ollama isn't reachable *and* `GROQ_API_KEY` is set in the
  environment. Runs the same model family via Groq's hosted
  `llama-3.1-8b-instant`, so it works on any machine -- a laptop with no
  GPU, a CI runner, a grader's machine that doesn't have Ollama installed.

**How to set the key:**

```bash
# Linux/Mac
export GROQ_API_KEY=your_key_here
```

```powershell
# Windows PowerShell
$env:GROQ_API_KEY="your_key_here"
```

Get a free key at https://console.groq.com/keys. See `.env.example` for
the expected format if you're using a `.env` file / process manager
instead of exporting it directly. The Streamlit app (`app.py`) also lets
you paste a key directly into a password-masked field in the UI, shown
only when Ollama isn't detected -- see "Run the Streamlit UI" below.

If neither Ollama nor `GROQ_API_KEY` is available, `--score`/`--benchmark`/
`app.py` will fail fast with a clear message telling you which one to fix,
rather than hanging or producing a confusing low-level connection error.
See `DECISIONS.md` #5 for why this is automatic-and-silent rather than a
manual `--backend` flag, and the trade-offs that come with that choice.

## Running the pipeline (in order)

```bash
python main.py --audit       # clean/classify the 399 raw facets -> outputs/enriched_facets.csv
python main.py --embed       # build the FAISS retrieval index -> outputs/faiss_index.bin
python main.py --score "I quit my job to backpack across South America with no plan."
python main.py --benchmark   # run the 10-conversation benchmark -> outputs/benchmark_report.json
```

`--audit` and `--embed` only need to be re-run when `data/Facets_Assignment.csv`
changes. `--score` and `--benchmark` reuse the saved index.

## Benchmark methodology: why two numbers, not one

Earlier versions of this project reported a single blended pass rate
("18/30 correct"). That number mixes two very different kinds of failure
-- a facet FAISS never retrieved, and a facet that WAS retrieved but scored
wrong -- into one figure, which makes it impossible to tell whether a low
score means "the LLM is bad at judging facets" or "the retriever didn't
even show the LLM the right facet to judge." Professional ML benchmarks
(RAG evaluation, in particular) don't make this mistake -- they report
retriever recall and generator/answer accuracy as separate numbers, and
`src/benchmark.py` now does the same:

1. **Retrieve normally** for each of the 10 benchmark conversations, same
   as production (`retrieve_relevant_facets`, `top_k=40`).
2. **Force-include** any reference facet that's genuinely supposed to be
   observable (`expected_status` "scored" or "insufficient_evidence") but
   wasn't naturally retrieved this time -- looked up directly from
   `outputs/observable_facets.json`, so a force-added facet is always a
   real, legitimately-observable facet, never something conjured up.
   Whether each reference facet was naturally retrieved or force-added is
   tracked per facet.
3. **Score everything** (naturally retrieved + force-included) the normal
   way, then compute two numbers from the same run:
   - **Retrieval recall** -- % of observable reference facets FAISS found
     on its own, *without* force-include. Purely "did the retriever do its job."
   - **Scoring accuracy** -- % correct *given* the right facet was in
     front of the LLM (force-included if the retriever missed it). Purely
     "given the right input, did the model judge it correctly."

Reference facets with `expected_status: "not_observable"` (the medical/
spiritual trap facets) are **never force-included** -- doing so would
defeat the two-layer safety architecture this project is built around (see
`docs/DECISIONS.md` #3). Those are checked separately as a **safety**
metric instead: did the system ever score something it categorically
shouldn't have. `hallucination_demo/` is the place that deliberately does
force a naive scorer to see those facets, on purpose, to show what
skipping this exclusion would look like.

**Latest real run** (`outputs/benchmark_report.json`):

| Metric | Value | What it means |
|---|---|---|
| Retrieval recall | **26%** (7/27) | FAISS found the intended facet on its own about 1 time in 4. This is the known, diagnosed weak point -- see `docs/DECISIONS.md` #1 (embedding-text quality, not `top_k`, is the root cause). |
| Scoring accuracy | **59%** (16/27) | When the LLM is actually handed the right facet, it judges it correctly a bit more than half the time -- meaningfully better than retrieval recall alone would suggest, but not high enough to call "solved." |
| Safety | **0/3 violations** | No medical/spiritual/malformed facet was ever scored, across every run this project has done. |

Read together: retrieval is the bigger current bottleneck (26% recall vs.
59% scoring accuracy), which tells you where to spend the next round of
engineering effort -- see "What I'd improve with another day" below. The
old blended style would have reported this same run as "19/30 (63%)
correct" -- a single number that looks fine on its own but tells you
nothing about *which* of the two very different problems above is
actually the one worth fixing.

## Run the Streamlit UI

```bash
streamlit run app.py
```

A paste-a-conversation-and-see-results web UI over the same pipeline the
CLI uses -- no code required. Requires `--audit` and `--embed` to have been
run at least once already (the app shows a red status badge and tells you
which command to run if the index is missing). Shows three status badges
before you type anything (facet index, **LLM Backend** -- which backend is
actually active, Ollama or Groq, see "LLM Backend Options" above -- and GPU
VRAM headroom), a summary of retrieved/scored/abstained counts, the full
facet scores table, and an outcome-distribution bar chart. If Ollama isn't
detected, a password-masked `GROQ_API_KEY` field appears so you can paste a
key directly into the browser instead of setting an environment variable
and restarting.

## Architecture

```
data/Facets_Assignment.csv
        |
        v
  src/audit.py            -- clean/normalize facet names, classify into 7
        |                     categories, flag observable vs not, assign
        |                     sensitivity + abstention_reason + scoring anchors
        v
outputs/enriched_facets.csv
        |
        v
  src/embeddings.py       -- embed only conversation-observable facets
        |                     (all-MiniLM-L6-v2) into a FAISS index
        v
outputs/faiss_index.bin + observable_facets.json
        |
        v
  src/pipeline.py --------> retrieve_relevant_facets(conversation, top_k=40)
        |                          |
        |                          v
        |                   src/scorer.py -- split into batches of 10,
        |                     one focused Ollama prompt per batch, robust
        |                     JSON parsing, hard-block on medical facets
        v
outputs/pipeline_output_{timestamp}.json
```

Two safety layers keep the system from hallucinating on facets it has no
business judging from a short chat:

1. **Retrieval-time filter**: `audit.py` marks medical/biological,
   spiritual/esoteric, social/demographic, and malformed facets as
   `conversation_observable = False`. `embeddings.py` only ever indexes the
   `True` rows, so those facets are structurally unreachable by retrieval --
   `FSH level` and `Basophil count` never enter the FAISS index at all, even
   if a conversation is entirely about symptoms and blood tests.
2. **Scoring-time filter**: `scorer.py` additionally hard-blocks scoring any
   facet tagged `medical_biological` even if it somehow reaches that stage,
   forcing `"status": "not_observable"` instead of trusting the LLM. This
   redundancy is deliberate -- see `DECISIONS.md` #3 for why I built two
   independent checks instead of one.

Verified empirically, not just by design: across two full 10-conversation
benchmark runs (including a "medical trap" and "spiritual trap" conversation
specifically written to bait the system), the safety-violation count was 0
both times. Three worked examples of this in action -- a naive scored
output next to what this system actually does -- are in
`docs/HALLUCINATION_EXAMPLES.md`.

## Known limitations

- **Retrieval misses real facets.** The benchmark's biggest failure mode by
  far is `retrieval_miss` -- a facet a human would clearly expect to be
  scored (`Emotionalism` for a conversation about crying in a grocery
  store, `Peacefulness` for a meditation-and-decision-making conversation)
  never gets retrieved into the top-`k` candidates at all, so the LLM never
  sees it. I diagnosed this down to the embedding text quality, not the
  retrieval window size -- widening `top_k` from 25 to 40 only recovered 1
  of 13 misses. Full writeup in `DEBUGGING.md` #1 and `DECISIONS.md` #1.
- **Facet classification in `audit.py` is rule-based** (keyword/regex
  heuristics), not ML -- reproducible and auditable, but it will
  misclassify some edge cases. I found a concrete example of this myself:
  the trailing-colon-means-malformed rule sweeps up genuine-looking traits
  like `Achievement Motivation:` and `HonestyHumility:` into the
  non-scorable bucket alongside actual leftover spreadsheet headers like
  `Computer Skills:`. See `DECISIONS.md` #2 for why I kept the conservative
  rule anyway.
- **llama3.1 running locally is not guaranteed to return clean JSON on
  every call.** I never actually hit a `parse_error` in either full
  benchmark run (70 batched calls combined, 0 failures), but I don't trust
  that as a guarantee going forward -- I stress-tested the fallback parser
  directly against markdown-fenced, prose-wrapped, and object-wrapped
  responses to confirm the defensive path works even though it wasn't
  organically triggered. See `DEBUGGING.md` #2.
- **Scoring is not perfectly deterministic between runs** even at low
  temperature (0.1) -- expect small score drift (usually +/-1) if you
  re-run the same conversation twice.
- **The benchmark's reference labels are my own hand-labels for 10
  conversations and 30 facet judgments** -- useful for catching systematic
  failure modes (which is exactly what it did with the retrieval-miss
  finding, and later with separating retrieval recall from scoring
  accuracy -- see "Benchmark methodology" above), but it's not a large or
  independently-validated ground truth set, so the 26%/59% numbers should
  be read as "diagnostic signal at n=27," not as precise population
  accuracy figures.

## Scaling to 5,000 Facets

The current system runs against 399 raw facets (322 conversation-observable
after filtering). Here's what actually changes, piece by piece, if that
grows to 5,000 -- and what doesn't.

**Indexing (`src/embeddings.py`):** No code changes needed. FAISS's flat
`IndexFlatIP` index is exact brute-force cosine similarity, and at 5,000
384-dimensional vectors that's still a trivially small index by FAISS
standards (people routinely run flat indexes at millions of vectors). Build
time scales roughly linearly with facet count for the embedding step
(`SentenceTransformer.encode()` is the actual bottleneck, not FAISS
`index.add()`): 399 facets embed in about 1-2 seconds on CPU in this
project, so 5,000 facets (~12.5x more) would land around 10-15 seconds --
still a one-time cost you only pay when `data/Facets_Assignment.csv`
changes, not per conversation.

**Retrieval (`retrieve_relevant_facets()`):** `top_k` stays fixed at 40
regardless of how many total facets are in the index -- that's the whole
point of retrieval-based filtering instead of scaling the LLM's input with
the corpus size. A flat FAISS search over 5,000 vectors is still a
single-digit-millisecond operation (roughly proportional to `N * dim`
multiply-adds, so ~12.5x the raw compute of 399 facets, but that's the
difference between ~0.2ms and ~2ms in practice, not something you'd notice
per conversation). The number of facets sent to the LLM per conversation
does not change.

**Batching (`src/scorer.py`):** Still 10 facets per Ollama call. With
`top_k=40` retrieved, that's still exactly 4 batches per conversation --
identical to today, because batching depends on `top_k`, not on total
corpus size. This is the key scaling property of the whole architecture:
**growing the facet library doesn't grow the number of LLM calls per
conversation** as long as `top_k` stays fixed. It only grows the pool that
retrieval selects from.

**Latency estimate:** Today, scoring one conversation is 4 batches x ~11s
per Ollama call on this hardware (RTX 4050) = roughly 45 seconds
end-to-end. At 5,000 facets, retrieval still returns 40 candidates, still
splits into 4 batches, so per-conversation scoring latency stays at roughly
the same ~45 seconds -- the LLM has no idea the underlying facet library
grew. The only latency that increases is the one-time index build:
`python main.py --embed` goes from ~2s to an estimated ~10-15s, which you
only pay once (or whenever the CSV changes), not per conversation scored.

**Where the real bottleneck moves:** At 5,000 facets with `top_k` still 40,
the system as designed actually scales fine -- the bottleneck stays exactly
where it is today: Ollama's single-threaded local inference throughput
during the 4 sequential batch calls per conversation. That bottleneck would
only get *worse* if `top_k` were naively scaled up alongside the facet
count (e.g. someone deciding "5,000 facets means we should retrieve 500,
not 40") -- that's the scenario to actively avoid, since it would turn 4
batches into 50 and make each conversation take 10x longer. Keeping
`top_k` fixed regardless of corpus size is what keeps this cheap.

If conversation *volume* (not facet count) became the actual scaling
problem -- many conversations scored per minute rather than one at a time
-- the fixes would be:
- **Async/concurrent batching:** fire multiple batch requests to Ollama
  concurrently instead of the current sequential loop in
  `score_facets()`, or move to a serving layer built for concurrent
  local inference (e.g. vLLM, or Ollama's own request queuing) instead of
  one conversation at a time.
- **Caching:** add a simple cache keyed on `(facet_normalized,
  hash(conversation_text))` -- a dict for a single-process demo, Redis for
  anything multi-process/persistent -- so re-scoring the exact same
  conversation (e.g. a user re-submitting, or a retry after a parse error)
  doesn't re-spend an LLM call on facets already scored for that exact
  text. This wouldn't help with novel conversations, but it's a real,
  cheap win for repeated/retried requests.

## What I'd improve with another day

1. **Fix the actual retrieval-miss root cause instead of just widening
   `top_k`.** Rewrite `generate_scoring_anchor()` in `audit.py` to produce
   a couple of natural example phrasings per facet (what someone
   demonstrating high/low `Emotionalism` might actually say in
   conversation) instead of the generic "1=Very low X; 5=Very high X"
   rubric sentence, re-embed against that richer text, and re-run the
   benchmark to see if the miss rate actually drops instead of just
   recovering 1 out of 13.
2. **Manually review all 32 `header_or_malformed` entries** against the
   original spreadsheet structure to separate genuine leftover headers from
   traits that just have a stray trailing colon, instead of the blanket
   colon-based rule. This would recover facets like `Achievement
   Motivation` and `Leadership Potential` for scoring without
   reintroducing the risk of scoring an actual section header.
3. **Spot-check the ~293 facets that defaulted to `personality_trait`** for
   misclassifications the keyword lists in `audit.py` might have missed --
   I only found the colon-heuristic issue because I happened to query
   specific facet names; there could be other systematic errors I haven't
   looked for yet.
4. **Add a cross-encoder reranker over the top-100 FAISS candidates**
   (retrieve wider, then rerank down to the batch size) instead of relying
   on raw embedding similarity alone -- widening `top_k` from 25 to 40 only
   recovered 1 of 13 retrieval misses (see `DECISIONS.md` #1), which tells
   me the embedding-similarity ranking itself is the weak link, not the
   cutoff. The hypothesis worth testing: retrieve top-100 candidates cheaply
   via FAISS, then rerank those 100 with a cross-encoder that scores
   facet-conversation pairs jointly (rather than via two separately-embedded
   vectors) before batching the top ones for the LLM. Now that retrieval
   recall is measured on its own instead of blended with scoring accuracy
   (see "Benchmark methodology" above), the real baseline to beat is
   **26% (7/27)**, not the older, looser "18/30 = 60%" blended estimate --
   the honest number is worse than I originally thought, which if anything
   makes this a higher-priority fix, not a lower one. I haven't built or
   measured a reranker yet, so treat any target number here as a hypothesis
   to validate, not a result.
5. **Add confidence calibration**: compare the LLM's stated `confidence`
   field ("high"/"medium"/"low") against actual agreement rate with my
   reference labels. If "high confidence" scores aren't meaningfully more
   accurate than "medium confidence" ones, the confidence field is
   decorative rather than informative, and that's worth knowing before
   anyone downstream treats it as a real signal.
6. **Replace the CPU-only torch build with a CUDA build.** I found during
   Streamlit testing that this project's installed `torch` is `2.6.0+cpu`
   (`torch.version.cuda is None`), so `torch.cuda.is_available()` always
   returns `False` -- even though `nvidia-smi` confirms a real RTX 4050
   with free VRAM that Ollama is actively using. `app.py`'s GPU VRAM status
   badge currently has to fall back to "unknown" instead of showing real
   free-VRAM numbers because of this. Installing the CUDA build (`pip
   install torch --index-url https://download.pytorch.org/whl/cu121` or
   similar for this driver) would let that badge report actual green/
   yellow/red VRAM status instead of an honest shrug.
7. **Add inter-annotator agreement** by having a second reviewer
   independently label a sample of the benchmark's reference facets before
   seeing my labels or the system's output. Right now the "reference"
   scores in `src/benchmark.py` are entirely my own single judgment call --
   I don't actually know how much two reasonable humans would agree with
   each other on, say, whether a given conversation supports `Emotionalism:
   4` vs `5`. Without that baseline, it's hard to say how much of the
   system's disagreement with my labels reflects a real system error versus
   normal human-to-human disagreement on an inherently subjective 1-5 scale.
8. **Grow the benchmark past 10 conversations / 30 reference facets.** The
   current set was enough to surface the retrieval-miss pattern, but a
   larger, more systematically varied reference set would tell me whether
   that failure mode generalizes the way I think it does or whether it's
   specific to how I phrased these particular 10 conversations.
9. **Test batch sizes other than 10** (the brief specifies "max 10," but I
   never actually swept 5/8/10/12 against parse-error rate to confirm 10
   is the right point on the reliability/throughput curve for llama3.1 on
   this specific hardware -- I just never saw a failure at 10, which isn't
   the same as knowing where the failures start.)
