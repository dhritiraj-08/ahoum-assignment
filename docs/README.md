# Conversation -> Personality Facet Scorer

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

## Running the pipeline (in order)

```bash
python main.py --audit       # clean/classify the 399 raw facets -> outputs/enriched_facets.csv
python main.py --embed       # build the FAISS retrieval index -> outputs/faiss_index.bin
python main.py --score "I quit my job to backpack across South America with no plan."
python main.py --benchmark   # run the 10-conversation benchmark -> outputs/benchmark_report.json
```

`--audit` and `--embed` only need to be re-run when `data/Facets_Assignment.csv`
changes. `--score` and `--benchmark` reuse the saved index.

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
both times.

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
  finding), but it's not a large or independently-validated ground truth
  set, so the 18/30 number should be read as "diagnostic signal," not as a
  precise accuracy percentage.

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
4. **Add a small cross-encoder reranker** on top of the FAISS top-`k`
   candidates to re-score them more precisely against the actual
   conversation before batching for the LLM -- this targets precision
   rather than the recall problem `top_k` was aimed at, and might help with
   cases where a *wrong* facet gets retrieved ahead of the right one.
5. **Grow the benchmark past 10 conversations / 30 reference facets.** The
   current set was enough to surface the retrieval-miss pattern, but a
   larger, more systematically varied reference set would tell me whether
   that failure mode generalizes the way I think it does or whether it's
   specific to how I phrased these particular 10 conversations.
6. **Test batch sizes other than 10** (the brief specifies "max 10," but I
   never actually swept 5/8/10/12 against parse-error rate to confirm 10
   is the right point on the reliability/throughput curve for llama3.1 on
   this specific hardware -- I just never saw a failure at 10, which isn't
   the same as knowing where the failures start.)
