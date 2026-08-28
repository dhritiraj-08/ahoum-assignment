# Debugging Log

Real bugs found while running the system, and how they were fixed. Keep one
entry per bug -- future-you (and whoever reviews this for placement) will
want to see that you actually ran this thing and hit real problems, not just
that it worked on the first try.

Template for new entries:

```
## [N]. <short title>

**Symptom:** what you observed (error message, wrong output, etc.)
**Where:** file + function
**Root cause:** what was actually wrong
**Fix:** what you changed
**How you found it:** how you diagnosed it (logs, print debugging, etc.)
```

---

## 1. Retrieval miss -- reference facets never showing up in results at all

**Symptom:** Ran `python main.py --benchmark` and the report came back with
17/30 reference facets handled correctly, with `retrieval_miss: 13` as by
far the biggest bucket. That means for 13 of my hand-labeled reference
facets (things like `Emotionalism`, `Self-improvement`, `Peacefulness`,
`Decency`, `Doggedness`), the facet wasn't in the `results` list of the
pipeline output *at all* -- it never got a chance to be scored or to
abstain, it just wasn't retrieved by FAISS in the first place.

**Where:** `src/embeddings.py`, `retrieve_relevant_facets()` (called from
`src/pipeline.py`, `run_pipeline()`, step 1).

**Root cause:** I read `outputs/benchmark_report.json` directly (each case
has an `evaluations` list with an `outcome` field per reference facet) and
pulled out every `"outcome": "retrieval_miss"` entry to see exactly which
facets were missing and in which conversations. My first guess was that
`top_k=25` was just too narrow out of 322 observable facets, so I bumped it
to 40 in `pipeline.py` and re-ran the benchmark. That only fixed 1 of the 13
misses (18/30 instead of 17/30) -- everything else was *still* missing even
with 15 more slots to work with. Since widening the window barely moved the
number, the real cause has to be that the embedding itself doesn't rank
these facets highly enough for their conversations, not that the cutoff is
in the wrong place. The embedded text per facet is
`"{facet name}: {generic 1-5 anchor template}"`, and that templated anchor
sentence doesn't read anything like natural conversation, which is probably
why common trait words don't match well against conversational phrasing
that expresses them indirectly.

**Fix:** Kept `top_k=40` since it's a strict (if small) improvement and
basically free. I did *not* attempt the real fix (richer, more
example-phrasing-based embedding text per facet, see `DECISIONS.md`
Decision 2) -- with 24 hours total, I decided documenting a diagnosed,
understood limitation was a better use of remaining time than rebuilding
the anchor-generation logic and risking a regression right before handing
this in.

**How you found it:** Ran `python main.py --benchmark` twice (before/after
the `top_k` change) and diffed the `retrieval_miss` lists from
`outputs/benchmark_report.json` using a small inline Python snippet to print
just the missed facets per case, rather than eyeballing the whole JSON file.

---

## 2. Occasional malformed JSON from llama3.1 on ambiguous facets

**Symptom:** Not every batch of 10 facets came back as clean, parseable
JSON from Ollama -- on some runs the model would wrap the array in prose
("Here is the JSON: [...]"), use a trailing comma, or occasionally return an
object like `{"results": [...]}` instead of a bare array. If I'd just
called `json.loads()` on the raw response, this would have thrown and
killed the whole batch (and depending on where in the loop, possibly the
whole conversation's scoring run).

**Where:** `src/scorer.py`, `_extract_json_array()` and `score_facet_batch()`.

**Root cause:** llama3.1 is a much smaller, less consistent model than a
frontier hosted model at strictly following a "return JSON only" instruction,
especially once the batch includes facets that are genuinely ambiguous for
the given conversation (the model seems to "want" to explain its reasoning
in prose around the array on those). This is basically the trade-off I
already expected from Decision 1 (local model vs. hosted API) showing up in
practice.

**Fix:** `_extract_json_array()` doesn't just call `json.loads()` blind --
it first strips markdown code fences if present, then falls back to
regex-extracting the first `[...]` block in the text if a direct parse
fails, and also unwraps common object-wrapper shapes (`{"results": [...]}`,
`{"facets": [...]}`, etc). If none of that produces valid JSON, the
`except (json.JSONDecodeError, ValueError)` in `score_facet_batch()` catches
it and marks every facet in that specific sub-batch as `"status":
"parse_error"` with the parse failure message in `"evidence"`, instead of
raising and crashing the pipeline. One bad batch only costs that batch's 10
facets, not the whole conversation or the whole benchmark run.

**How you found it:** Noticed it during manual `--score` testing before I
even got to the benchmark -- printed the raw `response["message"]["content"]`
from Ollama a few times while debugging batch output and saw the
non-JSON-wrapped responses directly. Confirmed the try/except path works by
checking that `outputs/benchmark_report.json` never shows a crash mid-run
across all 10 conversations, and that both benchmark runs (`top_k=25` and
`top_k=40`) exited cleanly with `[exited with code 0]`.

---

### Things worth stress-testing while filling this in

- Run `python main.py --audit` and actually skim `outputs/enriched_facets.csv`
  for misclassified facets (the classifier in `src/audit.py` is keyword-based
  and WILL get some wrong on 399 diverse rows) -- log any you fix here.
- Run `--benchmark` more than once and check whether `SAFETY_VIOLATION_*`
  ever appears -- if it does, that's a real bug worth a full entry.
- Try `--score ""` (empty string) and a conversation with only emoji/non-English
  text to see how retrieval + the LLM handle degenerate input.
- Kill Ollama (`taskkill` it or close the app) mid-run and confirm
  `scorer.py` reports `parse_error`/a clean failure instead of an
  unhandled exception.
