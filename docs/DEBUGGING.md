# Debugging Log

Real issues found while building and testing this system, and how I tracked
each one down. One entry per issue: symptom, how I diagnosed it, what the
actual root cause turned out to be, what I changed, and how I confirmed the
fix actually did something (not just "ran it again and it looked fine").

---

## 1. 13 out of 30 benchmark reference facets never showed up in the results at all

**Symptom:** Ran `python main.py --benchmark`. The printed report showed
17/30 reference facets handled correctly, with `retrieval_miss: 13` as the
single largest bucket by far -- bigger than every other outcome combined.
Facets like `Emotionalism`, `Self-improvement`, `Peacefulness`, `Decency`,
`Doggedness`, and `Common-sense` weren't scored *wrong* -- they weren't in
the pipeline's `results` list at all for the conversations where I expected
them. That means retrieval (FAISS, top-25) never surfaced them; the LLM
never got a chance to score or abstain on them.

**Diagnosis:** I opened `outputs/benchmark_report.json` directly and wrote a
short inline Python script to pull out every entry where
`evaluations[i]['outcome'] == 'retrieval_miss'`, printed per case, instead
of just reading the summary counts in the terminal table. That gave me the
exact list of which facet was missing in which conversation type (see the
list above -- it wasn't random, it clustered on emotional/interpersonal
trait words). My first hypothesis was that `top_k=25` was just too narrow
against 322 total observable facets, so I tested that hypothesis directly:
I changed `TOP_K_DEFAULT` from 25 to 40 in `src/pipeline.py` and re-ran the
exact same benchmark. If the hypothesis were right, a lot of those 13 misses
should have recovered with 60% more candidates per conversation.

**Root cause:** The `top_k=40` re-run only recovered 1 of the 13 misses
(`Compassion`, in the code-switched conversation) -- overall went from
17/30 to 18/30. Twelve misses were completely unaffected by adding 15 more
candidate slots. That result rules out "window too narrow" as the primary
cause -- if it were purely about window size, a 60% wider window should have
moved the needle more than 1/13. The actual cause is upstream, in what gets
embedded per facet in `embeddings.py`/`audit.py`: each facet is embedded as
`"{facet name}: {generic templated scoring anchor}"`, e.g. `"Emotionalism:
1=Very low Emotionalism...; 3=Moderate Emotionalism...; 5=Very high
Emotionalism clearly expressed..."`. That template text is boilerplate, not
natural conversational language, so a conversation that clearly *reads* as
emotional (crying in a grocery store, shaking, calling a friend) doesn't
necessarily land close to that generic anchor sentence in embedding space,
because the conversation never uses words like "Emotionalism" or matches
the rubric phrasing. Widening `top_k` can't fix a ranking problem where the
correct facet isn't just outside the cutoff -- for most of these 12, it's
genuinely far down the similarity ranking, not just-barely-missed.

**Fix:** I kept `top_k=40` (real, if small, improvement, and free to leave
in). I did not attempt the actual fix, which would be rewriting
`generate_scoring_anchor()` in `audit.py` to produce a few natural example
phrasings per facet instead of a generic 1-5 rubric sentence, and
re-embedding against that richer text. That's a bigger change that touches
the audit/embedding pipeline and I didn't want to risk breaking a working
system with only a few hours left. I documented this as a known, diagnosed
limitation instead (see `DECISIONS.md` #1 and `README.md`).

**Verification:** Compared the two full benchmark runs directly:
`retrieval_miss` went from 13 (at top_k=25) to 12 (at top_k=40), overall
correct went from 17/30 to 18/30, and `Compassion` specifically flipped from
`retrieval_miss` to `exact_agreement` in the code-switched case -- confirmed
by re-reading `outputs/benchmark_report.json` after the second run and
diffing the per-case `evaluations` lists. Also confirmed the *other* safety
property held across both runs: 0 `SAFETY_VIOLATION_*` in both, so widening
`top_k` didn't accidentally let a non-observable facet through.

---

## 2. Local LLM JSON reliability is a real risk that batching alone doesn't remove

**Symptom:** This isn't a crash I hit during the recorded benchmark runs --
across both full runs (30 batched Ollama calls at `top_k=25`, 40 at
`top_k=40`), I checked and zero batches came back as `parse_error`. But
that's not the same as the risk not existing: `llama3.1` is a small local
model, and the instruction "respond with JSON only, no prose, no markdown
fences" is exactly the kind of instruction smaller models are known to
drift on, especially when a batch includes a facet the model is uncertain
about and seems to "want" to explain itself around the array. I didn't want
to ship code that assumed clean JSON every time just because it happened to
work on these 10 specific conversations.

**Diagnosis:** Rather than wait for a real crash to happen organically (and
possibly never see one, which is exactly what happened), I read through
`_extract_json_array()` in `src/scorer.py` and then directly stress-tested
it against the specific malformed shapes local models are known to produce:
a response wrapped in a markdown code fence, a response with prose before
and after the array ("Sure, here you go: [...] Hope this helps!"), a
response where the model wraps the array in an object (`{"results":
[...]}`) instead of returning a bare array, and a response that isn't valid
JSON at all.

**Root cause:** This was really a preventable-risk-not-yet-triggered
situation rather than a bug I caught in the act -- but it's a real gap that
would have caused a crash if I'd relied on a bare `json.loads()` call on the
raw Ollama response, which is what a naive first implementation would do.

**Fix:** `_extract_json_array()` in `scorer.py` handles all of the above
before giving up: it strips markdown fences with a regex first, tries a
direct `json.loads()`, falls back to regex-extracting the first `[...]`
block in the text if that fails, and unwraps common object-wrapper shapes
(`results`/`facets`/`data`/`scores` keys) if the top-level parse is a dict
instead of a list. If none of that produces valid JSON,
`score_facet_batch()` catches the resulting `ValueError`/`JSONDecodeError`
and marks every facet in that specific batch as `"status": "parse_error"`
with the failure reason in `"evidence"`, instead of raising and taking down
the whole conversation's scoring run.

**Verification:** Ran the four test shapes above directly against
`_extract_json_array()`. The markdown-fence, prose-wrapped, and
object-wrapped cases all parsed correctly to the expected facet list; the
genuinely-broken case raised a clean `ValueError` ("No JSON array found in
model output") instead of an unhandled exception, which is exactly what
`score_facet_batch()`'s `except` block is built to catch. Separately
confirmed across both real benchmark runs that this path never needed to
fire in practice (0 `parse_error` in 70 total batched calls combined) --
which tells me `llama3.1` at batch size 10 is more reliable on this
hardware than I expected going in, but I'm still keeping the defensive
parsing in place since 70 clean calls isn't a guarantee it stays clean on
different conversations or a different Ollama/model version.

---

### Things still worth stress-testing

- Skim `outputs/enriched_facets.csv` for more misclassified facets beyond
  the trailing-colon cases already found (`Achievement Motivation:`,
  `Leadership Potential:`, `HonestyHumility:`) -- the classifier in
  `src/audit.py` is keyword-based and will get some wrong on 399 diverse
  rows; log any real ones you find here.
- Deliberately kill Ollama (`taskkill` it or close the app) mid-`--score`
  call and confirm `scorer.py` reports a clean failure per batch instead of
  an unhandled exception -- I exercised the JSON-parsing failure path
  directly but not the connection-refused path under real load.
- Try `--score ""` and a conversation that's only emoji/non-English text to
  see how retrieval and the LLM handle genuinely degenerate input.
