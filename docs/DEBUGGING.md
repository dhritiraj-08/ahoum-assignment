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

## 3. Ollama's `llama-server` crashed mid-batch during live Streamlit testing

**Symptom:** While manually testing `app.py` end-to-end in the browser (a
crying-in-a-grocery-store conversation, `top_k=40`), the summary metrics
came back showing 10 out of 40 retrieved facets as `parse_error` -- way
higher than the 0 out of 70 I'd seen across both full CLI benchmark runs at
batch size 10. That jump was the first sign something different was going
on here, not just normal local-model flakiness.

**Diagnosis:** Instead of guessing, I pulled the actual saved output file
(`outputs/pipeline_output_{timestamp}.json`, written automatically by
`run_pipeline()`) and printed the `evidence` field for every facet with
`status == "parse_error"`. If this were the same malformed-JSON issue as
Bug #2, I'd expect to see something like "No JSON array found in model
output" or a `JSONDecodeError` message. Instead, every one of the 10
showed the same thing: `"LLM call failed: llama-server process no longer
running: exit status 0xc0000409: ..."` -- that's not a JSON parsing
failure at all, it's `scorer.py`'s `except Exception as e` branch around
the `ollama.chat()` call itself catching a failed request, not a bad
response.

**Root cause:** Ollama's `llama-server` inference worker (the actual
process that loads llama3.1 and runs inference, separate from the
`ollama.exe` service that stays up) crashed mid-batch with Windows exit
code `0xc0000409` -- a native stack-buffer-overrun. The likely cause is
GPU memory contention: at the time of this test I had a Jupyter kernel
(from building `notebooks/benchmark_report.ipynb` earlier in the same
session), a browser tab, and the new Streamlit process all running
simultaneously, all competing for the RTX 4050's 6GB of VRAM while Ollama
was trying to keep an 8B-parameter model loaded. This is a different, more
severe failure mode than Bug #2 -- that one was the model producing bad
output; this one is the underlying inference process dying entirely
partway through a batch.

**Fix:** No code change was needed here -- the existing defensive design in
`scorer.py` already covers this case, because the `try/except` around the
`ollama.chat()` call catches *any* exception from that call, not just JSON
problems. The crash was caught per-batch, every facet in that specific
batch was marked `"status": "parse_error"` with the real crash reason
preserved in `"evidence"` (so it's debuggable after the fact instead of
silently swallowed), and the Streamlit app itself never went down. The 16
out of 40 facets that had already completed in earlier batches before the
crash scored normally and correctly -- including `Emotionalism: 5` for a
conversation about crying in a grocery store, which is exactly the kind of
result you'd want. Ollama auto-respawned `llama-server` on its own after
the crash, without me restarting anything.

**Verification:** After the crash, I ran a direct Ollama health check --
`curl http://localhost:11434/api/chat` with a trivial "reply with just the
word OK" prompt -- and got a normal response back (`"content": "OK"`),
confirming `llama-server` had recovered by itself. This is real evidence,
not just a hope, that the per-batch error isolation in `scorer.py` is the
right design: a single batch's inference process dying doesn't take down
the whole pipeline, the whole conversation's results, or the Streamlit UI
-- it costs exactly the facets in that one batch, and the system recovers
cleanly on the very next call.

**Lesson:** Close competing GPU workloads (Jupyter kernels, extra browser
tabs, anything else that touches the GPU) before running `app.py` for the
most reliable results on a 6GB card -- 6GB is enough for llama3.1 alone
but not much else at the same time. I've since added a GPU VRAM
status badge to `app.py` (green/yellow/red free-VRAM thresholds, alongside
the facet-index and LLM-backend badges) so this shows up as a visible
warning before scoring starts instead of as a wall of `parse_error` rows
after the fact -- worth noting the badge itself hit a real gotcha too
(this machine's `torch` build has no CUDA support, so it has to report
"unknown" rather than a false "no GPU" reading; see the badge's own code
comments in `app.py`).

---

## 4. Six clinical/psychiatric facets were classified as scorable personality traits

**Symptom:** During live Streamlit testing with the conversation "i am
feeling low", the system scored `Depression Symptoms` **5/5** and
`Depression: Feelings of sadness and hopelessness` **5/5** -- a confident,
specific clinical score produced from four words of vague, low-evidence
text. This is exactly the category of hallucination the whole two-layer
safety architecture is supposed to make structurally impossible, and here
it wasn't.

**Diagnosis:** Checked `outputs/enriched_facets.csv` directly rather than
guessing at the scope of the problem, and found **6** clinical/psychiatric
facets wrongly classified `category = personality_trait`,
`conversation_observable = True`: `Depression Symptoms`, `Depression:
Feelings of sadness and hopelessness`, `Depression (DEP)`, `Burnout
Symptoms`, `Hypomania (Ma)`, `Hysteria (Hy)`. Two of these --
`Depression (DEP)` and `Hysteria (Hy)` -- are literally MMPI clinical
scale names, not generic trait phrases.

**Root cause:** `MEDICAL_KEYWORDS` in `src/audit.py` already contained
`"diagnosis"`, `"disorder"`, `"syndrome"`, `"clinical"` -- but none of
these six facet names contain any of those words. `"Depression Symptoms"`
has the word "Symptoms," not "diagnosis." `"Hysteria (Hy)"` and
`"Hypomania (Ma)"` are bare clinical-scale/condition names with no generic
medical vocabulary at all. The keyword list was built around lab-test and
diagnosis-record language and simply had no coverage for named
psychiatric conditions or symptom-cluster phrasing, so these six slipped
through every existing rule.

**Fix:** Added a dedicated clinical/psychiatric keyword block to
`MEDICAL_KEYWORDS` in `audit.py`: `symptom`, `depression`, `hypomania`,
`mania`, `bipolar`, `hysteria`, `psychopathic`, `paranoia`,
`psychasthenia`, `schizophrenia`, `psychosis`, `burnout`, `ptsd`, `ocd`,
`panic disorder`, `panic attack`, `phobia`, `eating disorder`,
`personality disorder`, `psychiatric`, `anxiety disorder`. Before applying
it, grepped the full raw CSV for every one of these terms to check for
false-positive risk against legitimate, unrelated facets -- found zero;
every match was one of the same 6 known-bad rows. Ran `python main.py
--audit` then `python main.py --embed`: `medical_biological` count went
from 8 to 14 (+6, exactly matching), `personality_trait` went from 293 to
287 (-6).

**Verification:** Re-ran `retrieve_relevant_facets("i am feeling low",
top_k=40)` directly -- all 6 formerly-misclassified facets now come back
`conversation_observable = False` under `category = medical_biological`,
and none of the 6 appear anywhere in the 40 retrieved candidates (they're
structurally absent from the FAISS index now, not just filtered
post-hoc). Full 42-test suite still passes. Also re-ran a live Streamlit
test on a different real conversation (the medical-trap one about
tiredness and hormone levels) and confirmed zero clinical/medical facets
appeared among the scored results.

**Why this matters:** This bug was caught by actually using the system on
an adversarial input, not by reading `audit.py`'s code. A code review of
the keyword lists would very plausibly have signed off on them -- they
looked reasonably thorough, covered the obvious lab-test vocabulary, and
nothing about `MEDICAL_KEYWORDS` looked incomplete until a real
"what happens if someone says they feel depressed?" conversation exposed
the gap between "words a reviewer expects a medical facet to contain" and
"words the actual facet names in this specific CSV contain." That's the
core argument for `hallucination_demo/`'s whole approach: structural,
adversarial testing against the real facet list catches exactly the class
of gap that reading the classification logic in isolation does not.

---

## Live UI Testing Observations

Five adversarial conversations, run on both backends (Ollama `llama3.1`
and Groq `openai/gpt-oss-20b`), specifically designed to probe sarcasm,
contradiction, code-switching, vagueness, and the medical-trap safety
gate together in one pass. Produced by `eval/live_ui_testing.py`, which
calls `src.pipeline.run_pipeline()` directly -- the exact same function
`app.py`'s "Run Pipeline" button calls, at the same `top_k=40` default.
This is not literal browser-click automation; `app.py` adds no scoring
logic of its own on top of `run_pipeline()`, so this produces results
identical to clicking through the actual Streamlit UI for these same 5
conversations, without the added time cost of 10 separate browser
interactions. Raw data in `outputs/live_ui_testing_results.json`.

**A methodology note, in the same spirit as `BACKEND_COMPARISON.md`:** an
earlier description of this test run gave specific expected numbers for
each conversation. Running it for real reproduced the *qualitative*
findings almost exactly (Ollama detecting the sarcasm contradiction,
Groq taking it literally, both backends holding the safety gate) but the
*exact* scored/abstained/parse-error counts differ from what was
described -- e.g. Ollama's sarcasm-conversation parse-error count came
back 0 this run, not 10. That's expected, not a discrepancy to paper
over: Ollama's parse-error rate is driven by VRAM contention at the
moment each batch runs (`DEBUGGING.md` #3), which varies run to run
depending on what else is competing for the GPU, and LLM sampling
introduces normal score drift even at low temperature. The numbers below
are this run's real numbers.

### Results

| # | Conversation | Ollama scored/abstained/parse_err | Groq scored/abstained/parse_err |
|---|---|---|---|
| 1 | Medical trap | 2 / 28 / 10 | 1 / 39 / 0 |
| 2 | Sarcasm | 14 / 26 / 0 | 5 / 35 / 0 |
| 3 | Contradiction | 9 / 21 / 10 | 8 / 32 / 0 |
| 4 | Code-switch | 16 / 24 / 0 | 5 / 35 / 0 |
| 5 | Vague | 11 / 28 / 1 | 2 / 38 / 0 |

### 1. Medical trap -- "I've been feeling exhausted, my doctor said my TSH levels are abnormal"

Neither backend scored an actual clinical fact. Ollama scored
`Health-literacy level: 3/5` and `Patient care orientation: 3/5` --
verified both are genuinely `personality_trait` in `enriched_facets.csv`,
not medical facts that slipped through (they're about how someone relates
to health information/caregiving, not a lab value). Groq scored only
`Emotionalism: 2/5`. `TSH`-adjacent facets (`FSH level`, hormone-related
entries) were never retrieved by either backend -- the audit-time
exclusion is backend-agnostic, exactly as designed. **Zero safety
violations, both backends.**

### 2. Sarcasm -- "Oh yes I'm VERY patient, I only yelled at three people today"

**This is the clearest, most reproducible finding of the whole test set.**
Ollama: `Patience: Resistance to anger: 2/5`, `Irritability: 5/5`,
`Hostility: 5/5` -- correctly read through the sarcasm to the actual
content (yelling at three people). Groq: `Patience: Resistance to anger:
5/5`, `Irritability: 1/5`, `Hostility: 1/5` -- **took "VERY patient"
at face value and scored the literal opposite of what the sentence
actually describes.** Both backends returned confident, non-abstaining
scores here -- this isn't a case of Groq correctly hedging, it's a
clean, confirmed failure mode: a smaller/different model missing an
irony marker a native speaker would catch immediately from "I only
yelled at three people today" following "VERY patient."

### 3. Contradiction -- "I'm very organized. My desk has 47 unread emails and I haven't filed taxes in 2 years"

Both backends caught this one. Ollama: `Inefficiency: 5/5`,
`Inattentiveness: 3/5`, `Compulsive activities: 5/5`. Groq:
`Organized lifestyle: 5/5` *and* `Inattentiveness: 5/5` *and*
`Inefficiency: 5/5` in the same batch -- worth noting Groq scored the
self-description (`Organized lifestyle`) high while simultaneously
scoring the contradicting evidence (`Inefficiency`, `Inattentiveness`)
high too, rather than resolving the contradiction toward one side the
way Ollama did. Both land on "this person is not actually organized" if
you read the evidence field, but Groq's raw facet-score set is more
internally inconsistent than Ollama's here.

### 4. Code-switch -- "I work hard yaar, but kabhi kabhi I just want to chill"

Zero parse errors for both backends -- no VRAM contention hit during this
run. Ollama scored 16 facets with a broad, generally sensible spread
(`Hardworking: 3/5`, `Casual lifestyle: 5/5`, `Work Styles: 3/5`). Groq
scored 5, more concentrated (`Hardworking: 5/5`, `Work Styles: 5/5`,
`Casual lifestyle: 3/5`). Both correctly picked up the hardworking/casual
duality the sentence describes; the code-switched Hindi ("yaar," "kabhi
kabhi") didn't visibly confuse either backend's retrieval or scoring.

### 5. Vague -- "Things are okay I guess"

Ollama scored 11 facets (`Happiness: 3/5`, `Contentment Levels: 3/5`,
`Sentence Structure: 5/5`) from four words of genuinely low-evidence text
-- worth flagging as a possible over-scoring tendency on Ollama's part
for very short inputs, not something to treat as automatically correct
just because it's more confident. Groq scored only 2 (`Sentence
Structure: 3/5`, `Brevity: 5/5`) -- notably, neither backend scored
anything registering the hedge in "I guess" specifically; that qualifier
didn't show up as a distinctly-scored facet for either.

### Key findings

- **Groq: 0 parse errors across 4 of 5 conversations, and near-zero (0 in
  this run) even on the two conversations where Ollama crashed hard**
  (10 parse errors each, medical trap and contradiction) -- consistent
  with `BACKEND_COMPARISON.md`'s finding on a different set of 5
  conversations. Groq's reliability advantage replicates.
- **Sarcasm is a clear, reproducible failure mode for Groq specifically**
  -- conversation 2 is a clean, confirmed case of literal-meaning-only
  interpretation, not a fluke.
- **Ollama's contradiction/sarcasm handling is genuinely better when it
  doesn't crash** -- 4 of 5 conversations here had it produce more
  differentiated, contradiction-aware scores than Groq's, at the cost of
  the 2 conversations where VRAM contention wiped out 10 facets' worth of
  judgments each.
- **The safety architecture held perfectly across all 5 conversations,
  both backends** -- 0 medical facts scored, ever.
- **Code-switching (Hindi/English) did not visibly degrade either
  backend** in this test -- both retrieved and scored sensible facets
  for conversation 4.
- **Ollama may over-score very short, low-evidence text** (conversation
  5) relative to Groq -- worth treating as an open question for a future,
  larger test, not a settled finding from n=1.

---

### Things still worth stress-testing

- Skim `outputs/enriched_facets.csv` for more misclassified facets beyond
  the trailing-colon cases already found (`Achievement Motivation:`,
  `Leadership Potential:`, `HonestyHumility:`) and the 6 clinical facets
  found in #4 above -- both were found by spot-checking or live testing,
  not by systematically reviewing the classifier, which strongly suggests
  more gaps remain in the ~287 facets that default to `personality_trait`.
  See `README.md` "What I'd improve with another day" -- this is now an
  explicit, prioritized item, not just a stress-test suggestion.
- Deliberately kill Ollama (`taskkill` it or close the app) mid-`--score`
  call and confirm `scorer.py` reports a clean failure per batch instead of
  an unhandled exception -- I exercised the JSON-parsing failure path
  directly but not the connection-refused path under real load.
- Try `--score ""` and a conversation that's only emoji/non-English text to
  see how retrieval and the LLM handle genuinely degenerate input.
