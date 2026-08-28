# Backend Comparison: Ollama vs. Groq

Real, live comparison of the two LLM backends `src/scorer.py` supports,
run on the same 5 benchmark conversations, same `top_k=25`, same prompts,
same retrieval -- the only variable is which backend actually scored the
facets. Produced by `eval/backend_comparison.py`; raw data in
`outputs/backend_comparison_results.json`.

**A note on how this doc came to exist:** an earlier draft of this
request described specific expected findings (a 2/5 vs. 5/5 split on the
`Patience` facet, Groq scoring 0 facets vs. Ollama 2 on the medical trap).
This document does **not** contain those numbers, because they don't match
what actually happened when the comparison was run for real. The numbers
below are what the two backends actually did.

## Setup

- 5 of the 10 `src/benchmark.py` conversations: `clear_direct`,
  `contradictory`, `sarcastic`, `medical_trap`, `high_emotional`.
- `top_k=25` for both backends (same candidate pool size -- retrieval
  quality is not the variable being compared here).
- Ollama: local `llama3.1`, run first for each conversation.
- Groq: `openai/gpt-oss-20b`, run second, with Ollama detection forced off
  (same mechanism as `main.py --test-groq`) so it's genuinely exercising
  Groq regardless of Ollama's state.

## Results

| Conversation | Ollama scored | Ollama abstained | Ollama parse_err | Groq scored | Groq abstained | Groq parse_err |
|---|---|---|---|---|---|---|
| 1. clear_direct | 5 | 10 | **10** | 8 | 17 | 0 |
| 3. contradictory | 16 | 9 | 0 | 20 | 5 | 0 |
| 4. sarcastic | 11 | 13 | 1 | 7 | 18 | 0 |
| 7. medical_trap | 6 | 9 | **10** | 2 | 23 | 0 |
| 9. high_emotional | 16 | 9 | 0 | 8 | 17 | 0 |
| **Total (125 facets each)** | **54** | **50** | **21** | **45** | **80** | **0** |

## Key findings

**Groq: 0 parse errors across all 5 conversations, confirmed.** Every
single one of the 125 facets Groq was asked to judge came back as valid,
parseable JSON. Zero exceptions, zero malformed output.

**Ollama: 21 parse errors, concentrated in 2 of 5 conversations (10 each),
not spread evenly.** Checked the actual evidence field on these -- they're
the same `llama-server process has terminated: exit status 0xc0000409`
crash documented in `DEBUGGING.md` #3 (GPU/VRAM contention), not JSON
malformation. This is an infrastructure reliability problem specific to
running a local model on shared, limited VRAM, not evidence the model
itself is worse at the task -- the 3 conversations where Ollama didn't
crash (`contradictory`, `sarcastic`, `high_emotional`) show it scoring
normally.

**Groq is more conservative in aggregate**, but modestly, not dramatically:
scored 45/125 (36%) vs. Ollama's 54/125 (43%). Excluding Ollama's 21
crashed slots (which have no real status, not a genuine abstention) and
comparing only the 104 facets Ollama actually completed, Ollama scored
54/104 (52%) -- so the gap is larger than the raw totals suggest once
crash noise is removed: Groq abstained noticeably more often when it did
get to render a judgment.

**Contradiction handling (`Patience: Resistance to anger`, expected score
1/5): both backends scored it identically -- 1/5 each.** Both correctly
picked up on the barista-screaming contradiction despite the person
claiming to be patient. No meaningful difference observed between backends
on this specific test, contrary to what was initially expected going into
this comparison.

**Medical trap safety: identical and complete for both backends.** Neither
`FSH level` nor `Basophil count` was ever retrieved by either backend --
expected, since retrieval-time exclusion (`src/audit.py` /
`src/embeddings.py`) is backend-agnostic; the LLM backend only ever sees
what's already been retrieved. `Emotionalism` (the one genuinely-observable
reference facet in that conversation) was a **retrieval miss for both
backends** too -- consistent with the known retrieval-recall problem
documented in `DECISIONS.md` #1, and itself evidence that the retrieval
step, not the LLM backend, is what determines whether this facet gets
judged at all. Zero safety violations for either backend across all 5
conversations.

**Speed: not measured in this comparison.** Wall-clock timing per backend
wasn't captured by `eval/backend_comparison.py`. Local inference avoiding
a network round-trip is a reasonable expectation for why Ollama would be
faster when it isn't crashing, but that's an expectation, not a number
this document can back up -- a real gap worth closing in a follow-up run
if backend latency ever matters for a production decision.

## Recommendation

**Use Groq when reliability matters more than data locality.** Zero parse
errors across 125 real facet judgments is the standout finding here --
whatever GPU contention exists on the machine, Groq's hosted inference
doesn't experience it.

**Use Ollama when privacy matters more than that reliability gap, or when
GPU contention is under control.** The conversation never leaves the
machine, and 3 of the 5 test conversations show it performing normally with
zero issues -- the failures are tied to a specific, already-diagnosed
resource contention problem (`DEBUGGING.md` #3), not a fundamental flaw
in running the model locally. Closing Jupyter/browser/other GPU workloads
before scoring (the same mitigation already documented for Bug #3) should
substantially reduce Ollama's parse-error rate.

Both backends held the safety architecture perfectly -- that part of the
recommendation doesn't change based on which one you pick.
