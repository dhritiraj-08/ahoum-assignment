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

## 1. [FILL IN -- placeholder]

**Symptom:** [FILL IN -- e.g. "python main.py --score ... hung for 2 minutes
then returned all insufficient_evidence"]

**Where:** [FILL IN -- e.g. src/scorer.py, score_facet_batch]

**Root cause:** [FILL IN -- e.g. Ollama was still loading llama3.1 into VRAM
on first call, request eventually succeeded but with a cold-start delay
that looked like a hang]

**Fix:** [FILL IN]

**How you found it:** [FILL IN -- e.g. checked `ollama ps` while the call
was running, checked Ollama server logs]

---

## 2. [FILL IN -- placeholder]

**Symptom:** [FILL IN -- e.g. "benchmark report showed a SAFETY_VIOLATION
for a medical facet" or "JSON parse_error on every batch"]

**Where:** [FILL IN]

**Root cause:** [FILL IN]

**Fix:** [FILL IN]

**How you found it:** [FILL IN]

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
