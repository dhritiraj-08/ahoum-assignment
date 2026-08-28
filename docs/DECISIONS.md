# Design Decisions

These are the decisions where the assignment brief genuinely left me room to
choose -- not things like "use Ollama" or "batch size 10" or "use FAISS +
sentence-transformers," which the brief already specified outright. Each
entry below is a real problem I hit while building this where the brief
didn't tell me what to do, so I had to actually pick.

---

## 1. What to do about the retrieval-miss failure mode the benchmark exposed

**The ambiguity/problem:** The brief says to retrieve `top_k` facets and
gives `top_k=25` as an example default in the function signature -- it
doesn't say what to do if that number turns out to be wrong for the actual
data. After building everything and running `python main.py --benchmark`
for the first time, the report came back 17/30 correct with
`retrieval_miss: 13` as by far the biggest bucket. Facets like
`Emotionalism`, `Self-improvement`, `Peacefulness`, `Decency`, `Doggedness`
were never even retrieved by FAISS for conversations that clearly should
have surfaced them -- they weren't scored wrong, they were never sent to the
LLM at all. I had to decide how to respond to that.

**Options considered:**
1. Leave `top_k=25` alone and just document the miss rate as-is.
2. Widen `top_k` (more candidates per conversation, more LLM calls) and see
   if the misses go away.
3. Keep `top_k` where it is and instead rework what gets embedded per facet
   (richer example phrasings instead of the generic anchor template), since
   my hypothesis was that the embedding text itself was the actual problem.
4. Add a reranker (e.g. a cross-encoder) on top of FAISS to re-score the
   top candidates against the conversation more precisely.

**Choice made:** I did option 2 first, as a quick test -- bumped `top_k`
from 25 to 40 and re-ran the full benchmark. It only recovered 1 of the 13
misses (18/30 instead of 17/30; `Compassion` was the one that came back).
Everything else that missed at 25 was *still* missing at 40, out of 322
total observable facets. That result told me option 2 alone wasn't the real
fix -- if it were purely a window-size problem, going from 25 to 40 (60%
more candidates) should have recovered more than 1 out of 13. I kept
`top_k=40` since it's a strict, free improvement, but I did not chase option
3 or 4 further given the time left -- I documented the diagnosis (see
`DEBUGGING.md` #1) instead of rebuilding the embedding pipeline under
pressure.

**Trade-off:** I'm shipping a system that I know still misses roughly 40%
of the facets a human would expect it to catch on this benchmark, and I'm
explicitly not pretending otherwise. The alternative -- silently leaving
`top_k=25` and not investigating -- would have looked cleaner in a demo but
would have hidden a real, now-understood limitation instead of surfacing
it. I'd rather hand in 18/30 with a correct diagnosis than 25/30 achieved by
tuning parameters until the specific 10 benchmark conversations happened to
pass.

---

## 2. How aggressively to treat trailing colons as "malformed/header" during audit

**The ambiguity/problem:** The brief says to detect "entries that are
category headers not actual traits" but doesn't say how to tell the
difference between a genuine category header row (like a leftover
spreadsheet section title) and a genuine trait whose name just happens to
end in a colon. When I ran `--audit`, 32 facets got flagged
`header_or_malformed` this way, including things like `Achievement
Motivation:`, `Leadership Potential:`, and `HonestyHumility:` -- and looking
at those names, they read exactly like plausible personality traits, not
obviously like section headers. I couldn't be sure without opening the raw
CSV in a spreadsheet tool and checking row-by-row whether these were
originally meant as scorable facets or as headers that leaked into the data
rows.

**Options considered:**
1. Treat every trailing-colon entry as malformed, no exceptions (simple,
   deterministic, but will exclude some genuine traits).
2. Try to distinguish "looks like a broad category" from "looks like a
   specific trait" using some secondary heuristic (e.g. word count, whether
   it matches a known category-word list) before deciding.
3. Manually review all 32 flagged entries by hand and hand-correct the ones
   that look like real traits.

**Choice made:** Option 1. I kept the rule simple: if the raw string's last
character is `:`, it's `header_or_malformed`, full stop. I considered option
2 but couldn't come up with a reliable secondary signal in the time I had --
"Achievement Motivation" and "Democratic Leadership" are both two-word
title-case phrases, so word count or capitalization alone can't tell them
apart, and I didn't want to hand-write a list of "known category words" that
I'd just be guessing at. Option 3 (manual review) is the honest fix but I
didn't have time to review all 32 against the source spreadsheet's original
formatting to know for sure which were really headers.

**Trade-off:** I'm accepting some false positives -- real traits like
`Achievement Motivation` get excluded from scoring entirely, even though
they're probably legitimate facets that just have a stray colon from how
the CSV was authored. But the alternative failure mode is worse for this
use case: if I got the heuristic wrong in the other direction and a genuine
section header like `Computer Skills:` slipped through as a "real" facet,
the system would confidently try to score something that was never meant to
be a trait at all. Given the assignment's whole emphasis is on not
hallucinating judgments it can't support, I'd rather lose recall on a few
ambiguous entries than risk scoring a formatting artifact as if it were a
real personality dimension.

---

## 3. Single safety check vs. a structural, two-layer safety architecture

**The ambiguity/problem:** The brief's actual requirement for this is one
line: "Never hallucinate scores for medical/biological facets -- if any
slip through, force abstention." That's a minimum bar, not a design -- it
doesn't say *where* in the pipeline that check should live, and "if any
slip through" implies the brief's author already expected a single
best-effort check might not be airtight.

**Options considered:**
1. One check, in the prompt: tell the LLM in `scorer.py`'s instructions not
   to score medical/spiritual facets, and trust it to comply.
2. One check, in code: after the LLM responds, scan the results and force
   any medical-tagged facet to `not_observable` regardless of what the model
   said (a single code-level safety net).
3. Two checks: exclude non-observable facets (medical, spiritual, social
   demographic, malformed) from the FAISS index entirely in `embeddings.py`
   -- so they are structurally impossible to retrieve in the first place --
   *and* keep the code-level hard block in `scorer.py` as a second net in
   case a non-observable facet ever reaches that stage some other way.

**Choice made:** Option 3. `audit.py` marks these categories
`conversation_observable = False`, `embeddings.py` filters the DataFrame to
`conversation_observable == True` before it ever builds the index, and
`scorer.py` still separately force-abstains any `medical_biological` facet
that shows up in a batch as a defensive backstop. This is more code to
maintain than option 1 or 2 alone -- there are now two places that need to
agree on which facets are "safe" -- but it means a medical facet has to fail
*two* independent checks to ever get scored, not just one.

**Trade-off:** More engineering complexity for a guarantee I could actually
verify empirically: across both full benchmark runs (`top_k=25` and
`top_k=40`, 30 reference facets each, including the medical-trap and
spiritual-trap conversations that specifically try to bait the system into
scoring `FSH level`, `Basophil count`, and esoteric facets), the
`SAFETY_VIOLATION_scored_non_observable_facet` count was 0 both times. I'd
rather have redundant, verifiable safety than a single check I'm just
hoping the model respects.

---

## 4. How to define "correct" for the benchmark, given scores are ordinal, not exact

**The ambiguity/problem:** The brief says to "compare system output vs
reference labels" and report agreements/abstentions/failure modes, but
doesn't define what counts as agreement. A 1-5 personality score isn't like
a classification label where you're either right or wrong -- if I hand-label
a conversation as `Risktaking: 5` and the system says `4`, that's a much
smaller miss than the system saying `1`. A binary "exact match / no match"
grading would make the benchmark look artificially harsh, but too loose a
tolerance would make it meaningless.

**Options considered:**
1. Binary exact-match only (score must equal my reference score exactly).
2. Exact match plus a tolerance band (e.g. off-by-one still counts as a
   "close" success) -- and separately track *why* something didn't match
   (never retrieved, vs. retrieved-but-wrong-score, vs. abstained when it
   shouldn't have, vs. scored when it should have abstained).
3. Some kind of correlation/ranking metric across the whole run instead of
   per-facet pass/fail.

**Choice made:** Option 2. In `benchmark.py`'s `_evaluate_case()`, I split
outcomes into `exact_agreement`, `close_agreement` (±1 of my reference
score), and then several distinct failure buckets instead of one generic
"wrong": `retrieval_miss` (never even retrieved), `incorrect_abstention`
(system abstained when I expected a score), `incorrect_overconfident_score`
(system scored when I expected abstention), `disagreement` (scored, but off
by more than 1), and the safety-critical `SAFETY_VIOLATION_*` bucket for a
non-observable facet getting scored at all. I skipped option 3 (a single
aggregate correlation number) because with only 30 reference facets across
10 conversations, a single summary statistic would hide exactly the kind of
detail (13 retrieval misses concentrated in specific facets) that turned
out to be the most useful thing the benchmark found.

**Trade-off:** This produces a more complicated report than a simple
"12/30 correct" -- there are 7 possible outcome categories instead of 2 --
but it's the reason I was able to diagnose the retrieval-miss problem at all
instead of just seeing a low pass rate and not knowing why. The cost is
that the grading logic itself (`_evaluate_case()`) is now something a
reviewer has to trust I implemented fairly, since I wrote both the reference
labels and the grading rule.
