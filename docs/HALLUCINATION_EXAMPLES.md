# Anti-Hallucination Examples

Three real scenarios showing where a naive "just ask the LLM to score
everything" system would confidently produce a wrong, unsafe, or
unverifiable score -- and where this system abstains instead. All facet
names below are copied verbatim from `outputs/enriched_facets.csv` (i.e.
they're real rows in the actual 399-facet CSV, not made-up examples), so
you can `grep` them yourself to confirm.

The mechanism behind all three is the same **two-layer safety design**
described in `DECISIONS.md` #3:

1. **Audit-time filter** (`src/audit.py` -> `src/embeddings.py`): facets
   classified as `medical_biological`, `spiritual_esoteric`, or
   `social_demographic` are marked `conversation_observable = False` during
   the audit step, and `embeddings.py` only ever builds the FAISS index
   from `conversation_observable == True` rows. That means these facets are
   **structurally absent from the retrieval index** -- there is no
   embedding vector for them to be matched against, so FAISS can never
   return them as a candidate, no matter how relevant the conversation
   sounds.
2. **Scorer-time hard block** (`src/scorer.py`): as a second, independent
   check, if a `medical_biological` facet ever reached the scorer anyway
   (e.g. through a future code path that bypasses retrieval), it is forced
   to `"status": "not_observable"` before the LLM is even asked, rather
   than trusting the LLM's own judgment to decline.

A naive system -- one that skips retrieval filtering and just prompts the
LLM with "score this facet from 1-5 based on the conversation" for every
facet in the CSV -- has neither layer, and will produce a confident-looking
integer score every time, because that's what instruction-following LLMs
tend to do when asked a direct question with a required answer format.

---

## Example A: Medical/biological facet from a vague symptom mention

**Conversation snippet:**
> "I've been feeling really tired and gaining weight lately. Not sure
> what's going on, honestly."

**What a naive LLM would score:** Given a direct prompt like "score
`Basophil count` 1-5 based on this conversation," an LLM will often
pattern-match "tired + weight gain" to "possible thyroid/blood-related
issue" (these are genuinely common thyroid-disorder symptoms) and produce
something like:

```json
{"facet": "Basophil count", "score": 4, "status": "scored", "confidence": "medium", "evidence": "Fatigue and weight gain suggest possible thyroid dysfunction, which can affect basophil levels."}
```

or the same for `Parathyroid-hormone level`. This is wrong on two levels:
(1) fatigue and weight gain are extremely non-specific symptoms with dozens
of possible causes having nothing to do with basophils or parathyroid
hormone, and (2) even if the model's medical reasoning were sound, **a
blood cell count or hormone level is a lab-measured number, not something
inferable from how someone describes feeling.** There is no conversational
evidence on Earth that could responsibly produce a specific 1-5 score for
an actual clinical measurement.

**What our system does:** Neither `Basophil count` nor
`Parathyroid-hormone level` is ever retrieved for this conversation.
`outputs/enriched_facets.csv` shows both are classified `category =
medical_biological`, `conversation_observable = False`. `embeddings.py`
filters on `conversation_observable == True` before building the FAISS
index, so these two facets have no embedding vector in the index at all --
`retrieve_relevant_facets()` cannot return them regardless of how
semantically close "tired and gaining weight" is to "thyroid" or "basophil"
in embedding space. The pipeline's `results` list for this conversation
would simply never contain them.

**Why our system is correct:** This is the audit-time filter doing its
job before the LLM is ever consulted -- the facet is unreachable, not
merely discouraged. Even in the hypothetical case where `top_k` were set
so high it should logically pull in every observable facet, `Basophil
count` and `Parathyroid-hormone level` still couldn't surface, because they
were excluded from the candidate pool entirely back at the embedding-build
step, not filtered post-hoc. And as a second, independent safeguard, even
if some other future retrieval path fed a `medical_biological` facet into
`scorer.py` directly, `score_facet_batch()`'s explicit category check would
force it to `"status": "not_observable"` before any prompt asking for a
score was even sent to Ollama.

---

## Example B: Spiritual/esoteric practice inferred from an unrelated routine

**Conversation snippet:**
> "I meditate every morning and follow my moon sign."

**What a naive LLM would score:** This sentence is dense with spiritual
vocabulary ("meditate," "moon sign"), and an LLM asked to score every
spirituality-adjacent facet in the CSV will tend to treat that vocabulary as
license to infer specifics well beyond what was actually said:

```json
{"facet": "Energy-healing practice: Reiki sessions / year", "score": 3, "status": "scored", "confidence": "medium", "evidence": "The person practices meditation and astrology, suggesting general engagement with alternative spiritual practices including possibly Reiki."}
```

```json
{"facet": "I Ching hexagram 36 resonance level", "score": 2, "status": "scored", "confidence": "low", "evidence": "No direct mention, but general spiritual interest implied."}
```

Both are fabrications. The person never mentioned Reiki, energy healing, or
the I Ching at all -- the model is generalizing from "this person seems
spiritual" to "therefore I can estimate their engagement with any specific
spiritual practice," which is exactly the kind of stereotype-driven
hallucination a facet-scoring system should not produce. Morning meditation
and following a moon sign say nothing about Reiki session frequency or I
Ching hexagram resonance specifically -- these are distinct, unrelated
practices that happen to share a broad "spiritual/esoteric" flavor.

**What our system does:** Both facets are classified `category =
spiritual_esoteric`, `conversation_observable = False` in
`outputs/enriched_facets.csv`, with `abstention_reason`: "Requires
self-reported practice history/records (e.g. session counts, ritual
frequency) that a short conversation cannot reliably establish." Like
Example A, they're excluded from the FAISS index at build time -- they
never enter `retrieve_relevant_facets()`'s candidate pool, so they can't be
retrieved even though "meditate" and "moon sign" would likely embed
reasonably close to spiritual-practice-related facet text in general.

**Why our system is correct:** The audit-time category filter doesn't
distinguish "mentions something vaguely spiritual" from "gives specific
evidence about this exact practice" -- it doesn't need to, because it
removes the entire category from being scorable at all, on the grounds that
practice-frequency facets (session counts, ritual counts, resonance levels)
require self-reported records a short conversation snippet can't establish,
regardless of how spiritually-themed the conversation sounds. This is a
deliberately blunter, safer rule than trying to teach the LLM "only infer
Reiki specifically if Reiki is mentioned" -- which would still be
vulnerable to the model conflating adjacent practices under prompt pressure
to produce *some* answer for every facet it's asked about.

---

## Example C: Demographic count that requires external records, not self-report

**Conversation snippet:**
> "I travel a lot for work, been to 15 countries."

**What a naive LLM would score:** This one is the trickiest of the three
because the person is stating something that sounds like exactly the fact
being asked about:

```json
{"facet": "Passport-stamps count", "score": 5, "status": "scored", "confidence": "high", "evidence": "The person explicitly states they have traveled to 15 countries, indicating a high passport-stamps count."}
```

This looks reasonable at first glance -- 15 countries does sound like a lot
of stamps -- but it's still a bad inference to bake into a scoring system:
"15 countries" is a self-reported, unverified claim, not a passport-stamp
count, and the two aren't even the same unit (some countries require no
stamp, some visits generate multiple stamps, land-border crossings and
customs unions like Schengen often produce zero stamps regardless of
countries visited). A system that confidently converts "self-reported
countries visited" into a specific 1-5 "passport stamps" score is
laundering an unverifiable, approximate claim into a precise-looking
number, which is exactly the kind of overconfidence the assignment's
abstention requirement is meant to prevent.

**What our system does:** `Passport-stamps count` is classified `category =
social_demographic`, `conversation_observable = False`. It's excluded from
the FAISS index the same way as Examples A and B, so it's never retrieved
for this conversation regardless of how directly the conversation seems to
be "about" travel.

**Why our system is correct:** Demographic count facets like this one are
things that, per the `abstention_reason` recorded in
`enriched_facets.csv`, "require a factual/demographic record (counts, logs,
official data) rather than something inferable from conversational tone or
content." Even a conversation that directly states a related number
("15 countries") doesn't make the specific facet being asked about
("passport stamps") verifiable -- self-report of a related-but-different
quantity is not the same as evidence for the actual facet, and a system
that treats them as interchangeable will silently smuggle in fabricated
precision. Filtering the whole category out at audit time avoids needing
the LLM to draw that self-report-vs-verified-record line correctly on a
case-by-case basis, which is a much harder and more error-prone judgment
call to get an LLM to make reliably under prompt pressure than "would you
like to abstain."

---

## Example D: Clinical facet misclassified as personality trait -- caught by live adversarial testing, not code review

**Conversation snippet:**
> "I am feeling low"

**What a naive system would score:** A system without proper clinical
keyword filtering would score `Depression Symptoms` 5/5 and `Depression:
Feelings of sadness and hopelessness` 5/5 with high confidence. This
isn't a hypothetical -- it's exactly what *this* system did before
`DEBUGGING.md` #4's fix. Both facets, along with 4 others (`Depression
(DEP)`, `Burnout Symptoms`, `Hypomania (Ma)`, `Hysteria (Hy)`), were
originally classified `category = personality_trait`,
`conversation_observable = True`, which meant they had real embedding
vectors sitting in the FAISS index and could be retrieved and scored
exactly like `Compassion` or `Risktaking` -- no different treatment at
all.

**What our system does now:** All 6 are classified
`category = medical_biological`, `conversation_observable = False`.
Running `retrieve_relevant_facets("I am feeling low", top_k=40)` confirms
none of the 6 appear among the 40 retrieved candidates -- they are
structurally unreachable, exactly the same as `FSH level` and `Basophil
count` in Examples A-C.

**Why this example is different from A, B, and C:** Examples A, B, and C
demonstrate the safety architecture working as designed from the start --
a category-based rule catches a facet before it's ever embedded, and that
rule was correct on day one. Example D demonstrates something more
important for real-world deployment: **the safety architecture has a
measured error rate, and that error rate was only discoverable by
actually using the system adversarially, not by reading the code.** The
`MEDICAL_KEYWORDS` list in `audit.py` looked complete on inspection -- it
already included `"diagnosis"`, `"disorder"`, `"syndrome"`, `"clinical"`.
None of the 6 misclassified facets contain any of those words. MMPI
clinical-scale abbreviations (`DEP`, `Ma`, `Hy`) and symptom-cluster
names like `"Burnout Symptoms"` slipped through the keyword rules
entirely -- not because the rules were poorly written, but because a
keyword list built around lab-test and diagnosis-record vocabulary has
no natural reason to also cover named psychiatric conditions unless
someone specifically thinks to add them. The fix (a dedicated
psychiatric keyword block) was straightforward once the gap was found --
but finding it required typing "I am feeling low" into the live UI and
watching `Depression Symptoms` come back scored 5/5, not reading
`audit.py` and noticing the gap by inspection. That's the whole argument
for why `hallucination_demo/` exists as a structural, repeatable test
rather than a one-time manual check: a classifier that looks complete on
review can still have a real, scoreable hole in it, and the only
reliable way found so far to surface that hole is adversarial use, not
code review.
