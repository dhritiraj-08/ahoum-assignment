# Retrieval Ablation: Does Embedding Text Format Change Recall?

Tests 4 embedding-text formats for the same **316** conversation-observable facets, retrieving at **top_k=25** against the same 10 benchmark conversations `src/benchmark.py` uses, measured against the same **27** retrievable reference facets its force-include mechanism tracks. No LLM calls anywhere in this script -- pure FAISS/sentence-transformers retrieval, same model (`all-MiniLM-L6-v2`) as production, just 4 different text formats instead of 1.

**Why this exists:** `docs/DECISIONS.md` #1 and `docs/DEBUGGING.md` #1 both concluded that embedding-text quality, not `top_k`, is the root cause of retrieval misses -- widening `top_k` from 25 to 40 only recovered 1 of 13 misses. That conclusion was inferred, never directly tested. This script tests it.

## Results

| Approach | Recall | Naturally retrieved |
|---|---|---|
| 1. Bare facet name only | **19%** | 5/27 |
| 2. Facet name + category | **19%** | 5/27 |
| 3. Facet name + scoring anchors (current implementation) | **22%** 🏆 | 6/27 |
| 4. Facet name + scoring anchors + 2 template example phrasings | **19%** | 5/27 |

## Reading the result

The current implementation (approach 3) came out on top -- none of the alternative text formats beat it. That's a real, useful negative result: it means the retrieval-miss problem documented in `DECISIONS.md` #1 isn't fixed by simply changing what generic text gets embedded alongside the facet name. The fix, if there is one at this embedding-model scale, likely needs richer, more conversation-like example text than the crude template in approach 4 -- or a fundamentally different retrieval strategy (a cross-encoder reranker, per `README.md` item 4).

## Per-approach misses

Reference facets each approach failed to naturally retrieve -- useful for spotting whether the SAME facets keep missing across every approach (a harder retrieval problem) or whether different approaches miss different facets (more sensitive to phrasing).

**1. Bare facet name only** -- 22 miss(es):
- `Common-sense` (conversation 1: clear_direct)
- `Doggedness` (conversation 2: ambiguous)
- `Self-Efficacy` (conversation 2: ambiguous)
- `Common-sense` (conversation 2: ambiguous)
- `Emotionalism` (conversation 3: contradictory)
- `Decency` (conversation 3: contradictory)
- `Doggedness` (conversation 4: sarcastic)
- `Self-improvement` (conversation 4: sarcastic)
- `Discontentment` (conversation 4: sarcastic)
- `Compassion` (conversation 5: low_evidence)
- `Risktaking` (conversation 5: low_evidence)
- `Emotionalism` (conversation 5: low_evidence)
- `Compassion` (conversation 6: code_switched)
- `Unassertiveness` (conversation 6: code_switched)
- `Assertiveness and control in relationships` (conversation 6: code_switched)
- `Emotionalism` (conversation 7: medical_trap)
- `Self-improvement` (conversation 8: spiritual_trap)
- `Peacefulness` (conversation 8: spiritual_trap)
- `Discontentment` (conversation 9: high_emotional)
- `Peacefulness` (conversation 9: high_emotional)
- `Cordiality` (conversation 10: professional_formal)
- `Common-sense` (conversation 10: professional_formal)

**2. Facet name + category** -- 22 miss(es):
- `Common-sense` (conversation 1: clear_direct)
- `Doggedness` (conversation 2: ambiguous)
- `Self-Efficacy` (conversation 2: ambiguous)
- `Common-sense` (conversation 2: ambiguous)
- `Emotionalism` (conversation 3: contradictory)
- `Decency` (conversation 3: contradictory)
- `Doggedness` (conversation 4: sarcastic)
- `Self-improvement` (conversation 4: sarcastic)
- `Discontentment` (conversation 4: sarcastic)
- `Compassion` (conversation 5: low_evidence)
- `Risktaking` (conversation 5: low_evidence)
- `Emotionalism` (conversation 5: low_evidence)
- `Compassion` (conversation 6: code_switched)
- `Unassertiveness` (conversation 6: code_switched)
- `Assertiveness and control in relationships` (conversation 6: code_switched)
- `Emotionalism` (conversation 7: medical_trap)
- `Self-improvement` (conversation 8: spiritual_trap)
- `Peacefulness` (conversation 8: spiritual_trap)
- `Discontentment` (conversation 9: high_emotional)
- `Peacefulness` (conversation 9: high_emotional)
- `Cordiality` (conversation 10: professional_formal)
- `Common-sense` (conversation 10: professional_formal)

**3. Facet name + scoring anchors (current implementation)** -- 21 miss(es):
- `Common-sense` (conversation 1: clear_direct)
- `Doggedness` (conversation 2: ambiguous)
- `Self-Efficacy` (conversation 2: ambiguous)
- `Common-sense` (conversation 2: ambiguous)
- `Emotionalism` (conversation 3: contradictory)
- `Decency` (conversation 3: contradictory)
- `Doggedness` (conversation 4: sarcastic)
- `Self-improvement` (conversation 4: sarcastic)
- `Discontentment` (conversation 4: sarcastic)
- `Compassion` (conversation 5: low_evidence)
- `Risktaking` (conversation 5: low_evidence)
- `Emotionalism` (conversation 5: low_evidence)
- `Compassion` (conversation 6: code_switched)
- `Unassertiveness` (conversation 6: code_switched)
- `Emotionalism` (conversation 7: medical_trap)
- `Self-improvement` (conversation 8: spiritual_trap)
- `Peacefulness` (conversation 8: spiritual_trap)
- `Discontentment` (conversation 9: high_emotional)
- `Peacefulness` (conversation 9: high_emotional)
- `Cordiality` (conversation 10: professional_formal)
- `Common-sense` (conversation 10: professional_formal)

**4. Facet name + scoring anchors + 2 template example phrasings** -- 22 miss(es):
- `Common-sense` (conversation 1: clear_direct)
- `Doggedness` (conversation 2: ambiguous)
- `Self-Efficacy` (conversation 2: ambiguous)
- `Common-sense` (conversation 2: ambiguous)
- `Emotionalism` (conversation 3: contradictory)
- `Decency` (conversation 3: contradictory)
- `Doggedness` (conversation 4: sarcastic)
- `Self-improvement` (conversation 4: sarcastic)
- `Discontentment` (conversation 4: sarcastic)
- `Compassion` (conversation 5: low_evidence)
- `Risktaking` (conversation 5: low_evidence)
- `Emotionalism` (conversation 5: low_evidence)
- `Compassion` (conversation 6: code_switched)
- `Unassertiveness` (conversation 6: code_switched)
- `Assertiveness and control in relationships` (conversation 6: code_switched)
- `Emotionalism` (conversation 7: medical_trap)
- `Self-improvement` (conversation 8: spiritual_trap)
- `Peacefulness` (conversation 8: spiritual_trap)
- `Discontentment` (conversation 9: high_emotional)
- `Peacefulness` (conversation 9: high_emotional)
- `Cordiality` (conversation 10: professional_formal)
- `Common-sense` (conversation 10: professional_formal)

## Facets that missed under every approach

These missed regardless of embedding text format -- changing *what* gets embedded won't fix these; they need a different retrieval strategy entirely (wider top_k, a reranker, or hybrid keyword+embedding search):

- `Common-sense`
- `Compassion`
- `Cordiality`
- `Decency`
- `Discontentment`
- `Doggedness`
- `Emotionalism`
- `Peacefulness`
- `Risktaking`
- `Self-Efficacy`
- `Self-improvement`
- `Unassertiveness`

