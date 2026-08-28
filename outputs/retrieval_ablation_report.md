# Retrieval Ablation: Does Embedding Text Format Change Recall?

Tests 4 embedding-text formats for the same **316** conversation-observable facets, at **k in [10, 25, 40, 100]**, against the same 10 benchmark conversations `src/benchmark.py` uses, measured against the same **27** retrievable reference facets its force-include mechanism tracks. No LLM calls anywhere in this script -- pure FAISS/sentence-transformers retrieval, same model (`all-MiniLM-L6-v2`) as production, just 4 different text formats instead of 1, checked across multiple candidate-window sizes so a conclusion doesn't rest on a single k.

**Why this exists:** `docs/DECISIONS.md` #1 and `docs/DEBUGGING.md` #1 both concluded that embedding-text quality, not `top_k`, is the root cause of retrieval misses -- widening `top_k` from 25 to 40 only recovered 1 of 13 misses. That conclusion was inferred, never directly tested. This script tests it, and now tests it across multiple k values (10/25/40/100) rather than just one, so a claim like "approach 3 wins" can say at *which* k, or all of them.

## Results: recall by approach x k

| Approach | k=10 | k=25 | k=40 | k=100 |
|---|---|---|---|---|
| 1. Bare facet name only | 19% 🏆 | 19% | 26% 🏆 | 59% 🏆 |
| 2. Facet name + category | 11% | 19% | 22% | 52% |
| 3. Facet name + scoring anchors (current implementation) | 11% | 22% 🏆 | 26% | 56% |
| 4. Facet name + scoring anchors + 2 template example phrasings | 11% | 19% | 19% | 56% |

## Reading the result

The winning approach isn't consistent across every k tested -- at k=10, **1. Bare facet name only** wins; at k=25, **3. Facet name + scoring anchors (current implementation)** wins; at k=40, **1. Bare facet name only** wins; at k=100, **1. Bare facet name only** wins. That itself is worth knowing: it means the choice of embedding text format and the choice of retrieval window interact, rather than one strictly dominating regardless of the other.

## Per-approach misses at k=25

Detailed miss list at k=25 only (to keep this readable -- the summary table above already covers all 4 k values). Useful for spotting whether the SAME facets keep missing across every approach (a harder retrieval problem) or whether different approaches miss different facets (more sensitive to phrasing).

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

## Facets that missed under every approach (at k=25)

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

