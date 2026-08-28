# Design Decisions

Each entry: the decision, why, and the trade-off accepted. Filled in below
based on actually running the system end-to-end and benchmarking it (see
`PROMPT_LOG.md` Entry 2 and `DEBUGGING.md` for how these conclusions were
reached).

---

## 1. Why local Ollama (llama3.1) instead of a hosted API (OpenAI/Anthropic/etc.)

**Decision:** Use Ollama running llama3.1 locally at `http://localhost:11434`
instead of calling a hosted LLM API.

**Why:**
- Assignment explicitly requires it (local Ollama, llama3.1).
- No per-token API cost or rate limits while iterating on prompts across
  399 facets' worth of test runs -- I re-ran the benchmark (30-40 LLM calls
  a pop) several times while tuning `top_k`, which would add up fast on a
  paid API.
- My RTX 4050 (6GB VRAM) is enough to run llama3.1 with GPU acceleration,
  so batches weren't painfully slow -- a full 10-conversation benchmark run
  (30-40 batched calls) finished in a few minutes, not something I'd want
  to sit through if it were CPU-only.
- Works fully offline once the model is pulled -- no dependency on an
  internet connection or an API key during actual grading/demo.
- Personality/behavioural conversation data can be sensitive -- keeping it
  on-device avoids sending it to a third party at all.
- Satisfies the "open-source model" spirit of the assignment instead of
  routing through a closed commercial API.

**Trade-off accepted:** llama3.1 is noticeably less reliable at emitting
clean, complete JSON than a frontier hosted model would be -- that's exactly
why `scorer.py` has a defensive `_extract_json_array()` step and a
`parse_error` status instead of just trusting `json.loads()` on the raw
response. I did hit occasional malformed JSON during testing (see
`DEBUGGING.md` #2), which I don't think I'd have needed to guard against
nearly as much with a hosted frontier model.

---

## 2. Why FAISS + sentence-transformers for retrieval (instead of e.g. sending all facets, or a different retrieval approach)

**Decision:** Embed only the conversation-observable facets with
`all-MiniLM-L6-v2` and retrieve the top-k via a FAISS flat inner-product
(cosine similarity) index, rather than sending all ~399 facets to the LLM
or using a different retrieval method (BM25, a larger embedding model, a
vector DB service, etc).

**Why:**
- The assignment explicitly forbids sending all 399 facets in one prompt --
  retrieval-first is a hard requirement, not just an optimization.
- `all-MiniLM-L6-v2` is small (~80MB), CPU-friendly, and fast enough to
  embed 399 short strings in well under a second, which matters when the
  index needs to be rebuilt whenever the facet CSV changes.
- A flat FAISS index is exact (no approximation error) and trivially fast
  at this scale (hundreds of vectors, not millions) -- no need for IVF/HNSW
  indexing complexity.
- Embedding `"facet name: scoring anchor"` rather than just the bare facet
  name gives the retriever more semantic signal to match against.

**Trade-off accepted:** embedding-similarity retrieval genuinely misses
facets that are relevant but phrased differently from the conversation, and
I have real numbers on this now, not just a hypothetical. I started with
`top_k=25` and the first benchmark run came back with 13 out of 30 reference
facets never even retrieved ("retrieval_miss" in the report) -- things like
`Emotionalism`, `Self-improvement`, `Peacefulness`, and `Decency` just
weren't in the top 25 out of 322 observable facets for conversations that
clearly should have surfaced them.

My first instinct was "just widen the window," so I bumped `top_k` to 40.
That only recovered 1 of the 13 misses (`Compassion` in the code-switched
case) -- 18/30 instead of 17/30 overall. Everything else that missed at 25
was *still* missing at 40, out of 322 total candidates. That told me the
problem isn't really window size, it's the embedding text itself: I'm
currently embedding `"facet name: generic templated scoring anchor"` (see
`generate_scoring_anchor()` in `audit.py`), and that templated anchor text
("1=Very low X; 3=Moderate X; 5=Very high X clearly expressed...") is fairly
generic boilerplate, not naturalistic language -- so a facet like
`Emotionalism` doesn't necessarily land close to a conversation that
*sounds* emotional but doesn't use words that resemble the anchor template.

The real fix, which I didn't have time to build in the 24-hour window, would
be richer per-facet embedding text -- a few example phrasings of what
"scoring evidence" for that facet actually sounds like in conversation,
instead of a generic 1-5 rubric sentence. I chose to document this as a
known limitation with a diagnosed root cause (see `DEBUGGING.md` #1) rather
than rebuild the anchor-generation logic under time pressure and risk
breaking something that was already working.

---

## 3. Why batch size of 10 for LLM scoring

**Decision:** Score at most 10 facets per Ollama call.

**Why:**
- Balances round-trip overhead (fewer, larger batches = less repeated
  prompt preamble) against JSON reliability (asking a local 8B-class model
  for a large structured array in one shot increases truncation/malformed
  output risk).
- Keeps a single bad/ambiguous facet from invalidating an entire large
  batch -- with batches of 10, a parse failure only costs 10 facets' worth
  of results, not all 25 retrieved facets.
- It's basically a context-window / throughput trade-off. Ollama running
  llama3.1 locally has a limited effective context before instruction-
  following degrades, and asking for a big JSON array back makes that worse
  in both directions at once (long prompt in, long structured output out).
  Batch size 10 keeps both the input (facet list + anchors) and the output
  (10 JSON objects) small enough that the model doesn't start truncating or
  losing track of which facet it's on. Go much smaller (like batches of 2-3)
  and you're paying the fixed prompt preamble (conversation text + rules)
  over and over for barely any facets per call, which is wasteful.
- I didn't see a single parse_error in either benchmark run (top_k=25 or
  top_k=40) at batch size 10, which suggests 10 is comfortably inside
  llama3.1's reliable range on this hardware. I didn't have time to
  systematically test batch=15 or batch=20 to find the exact breaking point,
  but 10 felt conservative enough to not have to worry about it.

**Trade-off accepted:** more total LLM calls per conversation than one giant
call would need -- at `top_k=25` that's 3 batches per conversation, and at
`top_k=40` (the value I ended up keeping after the retrieval_miss
investigation, see Decision 2) that's 4 batches. That adds latency: the
10-conversation benchmark (30-40 batched calls total) took a few minutes to
run end-to-end on my RTX 4050. I'm fine with that trade -- for this
assignment, reliable JSON per batch matters a lot more than shaving off a
minute of wall-clock time.
