# Design Decisions

Each entry: the decision, why, and the trade-off accepted. Fill in the
`[FILL IN]` placeholders with your own reasoning/experience once you've
actually run the system -- these are starter drafts, not final answers.

---

## 1. Why local Ollama (llama3.1) instead of a hosted API (OpenAI/Anthropic/etc.)

**Decision:** Use Ollama running llama3.1 locally at `http://localhost:11434`
instead of calling a hosted LLM API.

**Why:**
- Assignment explicitly requires it (local Ollama, llama3.1).
- No per-token API cost or rate limits while iterating on prompts across
  399 facets' worth of test runs.
- Personality/behavioural conversation data can be sensitive -- keeping it
  on-device avoids sending it to a third party at all.
- [FILL IN: any latency/quality trade-offs you personally observed vs. a
  hosted model -- e.g. how much slower batches were on your RTX 4050, any
  JSON-formatting reliability differences you noticed.]

**Trade-off accepted:** [FILL IN -- e.g. weaker instruction-following than a
frontier hosted model, meaning more defensive JSON parsing was needed in
`scorer.py`; slower per-batch latency on consumer GPU hardware.]

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

**Trade-off accepted:** [FILL IN -- e.g. embedding-similarity retrieval can
miss facets that are relevant but phrased very differently from the
conversation (a "retrieval miss", see benchmark report); a cross-encoder
reranker on top of the FAISS candidates would likely improve precision but
adds latency/complexity not justified at 399 facets.]

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
- [FILL IN: once you've actually run `--benchmark`, note whether you saw
  parse_error batches at size 10, and whether a smaller/larger batch size
  changed that in your own testing.]

**Trade-off accepted:** [FILL IN -- e.g. more total LLM calls per
conversation than a single big call would need, which adds latency; on your
hardware, N batches of 10 for top_k=25 took roughly [FILL IN] seconds
end-to-end.]
