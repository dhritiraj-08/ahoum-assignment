"""
embeddings.py
-------------
Builds a semantic search index over ONLY the conversation-observable facets
(the output of audit.py) and exposes retrieve_relevant_facets() (pure FAISS)
and retrieve_hybrid() (FAISS + BM25, combined via Reciprocal Rank Fusion) so
the pipeline never has to send all ~399 facets to the LLM at once.

WHY: sending 399 facets in one prompt would blow past useful context,
produce unreliable JSON, and waste tokens scoring facets that have nothing
to do with the conversation. Instead we embed each facet's name + its 1-5
scoring anchor (so semantically similar facets with different anchors are
still distinguishable) with a small, fast sentence-transformer model that
comfortably runs on a 6GB laptop GPU (or even CPU), and use FAISS for
nearest-neighbour search. Given a conversation, we retrieve only the top_k
most relevant facets and hand THOSE to the LLM scorer in small batches.

WHY HYBRID (FAISS + BM25): `eval/retrieval_ablation.py` and
`docs/DECISIONS.md` #1 both found that retrieval recall, not scoring
quality, is this project's biggest measured weakness (26% recall vs. 59%
scoring accuracy) -- and that changing the embedding *text* alone (more
context, template example phrasings) didn't fix it. Dense embedding
similarity (FAISS) is good at semantic/paraphrase matches but can miss
exact-keyword matches that don't paraphrase well; BM25 is the reverse --
good at exact lexical overlap, blind to paraphrase. Combining both via
Reciprocal Rank Fusion is a standard retrieval-augmentation technique
precisely because the two methods' failure modes don't overlap much. See
`docs/DECISIONS.md` for the measured before/after recall this actually
produced -- this is described as a hope, not assumed as a result.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss-cpu is not installed. Run: pip install -r requirements.txt"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is not installed. Run: pip install -r requirements.txt"
    ) from e

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError(
        "rank-bm25 is not installed. Run: pip install -r requirements.txt"
    ) from e

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENRICHED_CSV_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss_index.bin"
FACETS_JSON_PATH = PROJECT_ROOT / "outputs" / "observable_facets.json"
BM25_INDEX_PATH = PROJECT_ROOT / "outputs" / "bm25_index.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small (~80MB), fast, good enough for retrieval
RRF_K = 60  # standard Reciprocal Rank Fusion constant; also this project's hybrid candidate-pool size per method

# Module-level caches so repeated calls (e.g. scoring many conversations in a
# benchmark run) don't reload the model / index from disk every time.
_model = None
_index = None
_facets_meta = None
_bm25_index = None


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for BM25 -- lowercase, alphanumeric runs only.
    BM25 is a lexical/keyword method, so this doesn't need to be anything
    fancier than what its own paper and most implementations assume."""
    return re.findall(r"\w+", text.lower())


def _get_model() -> "SentenceTransformer":
    """Lazy-load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def build_index(enriched_csv_path: Path = ENRICHED_CSV_PATH) -> None:
    """
    Reads outputs/enriched_facets.csv, filters to conversation_observable ==
    True, embeds "facet_name: scoring_anchor" for each row, and writes both
    the FAISS index and the facet metadata (so we can map index positions
    back to facet info at retrieval time) to outputs/.
    """
    try:
        df = pd.read_csv(enriched_csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{enriched_csv_path} not found. Run `python main.py --audit` first."
        ) from e

    observable = df[df["conversation_observable"] == True].copy()  # noqa: E712
    observable = observable.fillna("")

    if observable.empty:
        raise ValueError("No conversation-observable facets found -- check audit.py output.")

    # Combine facet name + its scoring anchor into one embedding text so the
    # retriever picks up on the *meaning/scale* of the facet, not just its name.
    texts = [
        f"{row['facet_normalized']}: {row['scoring_anchors']}"
        for _, row in observable.iterrows()
    ]

    model = _get_model()
    print(f"Embedding {len(texts)} observable facets with {EMBEDDING_MODEL_NAME} ...")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    # Inner product on normalized vectors == cosine similarity.
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    meta = observable[[
        "facet_normalized", "category", "sensitivity", "scoring_anchors", "abstention_reason",
    ]].to_dict(orient="records")
    with open(FACETS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) to {FAISS_INDEX_PATH}")
    print(f"Saved facet metadata to {FACETS_JSON_PATH}")

    # Build the BM25 index over the EXACT SAME texts, in the EXACT SAME
    # order, as the FAISS embeddings above -- retrieve_hybrid() relies on
    # position i in the BM25 index and position i in FACETS_JSON_PATH's
    # meta list referring to the same facet.
    tokenized_docs = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_docs)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index ({len(tokenized_docs)} docs) to {BM25_INDEX_PATH}")


def _load_index_and_meta():
    """Lazy-load the FAISS index + facet metadata JSON, cached at module level."""
    global _index, _facets_meta
    if _index is None or _facets_meta is None:
        if not FAISS_INDEX_PATH.exists() or not FACETS_JSON_PATH.exists():
            raise FileNotFoundError(
                "FAISS index/metadata not found. Run `python main.py --embed` first."
            )
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FACETS_JSON_PATH, "r", encoding="utf-8") as f:
            _facets_meta = json.load(f)
    return _index, _facets_meta


def retrieve_relevant_facets(conversation_text: str, top_k: int = 25) -> list[dict]:
    """
    Given a raw conversation string, returns the top_k most semantically
    relevant conversation-observable facets (as metadata dicts) using cosine
    similarity search over the FAISS index built by build_index().

    Robust to a blank/whitespace-only conversation (returns []) and to a
    missing index (raises a clear, actionable error rather than crashing
    deep inside numpy/faiss).
    """
    if not conversation_text or not conversation_text.strip():
        return []

    index, meta = _load_index_and_meta()
    top_k = max(1, min(top_k, index.ntotal))

    model = _get_model()
    query_vec = model.encode([conversation_text], convert_to_numpy=True, normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):  # faiss pads with -1 if fewer than top_k results exist
            continue
        facet = dict(meta[idx])
        facet["similarity"] = float(score)
        results.append(facet)
    return results


def _load_bm25_index() -> "BM25Okapi":
    """Lazy-load the BM25 index, cached at module level. Raises
    FileNotFoundError (not caught here -- retrieve_hybrid() catches it to
    implement the pure-FAISS fallback) if --embed hasn't been run since
    this feature was added, or was run against an older embeddings.py."""
    global _bm25_index
    if _bm25_index is None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"{BM25_INDEX_PATH} not found. Run `python main.py --embed` first."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_index = pickle.load(f)
    return _bm25_index


def _retrieve_bm25_candidates(conversation_text: str, top_k: int) -> list[dict]:
    """BM25 analog of retrieve_relevant_facets(): same facet metadata/order
    (_load_index_and_meta()'s meta list) but ranked by lexical/keyword
    overlap instead of embedding similarity."""
    bm25 = _load_bm25_index()
    _, meta = _load_index_and_meta()

    query_tokens = _tokenize(conversation_text)
    scores = bm25.get_scores(query_tokens)

    top_k = max(1, min(top_k, len(meta)))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        idx = int(idx)
        if idx < 0 or idx >= len(meta):
            continue
        facet = dict(meta[idx])
        facet["bm25_score"] = float(scores[idx])
        results.append(facet)
    return results


def retrieve_hybrid(conversation_text: str, top_k: int = 40) -> list[dict]:
    """
    Hybrid dense + lexical retrieval: combines FAISS (semantic/embedding
    similarity) and BM25 (lexical/keyword overlap) rankings via Reciprocal
    Rank Fusion (RRF) instead of using either alone.

    RRF score per facet = 1/(rank_faiss + RRF_K) + 1/(rank_bm25 + RRF_K),
    where rank is the facet's 1-indexed position in that method's own
    top-60 candidate list (0 if it doesn't appear in that list at all --
    i.e. only lists it actually appears in contribute to its score). This
    is the standard Cormack et al. RRF formula; RRF_K=60 is also used as
    each method's own candidate-pool size here, which is a common (not
    required) choice -- a facet ranked #1 by one method and completely
    absent from the other can still surface near the top of the fused
    list, which is exactly the point: the two methods' blind spots don't
    overlap much, so a facet either method is confident about should
    still surface.

    Falls back to pure retrieve_relevant_facets() (FAISS only) if the
    BM25 index hasn't been built yet (e.g. outputs/ from before this
    feature existed) -- never hard-fails just because BM25 is missing.
    """
    if not conversation_text or not conversation_text.strip():
        return []

    faiss_candidates = retrieve_relevant_facets(conversation_text, top_k=RRF_K)

    try:
        bm25_candidates = _retrieve_bm25_candidates(conversation_text, top_k=RRF_K)
    except FileNotFoundError:
        return faiss_candidates[:top_k]

    faiss_ranks = {f["facet_normalized"]: i + 1 for i, f in enumerate(faiss_candidates)}
    bm25_ranks = {f["facet_normalized"]: i + 1 for i, f in enumerate(bm25_candidates)}

    # Metadata for any facet that showed up in either list -- prefer the
    # FAISS copy (has "similarity") but fall back to the BM25 copy (has
    # "bm25_score") for facets FAISS's top-60 didn't surface at all.
    meta_by_name = {f["facet_normalized"]: f for f in faiss_candidates}
    for f in bm25_candidates:
        meta_by_name.setdefault(f["facet_normalized"], f)

    all_names = set(faiss_ranks) | set(bm25_ranks)
    rrf_scores = {}
    for name in all_names:
        score = 0.0
        if name in faiss_ranks:
            score += 1.0 / (faiss_ranks[name] + RRF_K)
        if name in bm25_ranks:
            score += 1.0 / (bm25_ranks[name] + RRF_K)
        rrf_scores[name] = score

    ranked_names = sorted(all_names, key=lambda n: rrf_scores[n], reverse=True)
    top_k = max(1, min(top_k, len(ranked_names)))

    results = []
    for name in ranked_names[:top_k]:
        facet = dict(meta_by_name[name])
        facet["rrf_score"] = rrf_scores[name]
        results.append(facet)
    return results


if __name__ == "__main__":
    build_index()
    # quick smoke test
    sample = "I decided to invest all my savings in a risky new venture without hesitation."
    print("\nFAISS-only:")
    for r in retrieve_relevant_facets(sample, top_k=5):
        print(f"  {r['similarity']:.3f}  {r['facet_normalized']}")
    print("\nHybrid (FAISS + BM25, RRF):")
    for r in retrieve_hybrid(sample, top_k=5):
        print(f"  {r['rrf_score']:.5f}  {r['facet_normalized']}")
