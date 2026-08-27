"""
embeddings.py
-------------
Builds a semantic search index over ONLY the conversation-observable facets
(the output of audit.py) and exposes retrieve_relevant_facets() so the
pipeline never has to send all ~399 facets to the LLM at once.

WHY: sending 399 facets in one prompt would blow past useful context,
produce unreliable JSON, and waste tokens scoring facets that have nothing
to do with the conversation. Instead we embed each facet's name + its 1-5
scoring anchor (so semantically similar facets with different anchors are
still distinguishable) with a small, fast sentence-transformer model that
comfortably runs on a 6GB laptop GPU (or even CPU), and use FAISS for
nearest-neighbour search. Given a conversation, we retrieve only the top_k
most relevant facets and hand THOSE to the LLM scorer in small batches.
"""

import json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENRICHED_CSV_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss_index.bin"
FACETS_JSON_PATH = PROJECT_ROOT / "outputs" / "observable_facets.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small (~80MB), fast, good enough for retrieval

# Module-level caches so repeated calls (e.g. scoring many conversations in a
# benchmark run) don't reload the model / index from disk every time.
_model = None
_index = None
_facets_meta = None


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


if __name__ == "__main__":
    build_index()
    # quick smoke test
    sample = "I decided to invest all my savings in a risky new venture without hesitation."
    for r in retrieve_relevant_facets(sample, top_k=5):
        print(f"{r['similarity']:.3f}  {r['facet_normalized']}")
