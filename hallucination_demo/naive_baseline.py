"""
naive_baseline.py
------------------
A DELIBERATELY UNSAFE facet scorer. This file exists only to produce a
real, honest comparison for hallucination_demo/run_comparison.py -- it is
NOT the reference for how to build this system. src/pipeline.py is that
reference. Do not import this module from anywhere outside hallucination_demo/.

WHAT THIS SKIPS, ON PURPOSE, RELATIVE TO THE REAL (SAFE) SYSTEM:

1. Retrieval-time filtering. src/embeddings.py only ever builds its FAISS
   index from facets where audit.py marked conversation_observable == True
   -- medical_biological, spiritual_esoteric, social_demographic, and
   header_or_malformed rows are excluded before embedding even happens, so
   they're structurally unreachable by retrieval. This file builds its
   index from ALL 399 raw facets instead, exactly those categories
   included, because that's what a developer gets if they wire up
   FAISS + sentence-transformers over a facet CSV without first running it
   through any category-aware audit step.

2. Scorer-time hard block. src/scorer.py forces any medical_biological
   facet to "not_observable" before the LLM is ever asked, regardless of
   what the model would have said. This file has no such check -- whatever
   gets retrieved gets sent straight to the LLM, and whatever the LLM
   returns is trusted as-is, medical or not.

Everything else (the embedding model, FAISS mechanics, the LLM prompt
format, JSON parsing, the Ollama/Groq backend) is reused as-is from src/,
so the comparison in run_comparison.py isolates exactly these two missing
safety layers as the only variables -- not a difference in retrieval
quality or prompt engineering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu is not installed. Run: pip install -r requirements.txt") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("sentence-transformers is not installed. Run: pip install -r requirements.txt") from e

from src.scorer import BATCH_SIZE, _build_prompt, _call_llm, _extract_json_array, _sanitize_result

ENRICHED_CSV_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level caches, same pattern as src/embeddings.py, so repeated calls
# across the 3 demo conversations don't rebuild the index each time.
_model = None
_index = None
_all_facets_meta = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _build_naive_index() -> None:
    """
    Builds a FAISS index over ALL 399 facets in outputs/enriched_facets.csv
    -- NOT filtered by conversation_observable. This is the single line
    that differs from src/embeddings.py's build_index() and is the root
    cause of every hallucination this demo produces: nothing here excludes
    medical_biological / spiritual_esoteric / social_demographic /
    header_or_malformed facets from ever being retrieved.
    """
    global _index, _all_facets_meta

    if not ENRICHED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{ENRICHED_CSV_PATH} not found. Run `python main.py --audit` first "
            "(this demo reuses the enriched CSV's category labels for reporting, "
            "even though it deliberately ignores conversation_observable for retrieval)."
        )

    df = pd.read_csv(ENRICHED_CSV_PATH).fillna("")
    texts = [f"{row['facet_normalized']}: {row['scoring_anchors']}" for _, row in df.iterrows()]

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    _index = index
    _all_facets_meta = df.to_dict(orient="records")


def naive_retrieve(conversation: str, top_k: int = 25) -> list[dict]:
    """
    Same FAISS search mechanics as src/embeddings.py's
    retrieve_relevant_facets() -- but over the UNFILTERED index built
    above, so medical/spiritual/demographic/malformed facets are fully
    eligible to be returned if they're semantically close to the
    conversation, which is exactly what happens in practice.
    """
    if _index is None:
        _build_naive_index()

    if not conversation or not conversation.strip():
        return []

    model = _get_model()
    query_vec = model.encode([conversation], convert_to_numpy=True, normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    top_k = max(1, min(top_k, _index.ntotal))
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_all_facets_meta):
            continue
        facet = dict(_all_facets_meta[idx])
        facet["similarity"] = float(score)
        results.append(facet)
    return results


def naive_score_batch(facets_batch: list[dict], conversation: str) -> list[dict]:
    """
    Scores one batch of facets with NO safety gate: no medical hard-block,
    no category check of any kind. Every facet handed to this function is
    sent to the LLM and scored purely based on what the model decides.

    Reuses src/scorer.py's prompt-building, backend dispatch, and JSON
    parsing verbatim -- the only thing missing, deliberately, is the
    `if f.get("category") == "medical_biological": force abstain` block
    that src/scorer.py's real score_facet_batch() has.
    """
    if not facets_batch:
        return []

    prompt = _build_prompt(facets_batch, conversation)
    expected_names = {f["facet_normalized"] for f in facets_batch}

    try:
        raw_text = _call_llm(prompt)
    except Exception as e:
        return [
            {
                "facet": f["facet_normalized"],
                "score": None,
                "status": "parse_error",
                "confidence": "low",
                "evidence": f"LLM call failed: {e}",
            }
            for f in facets_batch
        ]

    results_by_name: dict[str, dict] = {}
    try:
        parsed_items = _extract_json_array(raw_text)
        for item in parsed_items:
            if not isinstance(item, dict):
                continue
            sanitized = _sanitize_result(item, expected_names)
            if sanitized["facet"] in expected_names:
                # No safety net here -- whatever the model said, we keep it.
                results_by_name[sanitized["facet"]] = sanitized
    except Exception:
        pass  # anything missing falls through to the parse_error fill-in below

    final_results = []
    for f in facets_batch:
        name = f["facet_normalized"]
        if name in results_by_name:
            final_results.append(results_by_name[name])
        else:
            final_results.append(
                {
                    "facet": name,
                    "score": None,
                    "status": "parse_error",
                    "confidence": "low",
                    "evidence": "Model did not return a result for this facet.",
                }
            )
    return final_results


def naive_run_pipeline(conversation: str, top_k: int = 25) -> list[dict]:
    """Full naive pipeline: retrieve top_k from the UNFILTERED 399-facet
    corpus, batch at BATCH_SIZE (same as the real system), score -- no
    observability gate, no medical hard-block, anywhere."""
    retrieved = naive_retrieve(conversation, top_k=top_k)
    all_results = []
    for i in range(0, len(retrieved), BATCH_SIZE):
        batch = retrieved[i : i + BATCH_SIZE]
        all_results.extend(naive_score_batch(batch, conversation))
    return all_results


if __name__ == "__main__":
    sample = (
        "I've been so tired lately and gaining weight. My doctor mentioned "
        "my hormone levels might be off. I've been feeling really down about it."
    )
    print("Naive baseline smoke test (no safety gate) -- expect medical facets to be retrieved and possibly scored:\n")
    for r in naive_run_pipeline(sample, top_k=10):
        print(f"{r['status']:<22} {r['facet']:<40} score={r['score']}")
