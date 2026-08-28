"""
pipeline.py
-----------
Wires embeddings.py (retrieval) and scorer.py (LLM scoring) into one
callable function: run_pipeline(conversation) -> structured result dict.

This is the single entry point the rest of the project (main.py,
benchmark.py) should call -- it owns the retrieve -> batch -> score ->
dedupe -> save flow so that logic lives in exactly one place.
"""

import json
from datetime import datetime
from pathlib import Path

try:
    from src.embeddings import retrieve_relevant_facets
    from src.scorer import score_facets, BATCH_SIZE
except ImportError:
    # allow running as a script from inside src/ too
    from embeddings import retrieve_relevant_facets
    from scorer import score_facets, BATCH_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TOP_K_DEFAULT = 40


def run_pipeline(conversation: str, top_k: int = TOP_K_DEFAULT, save_output: bool = True) -> dict:
    """
    Runs the full retrieve -> score -> aggregate flow for one conversation.

    Steps:
      1. Retrieve top_k semantically relevant, conversation-observable facets
         via FAISS (embeddings.py). This is what keeps us from ever sending
         all 399 facets to the LLM.
      2. Split into batches of <=10 and score each batch with scorer.py.
      3. Deduplicate results by facet name (in case retrieval or batching
         ever produces a repeat).
      4. Return a structured summary dict, and optionally persist it as
         timestamped JSON under outputs/.

    Never raises on a missing/empty conversation -- returns an empty-ish
    result instead, since this may be called in a loop (benchmark.py) where
    one bad input shouldn't kill the whole run.
    """
    result = {
        "conversation_snippet": (conversation or "")[:200],
        "total_facets_retrieved": 0,
        "scored": 0,
        "abstained": 0,
        "results": [],
    }

    if not conversation or not conversation.strip():
        result["error"] = "Empty conversation text; nothing to score."
        return result

    # Step 1: retrieval
    try:
        retrieved = retrieve_relevant_facets(conversation, top_k=top_k)
    except Exception as e:
        result["error"] = f"Retrieval failed: {e}"
        return result

    result["total_facets_retrieved"] = len(retrieved)
    if not retrieved:
        return result

    # Step 2: batch + score
    try:
        raw_results = score_facets(retrieved, conversation, batch_size=BATCH_SIZE)
    except Exception as e:
        result["error"] = f"Scoring failed: {e}"
        result["results"] = []
        return result

    # Step 3: dedupe by facet name, keeping the first occurrence
    seen = set()
    deduped = []
    for r in raw_results:
        name = r.get("facet", "")
        if name in seen:
            continue
        seen.add(name)
        deduped.append(r)

    scored_count = sum(1 for r in deduped if r.get("status") == "scored")
    abstained_count = sum(1 for r in deduped if r.get("status") in ("insufficient_evidence", "not_observable"))

    result["scored"] = scored_count
    result["abstained"] = abstained_count
    result["parse_errors"] = sum(1 for r in deduped if r.get("status") == "parse_error")
    result["results"] = deduped

    if save_output:
        try:
            _save_result(result)
        except Exception as e:
            # Don't fail the whole call just because we couldn't write to disk.
            result["save_warning"] = f"Could not save output JSON: {e}"

    return result


def _save_result(result: dict) -> Path:
    """Save a pipeline result dict to outputs/pipeline_output_{timestamp}.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"pipeline_output_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved pipeline output to {out_path}")
    return out_path


if __name__ == "__main__":
    sample = (
        "I don't really think twice before jumping into new ventures. "
        "Last month I put my savings into a startup a friend was launching, "
        "no real research, just gut feeling."
    )
    out = run_pipeline(sample)
    print(json.dumps(out, indent=2))
