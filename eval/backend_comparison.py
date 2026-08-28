"""
eval/backend_comparison.py
---------------------------
Runs the SAME 5 benchmark conversations through both backends (local Ollama
llama3.1, and Groq's openai/gpt-oss-20b) for real, and records what each
one actually did -- scored/abstained/parse_error counts, and specific
facet-level results for docs/BACKEND_COMPARISON.md. No numbers in that doc
should exist without this script (or an equivalent real run) having
produced them first.

Forces the backend the same way main.py's --test-groq does for the Groq
runs (temporarily disables Ollama detection), and lets normal detection
pick Ollama for the Ollama runs (since Ollama is the preferred backend
when reachable). Uses top_k=25 for both, consistent with
eval/retrieval_ablation.py, so both backends see the same candidate pool
size per conversation -- retrieval isn't the variable being compared here.

Run with:
    python eval/backend_comparison.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import BENCHMARK_CASES
from src.pipeline import run_pipeline
from src import scorer

TOP_K = 25
RESULTS_PATH = PROJECT_ROOT / "outputs" / "backend_comparison_results.json"

# 5 of the 10 benchmark conversations, chosen to cover a clear/direct case,
# the contradictory case (has the "Patience: Resistance to anger" facet),
# the sarcastic case, the medical trap, and a high-emotion case.
SELECTED_CASE_IDS = [1, 3, 4, 7, 9]


def _run_ollama(conversation: str) -> dict:
    """Runs a conversation letting normal backend detection pick Ollama
    (the preferred backend when reachable -- no forcing needed)."""
    scorer._active_backend = None
    return run_pipeline(conversation, top_k=TOP_K, save_output=False)


def _run_groq(conversation: str) -> dict:
    """Runs a conversation with Ollama detection forced off, same
    mechanism as main.py's cmd_test_groq -- so this genuinely uses Groq
    regardless of whether Ollama is also up."""
    original_check = scorer._check_ollama_available
    original_backend = scorer._active_backend
    scorer._check_ollama_available = lambda: False
    scorer._active_backend = None
    try:
        return run_pipeline(conversation, top_k=TOP_K, save_output=False)
    finally:
        scorer._check_ollama_available = original_check
        scorer._active_backend = original_backend


def _summarize(output: dict) -> dict:
    results = output.get("results", [])
    return {
        "total_retrieved": output.get("total_facets_retrieved", 0),
        "scored": sum(1 for r in results if r["status"] == "scored"),
        "abstained": sum(1 for r in results if r["status"] in ("insufficient_evidence", "not_observable")),
        "parse_errors": sum(1 for r in results if r["status"] == "parse_error"),
        "results_by_facet": {r["facet"]: r for r in results},
    }


def run_comparison() -> list:
    cases = [c for c in BENCHMARK_CASES if c["id"] in SELECTED_CASE_IDS]
    all_results = []

    for case in cases:
        print(f"\n=== Case {case['id']} ({case['type']}) ===")

        print("  Running Ollama...")
        if not scorer._check_ollama_available():
            print("  WARNING: Ollama not reachable right now -- this run will fail/fallback unexpectedly.")
        ollama_output = _run_ollama(case["conversation"])
        ollama_summary = _summarize(ollama_output)
        print(f"    scored={ollama_summary['scored']} abstained={ollama_summary['abstained']} parse_errors={ollama_summary['parse_errors']}")

        print("  Running Groq...")
        groq_output = _run_groq(case["conversation"])
        groq_summary = _summarize(groq_output)
        print(f"    scored={groq_summary['scored']} abstained={groq_summary['abstained']} parse_errors={groq_summary['parse_errors']}")

        all_results.append({
            "id": case["id"],
            "type": case["type"],
            "conversation": case["conversation"],
            "reference_facets": [r["facet"] for r in case["reference"]],
            "ollama": ollama_summary,
            "groq": groq_summary,
        })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved raw comparison results to {RESULTS_PATH}")
    return all_results


if __name__ == "__main__":
    run_comparison()
