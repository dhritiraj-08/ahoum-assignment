"""
eval/live_ui_testing.py
------------------------
Runs the 5 adversarial conversations for docs/DEBUGGING.md's "Live UI
Testing Observations" section through the SAME pipeline app.py's Streamlit
UI calls (src.pipeline.run_pipeline), on both backends, for real. app.py
adds no scoring logic of its own -- it's a thin wrapper over run_pipeline
-- so this produces results identical to what clicking through the actual
Streamlit widget would, without the time cost/flakiness of browser
automation for 10 separate runs. Uses the real production default
(top_k=40, src.pipeline.TOP_K_DEFAULT), not the smaller top_k=25 used in
eval/backend_comparison.py's controlled ablation, since this is meant to
reflect actual --score / app.py usage.

Run with (GROQ_API_KEY must be set in the same shell, e.g. via
`set -a && source .env && set +a` first):
    python eval/live_ui_testing.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline
from src import scorer

RESULTS_PATH = PROJECT_ROOT / "outputs" / "live_ui_testing_results.json"

CONVERSATIONS = [
    {
        "id": 1,
        "label": "Medical trap",
        "text": "I've been feeling exhausted, my doctor said my TSH levels are abnormal",
    },
    {
        "id": 2,
        "label": "Sarcasm",
        "text": "Oh yes I'm VERY patient, I only yelled at three people today",
    },
    {
        "id": 3,
        "label": "Contradiction",
        "text": "I'm very organized. My desk has 47 unread emails and I haven't filed taxes in 2 years",
    },
    {
        "id": 4,
        "label": "Code-switch",
        "text": "I work hard yaar, but kabhi kabhi I just want to chill",
    },
    {
        "id": 5,
        "label": "Vague",
        "text": "Things are okay I guess",
    },
]


def _run_ollama(text: str) -> dict:
    scorer._active_backend = None
    return run_pipeline(text, save_output=False)


def _run_groq(text: str) -> dict:
    original_check = scorer._check_ollama_available
    original_backend = scorer._active_backend
    scorer._check_ollama_available = lambda: False
    scorer._active_backend = None
    try:
        return run_pipeline(text, save_output=False)
    finally:
        scorer._check_ollama_available = original_check
        scorer._active_backend = original_backend


def _summarize(output: dict) -> dict:
    results = output.get("results", [])
    scored = [r for r in results if r["status"] == "scored"]
    return {
        "total_retrieved": output.get("total_facets_retrieved", 0),
        "scored": len(scored),
        "abstained": sum(1 for r in results if r["status"] in ("insufficient_evidence", "not_observable")),
        "parse_errors": sum(1 for r in results if r["status"] == "parse_error"),
        "scored_facets": [{"facet": r["facet"], "score": r["score"]} for r in scored],
        "all_results": results,
    }


def run_all():
    all_results = []
    if not scorer._check_ollama_available():
        print("WARNING: Ollama not reachable right now -- results will not reflect real Ollama behavior.")

    for conv in CONVERSATIONS:
        print(f"\n=== {conv['id']}. {conv['label']} ===")
        print(f"  \"{conv['text']}\"")

        print("  Running Ollama...")
        ollama_out = _run_ollama(conv["text"])
        ollama_summary = _summarize(ollama_out)
        print(f"    scored={ollama_summary['scored']} abstained={ollama_summary['abstained']} parse_errors={ollama_summary['parse_errors']}")
        for sf in ollama_summary["scored_facets"]:
            print(f"      {sf['facet']}: {sf['score']}/5")

        print("  Running Groq...")
        groq_out = _run_groq(conv["text"])
        groq_summary = _summarize(groq_out)
        print(f"    scored={groq_summary['scored']} abstained={groq_summary['abstained']} parse_errors={groq_summary['parse_errors']}")
        for sf in groq_summary["scored_facets"]:
            print(f"      {sf['facet']}: {sf['score']}/5")

        all_results.append({
            "id": conv["id"],
            "label": conv["label"],
            "conversation": conv["text"],
            "ollama": ollama_summary,
            "groq": groq_summary,
        })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved raw results to {RESULTS_PATH}")
    return all_results


if __name__ == "__main__":
    run_all()
