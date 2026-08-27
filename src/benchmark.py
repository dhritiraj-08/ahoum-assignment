"""
benchmark.py
------------
Ten hand-written sample conversations covering the tricky cases this system
is specifically designed to handle well (ambiguity, sarcasm, contradiction,
medical/spiritual "traps" that a naive system would happily hallucinate
scores for, etc). Each conversation has a small hand-labeled reference set
(>=3 facets) with an expected outcome, which we compare against the actual
pipeline output.

All reference facet names below were verified to exist verbatim in
data/Facets_Assignment.csv (see docs/PROMPT_LOG.md).

NOTE ON HOW ABSTENTION IS TESTED: for the medical/spiritual "trap"
conversations, the expected facets (e.g. "FSH level", "Basophil count",
"Types of Mindfulness Techniques Used") are classified as NOT
conversation_observable by audit.py, which means embeddings.py never even
indexes them -- they are structurally unretrievable. That is itself the
correct behavior we're testing: the system abstains by construction, not by
the LLM "deciding" to. If one of these ever *does* show up scored in the
results, that is a serious failure worth flagging loudly.
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

try:
    from src.pipeline import run_pipeline
except ImportError:
    from pipeline import run_pipeline

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "outputs" / "benchmark_report.json"

# ---------------------------------------------------------------------------
# 10 benchmark conversations. "expected_status" is one of:
#   "scored"               -> we expect the system to produce a score
#   "insufficient_evidence" -> we expect the system to abstain (LLM-level)
#   "not_observable"        -> we expect the system to never even retrieve it
#     (medical/spiritual/demographic/malformed facets, filtered at embedding time)
# For "scored" expectations, "expected_score" gives our hand-labeled 1-5 value.
# ---------------------------------------------------------------------------
BENCHMARK_CASES = [
    {
        "id": 1,
        "type": "clear_direct",
        "conversation": (
            "I quit my stable corporate job last week to backpack solo through "
            "South America with no itinerary and barely any savings left. I've "
            "always been the one jumping off cliffs, literally and figuratively, "
            "before checking the depth of the water."
        ),
        "reference": [
            {"facet": "Risktaking", "expected_status": "scored", "expected_score": 5},
            {"facet": "Adventure-Seeking Behavior", "expected_status": "scored", "expected_score": 5},
            {"facet": "Common-sense", "expected_status": "scored", "expected_score": 2},
        ],
    },
    {
        "id": 2,
        "type": "ambiguous",
        "conversation": (
            "I told my sister I'd help her move this weekend, but then again, I "
            "also said I'd finally clean out my garage, and honestly I might just "
            "end up doing neither and binge a show instead. We'll see."
        ),
        "reference": [
            {"facet": "Doggedness", "expected_status": "insufficient_evidence"},
            {"facet": "Self-Efficacy", "expected_status": "insufficient_evidence"},
            {"facet": "Common-sense", "expected_status": "insufficient_evidence"},
        ],
    },
    {
        "id": 3,
        "type": "contradictory",
        "conversation": (
            "I tell everyone I'm a very patient person who never loses his temper. "
            "But this morning I screamed at the barista for getting my order wrong, "
            "threw the cup in the trash, and stormed out still fuming twenty "
            "minutes later."
        ),
        "reference": [
            {"facet": "Patience: Resistance to anger", "expected_status": "scored", "expected_score": 1},
            {"facet": "Emotionalism", "expected_status": "scored", "expected_score": 4},
            {"facet": "Decency", "expected_status": "scored", "expected_score": 2},
        ],
    },
    {
        "id": 4,
        "type": "sarcastic",
        "conversation": (
            "Oh sure, I LOVE waking up at 5am to go for a run in the freezing cold, "
            "it's basically my favorite thing in the world. Said no one ever. I hit "
            "snooze four times and rolled back to sleep, as usual."
        ),
        "reference": [
            {"facet": "Doggedness", "expected_status": "scored", "expected_score": 2},
            {"facet": "Self-improvement", "expected_status": "scored", "expected_score": 2},
            {"facet": "Discontentment", "expected_status": "insufficient_evidence"},
        ],
    },
    {
        "id": 5,
        "type": "low_evidence",
        "conversation": "Yeah, work was fine today. Nothing much happened.",
        "reference": [
            {"facet": "Compassion", "expected_status": "insufficient_evidence"},
            {"facet": "Risktaking", "expected_status": "insufficient_evidence"},
            {"facet": "Emotionalism", "expected_status": "insufficient_evidence"},
        ],
    },
    {
        "id": 6,
        "type": "code_switched",
        "conversation": (
            "Yaar, I told my roommate ki main uska kaam bhi kar dunga along with "
            "mine, even though I already had a packed week. I just can't say no "
            "when a dost asks for help, even if it stresses me out."
        ),
        "reference": [
            {"facet": "Compassion", "expected_status": "scored", "expected_score": 4},
            {"facet": "Unassertiveness", "expected_status": "scored", "expected_score": 4},
            {"facet": "Assertiveness and control in relationships", "expected_status": "scored", "expected_score": 2},
        ],
    },
    {
        "id": 7,
        "type": "medical_trap",
        "conversation": (
            "I've been getting these headaches lately, and my doctor mentioned my "
            "FSH levels looked a bit off in my last blood test. My basophil count "
            "was also flagged as high. I've also been more irritable than usual "
            "because of it."
        ),
        "reference": [
            {"facet": "FSH level", "expected_status": "not_observable"},
            {"facet": "Basophil count", "expected_status": "not_observable"},
            {"facet": "Emotionalism", "expected_status": "scored", "expected_score": 3},
        ],
    },
    {
        "id": 8,
        "type": "spiritual_trap",
        "conversation": (
            "I've started doing a short mindfulness session every morning before "
            "work, just five minutes of breathing. Sometimes I also check my "
            "I Ching hexagram reading for fun when I'm feeling stuck on a "
            "decision, though I don't take it too seriously."
        ),
        "reference": [
            {"facet": "Types of Mindfulness Techniques Used", "expected_status": "not_observable"},
            {"facet": "Self-improvement", "expected_status": "scored", "expected_score": 3},
            {"facet": "Peacefulness", "expected_status": "scored", "expected_score": 3},
        ],
    },
    {
        "id": 9,
        "type": "high_emotional",
        "conversation": (
            "I broke down crying in the middle of the grocery store today. "
            "Everything just hit me at once -- my dad's diagnosis, the breakup, "
            "the pressure at work. I couldn't stop shaking and had to call my "
            "best friend just to breathe through it."
        ),
        "reference": [
            {"facet": "Emotionalism", "expected_status": "scored", "expected_score": 5},
            {"facet": "Discontentment", "expected_status": "scored", "expected_score": 4},
            {"facet": "Peacefulness", "expected_status": "scored", "expected_score": 1},
        ],
    },
    {
        "id": 10,
        "type": "professional_formal",
        "conversation": (
            "Per our discussion yesterday, I have finalized the quarterly report "
            "and cross-verified the statistical projections against last year's "
            "figures. Please let me know if the committee requires further "
            "clarification before Thursday's review."
        ),
        "reference": [
            {"facet": "Statistical Reasoning", "expected_status": "scored", "expected_score": 4},
            {"facet": "Cordiality", "expected_status": "scored", "expected_score": 3},
            {"facet": "Common-sense", "expected_status": "insufficient_evidence"},
        ],
    },
]


def _evaluate_case(case: dict, pipeline_output: dict) -> list[dict]:
    """
    Compare one conversation's reference labels against actual pipeline
    output. Returns a list of per-facet evaluation dicts with an
    "outcome" field describing agreement / failure mode.
    """
    results_by_name = {r["facet"]: r for r in pipeline_output.get("results", [])}
    evaluations = []

    for ref in case["reference"]:
        facet_name = ref["facet"]
        actual = results_by_name.get(facet_name)
        expected_status = ref["expected_status"]

        if expected_status == "not_observable":
            # Correct iff it was never scored (either absent from retrieval,
            # or present with a non-"scored" status).
            if actual is None or actual.get("status") != "scored":
                outcome = "correct_abstention"
            else:
                outcome = "SAFETY_VIOLATION_scored_non_observable_facet"

        elif expected_status == "insufficient_evidence":
            if actual is None:
                outcome = "correct_abstention_not_retrieved"
            elif actual.get("status") in ("insufficient_evidence", "not_observable", "parse_error"):
                outcome = "correct_abstention"
            else:
                outcome = "incorrect_overconfident_score"

        else:  # expected_status == "scored"
            if actual is None:
                outcome = "retrieval_miss"
            elif actual.get("status") != "scored":
                outcome = "incorrect_abstention"
            else:
                expected_score = ref.get("expected_score")
                actual_score = actual.get("score")
                if actual_score == expected_score:
                    outcome = "exact_agreement"
                elif actual_score is not None and abs(actual_score - expected_score) <= 1:
                    outcome = "close_agreement"
                else:
                    outcome = "disagreement"

        evaluations.append({
            "facet": facet_name,
            "expected_status": expected_status,
            "expected_score": ref.get("expected_score"),
            "actual_status": actual.get("status") if actual else "not_retrieved",
            "actual_score": actual.get("score") if actual else None,
            "outcome": outcome,
        })

    return evaluations


def run_benchmark() -> dict:
    """Runs the pipeline on all 10 benchmark conversations, evaluates each
    against its reference labels, prints a report, and saves it to JSON."""
    all_case_results = []
    outcome_counts: dict[str, int] = {}

    for case in BENCHMARK_CASES:
        console.print(f"\n[bold cyan]Running case {case['id']} ({case['type']})...[/bold cyan]")
        try:
            pipeline_output = run_pipeline(case["conversation"], save_output=False)
        except Exception as e:
            console.print(f"[bold red]Pipeline crashed on case {case['id']}: {e}[/bold red]")
            pipeline_output = {"results": [], "error": str(e)}

        evaluations = _evaluate_case(case, pipeline_output)
        for ev in evaluations:
            outcome_counts[ev["outcome"]] = outcome_counts.get(ev["outcome"], 0) + 1

        all_case_results.append({
            "id": case["id"],
            "type": case["type"],
            "conversation": case["conversation"],
            "evaluations": evaluations,
            "total_retrieved": pipeline_output.get("total_facets_retrieved", 0),
        })

    report = {
        "cases": all_case_results,
        "outcome_counts": outcome_counts,
    }

    _print_report(report)

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\n[bold green]Saved benchmark report to:[/bold green] {REPORT_PATH}")
    except Exception as e:
        console.print(f"[bold red]Could not save benchmark report: {e}[/bold red]")

    return report


def _print_report(report: dict):
    console.print("\n[bold cyan]===== BENCHMARK REPORT =====[/bold cyan]\n")

    table = Table(title="Outcome counts across all 10 conversations (30 reference facets)")
    table.add_column("Outcome", style="bold")
    table.add_column("Count", justify="right")

    # Order roughly best -> worst so the table reads as a scorecard.
    ordering = [
        "exact_agreement", "close_agreement", "correct_abstention",
        "correct_abstention_not_retrieved", "disagreement", "incorrect_abstention",
        "incorrect_overconfident_score", "retrieval_miss",
        "SAFETY_VIOLATION_scored_non_observable_facet",
    ]
    counts = report["outcome_counts"]
    for key in ordering:
        if key in counts:
            table.add_row(key, str(counts[key]))
    for key in counts:
        if key not in ordering:
            table.add_row(key, str(counts[key]))

    console.print(table)

    total = sum(counts.values())
    good = counts.get("exact_agreement", 0) + counts.get("close_agreement", 0) + \
        counts.get("correct_abstention", 0) + counts.get("correct_abstention_not_retrieved", 0)
    console.print(f"\nOverall: [green]{good}/{total}[/green] reference facets handled correctly.")

    safety_violations = counts.get("SAFETY_VIOLATION_scored_non_observable_facet", 0)
    if safety_violations:
        console.print(
            f"[bold red]WARNING: {safety_violations} safety violation(s) -- a "
            f"non-observable (medical/spiritual/malformed) facet was scored![/bold red]"
        )
    else:
        console.print("[green]No safety violations: no medical/spiritual/malformed facet was ever scored.[/green]")

    # Top failure modes
    failure_keys = [
        "disagreement", "incorrect_abstention", "incorrect_overconfident_score",
        "retrieval_miss", "SAFETY_VIOLATION_scored_non_observable_facet",
    ]
    failures = sorted(
        ((k, counts.get(k, 0)) for k in failure_keys if counts.get(k, 0) > 0),
        key=lambda kv: -kv[1],
    )
    if failures:
        console.print("\n[bold yellow]Top failure modes:[/bold yellow]")
        for name, count in failures:
            console.print(f"  {name}: {count}")
    else:
        console.print("\n[green]No failure modes observed in this run.[/green]")


if __name__ == "__main__":
    run_benchmark()
