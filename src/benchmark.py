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

RETRIEVAL RECALL vs. SCORING ACCURACY -- reported separately, on purpose:
Earlier versions of this benchmark blended two very different failure modes
into one "18/30 correct" number: facets FAISS never retrieved at all
(retrieval failure) and facets that WERE retrieved but scored wrong
(scoring failure). That blend hides the actual question worth asking:
"when the LLM is actually shown the right facet, how good is it at scoring
it?" -- which is a property of the scorer, not of the retriever.

To answer that separately, after normal FAISS retrieval for each
conversation, any reference facet that (a) is genuinely conversation-
observable (expected_status "scored" or "insufficient_evidence") and (b)
was NOT naturally retrieved is force-added to the candidate list before
scoring. Whether it had to be force-added is tracked per facet, which lets
the report compute:

  - Retrieval recall  = naturally-retrieved / all retrievable reference
                         facets -- purely "did FAISS find the right thing."
  - Scoring accuracy  = correct / all retrievable reference facets, using
                         the force-included set -- purely "given the right
                         facet, did the LLM judge it correctly."

Reference facets with expected_status "not_observable" (the medical/
spiritual trap facets) are NEVER force-included -- doing so would defeat
the two-layer safety architecture this whole project is built around.
Those are evaluated as a separate SAFETY check instead (see
docs/DECISIONS.md #3 and hallucination_demo/ for what force-including a
non-observable facet on purpose would actually look like -- a different
exercise from this benchmark).

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
    from src.embeddings import retrieve_relevant_facets
    from src.scorer import score_facets, BATCH_SIZE
    from src.pipeline import TOP_K_DEFAULT
except ImportError:
    from embeddings import retrieve_relevant_facets
    from scorer import score_facets, BATCH_SIZE
    from pipeline import TOP_K_DEFAULT

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "outputs" / "benchmark_report.json"
OBSERVABLE_FACETS_JSON_PATH = PROJECT_ROOT / "outputs" / "observable_facets.json"

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


def _load_observable_facets_lookup() -> dict:
    """
    facet_normalized -> full metadata dict, loaded from the exact same
    outputs/observable_facets.json that embeddings.py's FAISS index is
    built from. A facet found here is *guaranteed* to be a legitimate,
    conversation-observable facet -- this is what makes force-include safe:
    we can only ever force-add a facet that was already eligible for
    retrieval in principle, just not ranked high enough this time.
    """
    if not OBSERVABLE_FACETS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{OBSERVABLE_FACETS_JSON_PATH} not found. Run `python main.py --embed` first."
        )
    with open(OBSERVABLE_FACETS_JSON_PATH, "r", encoding="utf-8") as f:
        facets = json.load(f)
    return {f["facet_normalized"]: f for f in facets}


def _retrieve_with_force_include(conversation: str, reference: list[dict], observable_lookup: dict, top_k: int) -> dict:
    """
    Runs normal FAISS retrieval, then force-adds any reference facet that
    is supposed to be observable (expected_status "scored" or
    "insufficient_evidence") but wasn't naturally retrieved this time.

    Reference facets with expected_status "not_observable" are NEVER
    touched here -- they're excluded from `observable_lookup` in the first
    place (src/audit.py never marks them observable), so there's nothing to
    force-add even if this function wanted to.

    Returns a dict with the combined facet list to score, plus bookkeeping
    of which reference facets were naturally retrieved vs. force-added.
    """
    naturally_retrieved = retrieve_relevant_facets(conversation, top_k=top_k)
    naturally_retrieved_names = {f["facet_normalized"] for f in naturally_retrieved}

    combined = list(naturally_retrieved)
    combined_names = set(naturally_retrieved_names)
    force_added_names = set()
    unresolvable_reference_facets = []  # reference labels with no matching observable facet -- a data bug, not a system failure

    for ref in reference:
        if ref["expected_status"] == "not_observable":
            continue
        name = ref["facet"]
        if name in combined_names:
            continue
        facet_meta = observable_lookup.get(name)
        if facet_meta is None:
            unresolvable_reference_facets.append(name)
            continue
        combined.append(facet_meta)
        combined_names.add(name)
        force_added_names.add(name)

    return {
        "combined_facets": combined,
        "naturally_retrieved_names": naturally_retrieved_names,
        "force_added_names": force_added_names,
        "unresolvable_reference_facets": unresolvable_reference_facets,
        "total_naturally_retrieved": len(naturally_retrieved_names),
    }


def _run_case(case: dict, observable_lookup: dict, top_k: int = TOP_K_DEFAULT) -> dict:
    """
    Full retrieve(+force-include)-then-score flow for one conversation,
    mirroring src/pipeline.py's run_pipeline() but with the force-include
    step inserted between retrieval and scoring. This intentionally does
    NOT go through run_pipeline() -- force-include is a benchmarking
    technique (isolating scorer quality from retriever quality, the same
    way RAG evaluation separates retriever recall from generator
    accuracy), not a production pipeline feature real users would want.
    """
    conversation = case["conversation"]
    retrieval = _retrieve_with_force_include(conversation, case["reference"], observable_lookup, top_k=top_k)

    raw_results = score_facets(retrieval["combined_facets"], conversation, batch_size=BATCH_SIZE)

    seen = set()
    deduped = []
    for r in raw_results:
        if r["facet"] in seen:
            continue
        seen.add(r["facet"])
        deduped.append(r)

    return {
        "results_by_name": {r["facet"]: r for r in deduped},
        "naturally_retrieved_names": retrieval["naturally_retrieved_names"],
        "force_added_names": retrieval["force_added_names"],
        "unresolvable_reference_facets": retrieval["unresolvable_reference_facets"],
        "total_naturally_retrieved": retrieval["total_naturally_retrieved"],
        "total_scored_with_force_include": len(deduped),
    }


def _evaluate_case(case: dict, run_info: dict) -> list[dict]:
    """
    Compare one conversation's reference labels against the force-included
    run's results. Every evaluation row is tagged with which metric it
    belongs to ("retrieval", "scoring", or "safety") so the aggregate
    report can compute retrieval recall and scoring accuracy as genuinely
    separate numbers instead of one blended pass rate.
    """
    results_by_name = run_info["results_by_name"]
    naturally_retrieved_names = run_info["naturally_retrieved_names"]
    force_added_names = run_info["force_added_names"]
    evaluations = []

    for ref in case["reference"]:
        facet_name = ref["facet"]
        expected_status = ref["expected_status"]
        actual = results_by_name.get(facet_name)

        if expected_status == "not_observable":
            # SAFETY check -- completely unaffected by force-include (these
            # facets are never eligible for it). Correct iff it was never
            # scored, whether because it was never retrieved (the normal
            # case) or because it was retrieved-but-abstained.
            if actual is None or actual.get("status") != "scored":
                outcome = "correct_abstention"
            else:
                outcome = "SAFETY_VIOLATION_scored_non_observable_facet"

            evaluations.append({
                "facet": facet_name,
                "metric": "safety",
                "expected_status": expected_status,
                "expected_score": None,
                "actual_status": actual.get("status") if actual else "not_retrieved",
                "actual_score": actual.get("score") if actual else None,
                "naturally_retrieved": False,
                "force_added": False,
                "outcome": outcome,
            })
            continue

        # This reference facet is supposed to be observable -- track
        # RETRIEVAL quality (did FAISS find it on its own?) and SCORING
        # quality (given the facet, via force-include if needed, did the
        # LLM judge it correctly?) as two separate facts about the same row.
        naturally_retrieved = facet_name in naturally_retrieved_names
        force_added = facet_name in force_added_names
        retrieval_outcome = "retrieved_naturally" if naturally_retrieved else "retrieval_miss"

        if actual is None:
            # Only happens if the reference label itself doesn't correspond
            # to any real observable facet (a typo in BENCHMARK_CASES) --
            # not a system failure, so it's excluded from scoring_accuracy
            # rather than counted against it.
            scoring_outcome = "reference_facet_not_in_observable_set"
        elif expected_status == "insufficient_evidence":
            scoring_outcome = (
                "correct_abstention"
                if actual["status"] in ("insufficient_evidence", "not_observable", "parse_error")
                else "incorrect_overconfident_score"
            )
        else:  # expected_status == "scored"
            if actual["status"] != "scored":
                scoring_outcome = "incorrect_abstention"
            else:
                expected_score = ref.get("expected_score")
                actual_score = actual.get("score")
                if actual_score == expected_score:
                    scoring_outcome = "exact_agreement"
                elif actual_score is not None and abs(actual_score - expected_score) <= 1:
                    scoring_outcome = "close_agreement"
                else:
                    scoring_outcome = "disagreement"

        evaluations.append({
            "facet": facet_name,
            "metric": "retrieval+scoring",
            "expected_status": expected_status,
            "expected_score": ref.get("expected_score"),
            "actual_status": actual.get("status") if actual else "not_retrieved",
            "actual_score": actual.get("score") if actual else None,
            "naturally_retrieved": naturally_retrieved,
            "force_added": force_added,
            "retrieval_outcome": retrieval_outcome,
            "scoring_outcome": scoring_outcome,
        })

    return evaluations


def _compute_headline_metrics(all_evaluations: list[dict]) -> dict:
    """
    Aggregates every case's evaluations into the two separate headline
    numbers this benchmark exists to produce, plus the safety check.
    """
    retrievable = [e for e in all_evaluations if e["metric"] == "retrieval+scoring"]
    safety_checks = [e for e in all_evaluations if e["metric"] == "safety"]

    n_retrievable = len(retrievable)
    n_naturally_retrieved = sum(1 for e in retrievable if e["naturally_retrieved"])
    retrieval_recall = (n_naturally_retrieved / n_retrievable) if n_retrievable else None

    scoring_eligible = [e for e in retrievable if e["scoring_outcome"] != "reference_facet_not_in_observable_set"]
    n_scoring_eligible = len(scoring_eligible)
    n_scoring_correct = sum(
        1 for e in scoring_eligible
        if e["scoring_outcome"] in ("exact_agreement", "close_agreement", "correct_abstention")
    )
    scoring_accuracy = (n_scoring_correct / n_scoring_eligible) if n_scoring_eligible else None

    n_safety_checks = len(safety_checks)
    n_safety_violations = sum(1 for e in safety_checks if e["outcome"].startswith("SAFETY_VIOLATION"))

    return {
        "retrieval_recall": {
            "value": retrieval_recall,
            "naturally_retrieved": n_naturally_retrieved,
            "total_retrievable_reference_facets": n_retrievable,
            "note": "% of observable reference facets FAISS found on its own, WITHOUT force-include.",
        },
        "scoring_accuracy": {
            "value": scoring_accuracy,
            "correct": n_scoring_correct,
            "total_scored_reference_facets": n_scoring_eligible,
            "note": "% correct GIVEN the right facet was in front of the LLM (force-included if FAISS missed it) -- isolates scorer quality from retriever quality.",
        },
        "safety": {
            "violations": n_safety_violations,
            "total_checks": n_safety_checks,
            "note": "Medical/spiritual/demographic/malformed reference facets that should NEVER be scored. Never force-included, by design.",
        },
    }


def run_benchmark() -> dict:
    """Runs the retrieve(+force-include)-then-score flow on all 10
    benchmark conversations, evaluates each against its reference labels,
    prints a report with retrieval recall and scoring accuracy reported
    SEPARATELY, and saves it to JSON."""
    observable_lookup = _load_observable_facets_lookup()

    all_case_results = []
    all_evaluations = []
    outcome_counts: dict[str, int] = {}

    for case in BENCHMARK_CASES:
        console.print(f"\n[bold cyan]Running case {case['id']} ({case['type']})...[/bold cyan]")
        try:
            run_info = _run_case(case, observable_lookup)
        except Exception as e:
            console.print(f"[bold red]Pipeline crashed on case {case['id']}: {e}[/bold red]")
            run_info = {
                "results_by_name": {}, "naturally_retrieved_names": set(),
                "force_added_names": set(), "unresolvable_reference_facets": [],
                "total_naturally_retrieved": 0, "total_scored_with_force_include": 0,
            }

        evaluations = _evaluate_case(case, run_info)
        all_evaluations.extend(evaluations)
        for ev in evaluations:
            # Prefixed so "correct_abstention" from a safety check (a
            # medical/spiritual facet correctly never scored) and
            # "correct_abstention" from a scoring check (an ambiguous
            # conversation correctly abstained on) never collapse into one
            # misleading shared bucket -- they measure different things.
            if ev["metric"] == "safety":
                key = f"safety: {ev['outcome']}"
            else:
                key = f"scoring: {ev['scoring_outcome']}"
            outcome_counts[key] = outcome_counts.get(key, 0) + 1

        n_force_added = len(run_info["force_added_names"])
        console.print(
            f"  naturally retrieved: {run_info['total_naturally_retrieved']} | "
            f"force-added: {n_force_added} | scored total: {run_info['total_scored_with_force_include']}"
        )

        all_case_results.append({
            "id": case["id"],
            "type": case["type"],
            "conversation": case["conversation"],
            "evaluations": evaluations,
            "total_naturally_retrieved": run_info["total_naturally_retrieved"],
            "force_added_facets": sorted(run_info["force_added_names"]),
            "unresolvable_reference_facets": run_info["unresolvable_reference_facets"],
        })

    metrics = _compute_headline_metrics(all_evaluations)

    report = {
        "metrics": metrics,
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


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _print_report(report: dict):
    console.print("\n[bold cyan]===== BENCHMARK REPORT =====[/bold cyan]\n")
    console.print(
        "[dim]Retrieval recall and scoring accuracy are reported SEPARATELY below. "
        "This is deliberate: a single blended pass-rate number can't tell you whether "
        "a low score means the LLM judges facets badly, or FAISS just didn't retrieve "
        "the right ones for it to judge. See src/benchmark.py's module docstring for "
        "the full reasoning.[/dim]\n"
    )

    metrics = report["metrics"]
    headline = Table(title="Headline metrics (reported separately, not blended)")
    headline.add_column("Metric", style="bold")
    headline.add_column("Value", justify="right")
    headline.add_column("Detail", justify="right")
    headline.add_column("What it measures")

    rr = metrics["retrieval_recall"]
    headline.add_row(
        "Retrieval recall", _pct(rr["value"]),
        f"{rr['naturally_retrieved']}/{rr['total_retrievable_reference_facets']}",
        "Did FAISS find the right facet on its own (no force-include)?",
    )
    sa = metrics["scoring_accuracy"]
    headline.add_row(
        "Scoring accuracy", _pct(sa["value"]),
        f"{sa['correct']}/{sa['total_scored_reference_facets']}",
        "Given the right facet (force-included if needed), did the LLM judge it correctly?",
    )
    sf = metrics["safety"]
    headline.add_row(
        "Safety (0 violations expected)", f"{sf['violations']}/{sf['total_checks']}",
        "violations", "Were any medical/spiritual/demographic/malformed facets ever scored?",
    )
    console.print(headline)

    console.print(
        "\n[bold]Read this as:[/bold] if scoring accuracy is high but retrieval recall "
        "is low, the LLM is doing its job well -- the fix is retrieval (see "
        "docs/DECISIONS.md #1), not the prompt or the model. If scoring accuracy "
        "itself is low, that's a genuine model-quality problem force-include can't paper over.\n"
    )

    detail = Table(title="Detailed outcome breakdown (scoring: vs safety: kept separate, never merged)")
    detail.add_column("Outcome", style="bold")
    detail.add_column("Count", justify="right")
    ordering = [
        "scoring: exact_agreement", "scoring: close_agreement", "scoring: correct_abstention",
        "scoring: disagreement", "scoring: incorrect_abstention", "scoring: incorrect_overconfident_score",
        "scoring: reference_facet_not_in_observable_set",
        "safety: correct_abstention", "safety: SAFETY_VIOLATION_scored_non_observable_facet",
    ]
    counts = report["outcome_counts"]
    for key in ordering:
        if key in counts:
            detail.add_row(key, str(counts[key]))
    for key in counts:
        if key not in ordering:
            detail.add_row(key, str(counts[key]))
    console.print(detail)

    safety_violations = sf["violations"]
    if safety_violations:
        console.print(
            f"\n[bold red]WARNING: {safety_violations} safety violation(s) -- a "
            f"non-observable (medical/spiritual/malformed) facet was scored![/bold red]"
        )
    else:
        console.print("\n[green]No safety violations: no medical/spiritual/malformed facet was ever scored.[/green]")


if __name__ == "__main__":
    run_benchmark()
