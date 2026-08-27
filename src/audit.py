"""
audit.py
--------
Cleans and enriches the raw 399-facet CSV so the rest of the pipeline never
has to reason about messy strings, category-header artifacts, or facets that
are impossible to judge from a short conversation.

WHY THIS EXISTS (design rationale):
The raw CSV mixes genuine personality/behavioural traits (e.g. "Risktaking")
with things that look like traits but are not conversation-observable at all
(e.g. "FSH level", "Basophil count", "I Ching hexagram") and with plain data
artifacts (trailing colons left over from a spreadsheet header row, numbered
list prefixes like "899. "). If we naively embedded and scored all 399 rows,
the LLM would happily hallucinate a "score" for someone's hormone level from
two sentences of chit-chat. This script is the safety gate that prevents
that: it labels every facet with a category, whether it is even observable
from conversation, and (if not) *why*, so downstream code can hard-block
scoring on unsafe/impossible facets instead of relying on the LLM to
self-police.

Classification is done with deterministic keyword/regex heuristics rather
than an LLM call, because (a) it only needs to run once, (b) it must be
reproducible/auditable for the write-up, and (c) it's fast enough to run in
under a second on 399 rows. The heuristics are intentionally conservative and
documented -- see docs/DEBUGGING.md for known misclassifications to review.
"""

import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "Facets_Assignment.csv"
ENRICHED_CSV_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"

# ---------------------------------------------------------------------------
# Keyword banks used for rule-based classification.
# These are intentionally explicit lists (not ML) so classification is
# reproducible and easy to hand-audit / extend later.
# ---------------------------------------------------------------------------
MEDICAL_KEYWORDS = [
    "fsh", "lh level", "tsh", "hormone", "cortisol", "estrogen", "testosterone",
    "insulin", "basophil", "eosinophil", "neutrophil", "lymphocyte", "monocyte",
    "platelet", "hemoglobin", "haemoglobin", "blood pressure", "blood sugar",
    "glucose", "cholesterol", "bmi", "heart rate", "pulse", "apnea", "apnoea",
    "diagnosis", "diagnosed", "syndrome", "disorder", "disease", "vitamin d",
    "vitamin b12", "enzyme", "antibody", "white blood cell", "red blood cell",
    "wbc", "rbc", "thyroid", "menstrual", "pregnan", "sperm", "libido",
    "sexual dysfunction", "arousal", "creatinine", "triglyceride", "hdl", "ldl",
    "hba1c", "oxygen saturation", "spo2", "bone density", "metabolic rate",
    "respiratory rate", "genetic marker", " dna ", "biomarker", "clinical",
    "lab result", "lab value", "medication dosage", "prescription",
    "sleep apnea", "cell count", "blood cell",
]

SPIRITUAL_KEYWORDS = [
    "i ching", "hexagram", "reiki", "sufi", "dhikr", "chakra", "tarot",
    "astrology", "zodiac", "karma", "kundalini", "feng shui", "numerology",
    "shamanic", "wicca", "kabbalah", "zen", "meditation retreat", "mantra",
    "yoga nidra", "prayer", "church attendance", "shabbat", "pilgrimage",
    "spiritual", "sacred", "ritual", "mindfulness technique", "mysticism",
    "aura", "crystal healing", "tantra", "vedic", "ayurved", "sutra",
    "satsang", "sangha", "dharma", "yajna", "puja", "namaz", "rosary",
]

SOCIAL_DEMO_KEYWORDS = [
    "passport stamps", "commute time", "commute", "household size", "income",
    "social media follower", "screen time", "purchase frequency",
    "travel frequency", "pet-enrichment", "number of friends",
    "subscriptions count", "zip code", "neighborhood", "neighbourhood",
    "census", "marital status", "employment status", "education level",
    "housing type", "salary",
]

COGNITIVE_KEYWORDS = [
    "reasoning", "memory", " iq", "processing speed", "attention span",
    "spelling accuracy", "arithmetic", "spatial awareness", "verbal fluency",
    "statistical reasoning", "problem solving", "problem-solving",
    "comprehension", "cognitive", "working memory", "recall",
    "concentration", "mental arithmetic", "logical thinking",
    "pattern recognition", "vocabulary", "learning speed",
    "executive function", "rapid cognitive processing",
]

BEHAVIORAL_KEYWORDS = [
    "seeking behavior", "seeking behaviour", "tendency", "avoidance",
    "procrastination", "risk-taking behavior", "coping style", "habit",
    "impulsivity", "compulsive", "addictive behavior", "behavior",
    "behaviour", "engagement in", "frequency of",
]

# Sexual-content keywords escalate sensitivity to "high" regardless of category.
SEXUAL_KEYWORDS = [
    "sexual", "libido", "arousal", "orgasm", "menstrual", "pregnan",
    "sperm", "contracept", "intimacy frequency",
]

# Facets whose title signals a categorical (non-ordinal) label rather than a
# 1-5 gradable trait -- these are not truly scorable on our scale even though
# they aren't malformed CSV artifacts per se.
CATEGORICAL_PREFIXES = ("type of ", "types of ")

NON_OBSERVABLE_CATEGORIES = {
    "medical_biological",
    "spiritual_esoteric",
    "social_demographic",
    "header_or_malformed",
}


def normalize_facet_name(raw: str) -> tuple[str, bool]:
    """
    Clean a raw facet string.

    Returns (normalized_name, had_trailing_colon) where had_trailing_colon
    is True only when the *original* string's last non-whitespace character
    was ':' (a strong signal of a leftover spreadsheet header row, e.g.
    "Democratic Leadership:") -- NOT for facets that merely contain a colon
    in the middle (e.g. "Patience: Resistance to anger", which is a valid
    facet with a sub-definition).
    """
    s = str(raw).strip()
    had_trailing_colon = s.endswith(":")

    # Strip leading numeric list markers like "899. " or "793. "
    s = re.sub(r"^\d+\.\s*", "", s).strip()

    # Strip a genuinely trailing colon (but leave mid-string colons alone).
    if s.endswith(":"):
        s = s[:-1].strip()

    # Light capitalization fix: only touch the first character, and only if
    # it's lowercase. We deliberately do NOT title-case the whole string --
    # that would mangle acronyms like "FSH", "I Ching", "IQ".
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s, had_trailing_colon


def classify_facet(name: str, had_trailing_colon: bool) -> str:
    """
    Rule-based category classifier. Order matters: more specific / higher-risk
    categories are checked first so e.g. "Basophil count" (medical) isn't
    accidentally swept into the generic "count -> social_demographic" bucket.
    """
    lower = f" {name.lower()} "  # pad so word-boundary-ish substring checks are safer

    # 1. Malformed / header artifacts
    if had_trailing_colon:
        return "header_or_malformed"
    if not name or len(name) < 2:
        return "header_or_malformed"
    if name.lower().startswith(CATEGORICAL_PREFIXES):
        return "header_or_malformed"

    # 2. Medical / biological (highest-stakes category -> checked early)
    if any(kw in lower for kw in MEDICAL_KEYWORDS):
        return "medical_biological"

    # 3. Spiritual / esoteric
    if any(kw in lower for kw in SPIRITUAL_KEYWORDS):
        return "spiritual_esoteric"

    # 4. Social / demographic (explicit keyword match, or generic countable
    #    life-stat patterns that aren't personality traits)
    if any(kw in lower for kw in SOCIAL_DEMO_KEYWORDS):
        return "social_demographic"
    if re.search(r"\bcount\b", lower) or lower.strip().startswith("number of"):
        return "social_demographic"

    # 5. Cognitive ability
    if any(kw in lower for kw in COGNITIVE_KEYWORDS):
        return "cognitive_ability"

    # 6. Behavioral tendency
    if any(kw in lower for kw in BEHAVIORAL_KEYWORDS):
        return "behavioral_tendency"

    # 7. Default: personality trait
    return "personality_trait"


def get_sensitivity(name: str, category: str) -> str:
    """High = medical or sexual content. Medium = spiritual/financial personal
    disclosure. Low = everything else (default personality/cognitive/behavioral)."""
    lower = name.lower()
    if category == "medical_biological" or any(kw in lower for kw in SEXUAL_KEYWORDS):
        return "high"
    if category == "spiritual_esoteric":
        return "medium"
    if category == "social_demographic" and any(kw in lower for kw in ("income", "salary", "finance")):
        return "medium"
    return "low"


def get_abstention_reason(category: str) -> str:
    """Why we refuse to score this facet from conversation text alone."""
    reasons = {
        "medical_biological": (
            "Requires a clinical measurement, lab test, or diagnosis record; "
            "cannot be responsibly inferred from conversational text alone."
        ),
        "spiritual_esoteric": (
            "Requires self-reported practice history/records (e.g. session counts, "
            "ritual frequency) that a short conversation cannot reliably establish."
        ),
        "social_demographic": (
            "Requires a factual/demographic record (counts, logs, official data) "
            "rather than something inferable from conversational tone or content."
        ),
        "header_or_malformed": (
            "This row is a CSV formatting artifact (category header, categorical "
            "label, or malformed entry), not a scorable 1-5 trait."
        ),
    }
    return reasons.get(category, "")


def generate_scoring_anchor(name: str, category: str) -> str:
    """
    Produce a short, LLM-friendly 1-5 anchor definition for observable facets.
    Templated per category so the scorer prompt has a concrete rubric instead
    of just a bare facet name -- this materially reduces score drift between
    batches, since the model gets the same anchor language every time.
    """
    if category == "cognitive_ability":
        return (
            f"1=No evidence of {name} in the conversation; "
            f"3=Average/typical evidence of {name}; "
            f"5=Strong, clear evidence of high {name} demonstrated in the person's reasoning or language."
        )
    if category == "behavioral_tendency":
        return (
            f"1=No indication of {name} in what the person says or describes; "
            f"3=Occasional or moderate indication of {name}; "
            f"5=Strong, repeated indication of {name} in the described behavior."
        )
    # personality_trait (default)
    return (
        f"1=Very low {name}, or the opposite trait is expressed; "
        f"3=Moderate/average {name}; "
        f"5=Very high {name} clearly expressed in the conversation."
    )


def audit_facets(raw_csv_path: Path = RAW_CSV_PATH, save_path: Path = ENRICHED_CSV_PATH) -> pd.DataFrame:
    """
    Main entry point. Loads the raw CSV, cleans/classifies every row, writes
    the enriched CSV, and prints a summary audit report. Returns the enriched
    DataFrame so callers (e.g. main.py) can reuse it without a re-read.
    """
    try:
        df = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        console.print(f"[bold red]ERROR:[/bold red] Could not find raw facets CSV at {raw_csv_path}")
        raise
    except Exception as e:
        console.print(f"[bold red]ERROR reading CSV:[/bold red] {e}")
        raise

    if "Facets" not in df.columns:
        raise ValueError(f"Expected a 'Facets' column, found columns: {list(df.columns)}")

    raw_names = df["Facets"].astype(str).tolist()

    records = []
    malformed_examples = []
    for raw in raw_names:
        norm_name, had_colon = normalize_facet_name(raw)
        category = classify_facet(norm_name, had_colon)
        observable = category not in NON_OBSERVABLE_CATEGORIES
        sensitivity = get_sensitivity(norm_name, category)
        abstention_reason = "" if observable else get_abstention_reason(category)
        scoring_anchors = generate_scoring_anchor(norm_name, category) if observable else ""

        if category == "header_or_malformed":
            malformed_examples.append((raw, norm_name))

        records.append({
            "facet_raw": raw,
            "facet_normalized": norm_name,
            "category": category,
            "conversation_observable": observable,
            "sensitivity": sensitivity,
            "abstention_reason": abstention_reason,
            "scoring_anchors": scoring_anchors,
        })

    enriched = pd.DataFrame(records)

    # Drop exact duplicate normalized names, keep first occurrence, but don't
    # crash if there are none (there shouldn't be, but real CSVs surprise you).
    before = len(enriched)
    enriched = enriched.drop_duplicates(subset=["facet_normalized"], keep="first").reset_index(drop=True)
    dupes_dropped = before - len(enriched)

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(save_path, index=False)
    except Exception as e:
        console.print(f"[bold red]ERROR writing enriched CSV:[/bold red] {e}")
        raise

    _print_audit_report(enriched, malformed_examples, dupes_dropped, save_path)
    return enriched


def _print_audit_report(enriched: pd.DataFrame, malformed_examples, dupes_dropped: int, save_path: Path):
    """Pretty-print a summary of what the audit found."""
    console.print("\n[bold cyan]===== FACET AUDIT REPORT =====[/bold cyan]\n")

    table = Table(title="Facets per category")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Observable?", justify="center")

    counts = enriched["category"].value_counts()
    for category in [
        "personality_trait", "cognitive_ability", "behavioral_tendency",
        "medical_biological", "spiritual_esoteric", "social_demographic",
        "header_or_malformed",
    ]:
        count = int(counts.get(category, 0))
        observable = "No" if category in NON_OBSERVABLE_CATEGORIES else "Yes"
        table.add_row(category, str(count), observable)

    console.print(table)

    total = len(enriched)
    n_observable = int(enriched["conversation_observable"].sum())
    n_not = total - n_observable
    console.print(f"\nTotal facets (after dedup): [bold]{total}[/bold]")
    console.print(f"  Conversation-observable:     [green]{n_observable}[/green]")
    console.print(f"  NOT observable (abstain):    [yellow]{n_not}[/yellow]")
    if dupes_dropped:
        console.print(f"  Duplicate normalized names dropped: {dupes_dropped}")

    sens_counts = enriched["sensitivity"].value_counts()
    console.print(
        f"\nSensitivity -> high: {int(sens_counts.get('high', 0))}, "
        f"medium: {int(sens_counts.get('medium', 0))}, "
        f"low: {int(sens_counts.get('low', 0))}"
    )

    if malformed_examples:
        console.print(f"\n[bold yellow]Malformed/header entries found ({len(malformed_examples)}):[/bold yellow]")
        for raw, norm in malformed_examples[:10]:
            console.print(f"  raw: {raw!r}  ->  normalized: {norm!r}")
        if len(malformed_examples) > 10:
            console.print(f"  ... and {len(malformed_examples) - 10} more (see enriched CSV)")

    console.print(f"\n[bold green]Saved enriched facets to:[/bold green] {save_path}\n")


if __name__ == "__main__":
    audit_facets()
