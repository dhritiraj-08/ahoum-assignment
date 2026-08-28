"""
Unit tests for src/audit.py.

None of these touch the real data/Facets_Assignment.csv -- each test that
needs a CSV builds a small, self-contained temporary one (via pytest's
tmp_path fixture), so the tests are fast, deterministic, and don't break if
the real 399-row dataset ever changes. No Ollama, no network, no GPU.
"""
import pandas as pd
import pytest

from src.audit import audit_facets, classify_facet, normalize_facet_name

# ---------------------------------------------------------------------------
# 1. Medical/biological facets are classified as not conversation-observable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "facet_name",
    ["FSH level", "Basophil count", "Parathyroid-hormone level", "Blood pressure"],
)
def test_medical_facets_classified_medical_biological(facet_name):
    """Known medical/biological facet names should classify as
    medical_biological via the keyword rules in classify_facet()."""
    normalized, had_colon = normalize_facet_name(facet_name)
    category = classify_facet(normalized, had_colon)
    assert category == "medical_biological"


def test_medical_facets_not_observable_via_full_audit(tmp_path):
    """End-to-end: run the actual audit_facets() pipeline over a small CSV
    and confirm medical facets come out conversation_observable=False, with
    a real abstention_reason -- and that a genuine trait in the same file
    is NOT swept into the medical bucket alongside them."""
    raw_csv = tmp_path / "facets.csv"
    pd.DataFrame({"Facets": ["Risktaking", "FSH level", "Basophil count", "Compassion"]}).to_csv(
        raw_csv, index=False
    )
    save_path = tmp_path / "enriched.csv"

    enriched = audit_facets(raw_csv_path=raw_csv, save_path=save_path)

    medical_rows = enriched[enriched["facet_normalized"].isin(["FSH level", "Basophil count"])]
    assert len(medical_rows) == 2
    assert (medical_rows["category"] == "medical_biological").all()
    assert (medical_rows["conversation_observable"] == False).all()  # noqa: E712
    assert (medical_rows["abstention_reason"].str.len() > 0).all()

    risktaking_row = enriched[enriched["facet_normalized"] == "Risktaking"].iloc[0]
    assert risktaking_row["category"] == "personality_trait"
    assert risktaking_row["conversation_observable"] == True  # noqa: E712
    assert save_path.exists()


# ---------------------------------------------------------------------------
# 2. Header/malformed rows are detected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_name",
    ["Democratic Leadership:", "Computer Skills:", "Achievement Motivation:"],
)
def test_trailing_colon_entries_classified_malformed(raw_name):
    """A raw facet string whose last character is ':' is a strong signal of
    a leftover spreadsheet header row -- should be flagged malformed."""
    normalized, had_colon = normalize_facet_name(raw_name)
    assert had_colon is True
    assert classify_facet(normalized, had_colon) == "header_or_malformed"


def test_categorical_type_of_prefix_classified_malformed():
    """'Type of X' / 'Types of X' facets are categorical labels, not
    ordinal 1-5 traits -- also flagged malformed even without a colon."""
    normalized, had_colon = normalize_facet_name("Types of Mindfulness Techniques Used")
    assert classify_facet(normalized, had_colon) == "header_or_malformed"


def test_mid_string_colon_not_treated_as_malformed():
    """A colon in the MIDDLE of a facet name (a sub-definition, e.g.
    'Patience: Resistance to anger') is a real facet, not a truncated
    header -- must NOT trigger the malformed rule."""
    normalized, had_colon = normalize_facet_name("Patience: Resistance to anger")
    assert had_colon is False
    assert classify_facet(normalized, had_colon) != "header_or_malformed"


def test_header_rows_not_observable_via_full_audit(tmp_path):
    raw_csv = tmp_path / "facets.csv"
    pd.DataFrame({"Facets": ["Democratic Leadership:", "Computer Skills:", "Compassion"]}).to_csv(
        raw_csv, index=False
    )
    save_path = tmp_path / "enriched.csv"

    enriched = audit_facets(raw_csv_path=raw_csv, save_path=save_path)

    header_rows = enriched[enriched["category"] == "header_or_malformed"]
    assert len(header_rows) == 2
    assert (header_rows["conversation_observable"] == False).all()  # noqa: E712
    assert set(header_rows["facet_normalized"]) == {"Democratic Leadership", "Computer Skills"}


def test_audit_raises_clear_error_on_missing_csv(tmp_path):
    """A missing input file should fail loudly and clearly, not with a
    confusing pandas stack trace deep in the call chain."""
    missing_csv = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        audit_facets(raw_csv_path=missing_csv, save_path=tmp_path / "enriched.csv")
