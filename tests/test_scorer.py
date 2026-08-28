"""
Unit tests for src/scorer.py.

src.scorer._client (the Ollama client) is mocked/monkeypatched in every
test here -- no real Ollama connection is ever made, so these tests run
fully offline and don't depend on Ollama being installed or running.
"""
from unittest.mock import MagicMock

import pytest

from src import scorer


def _fake_facet(name, category="personality_trait", anchor="1=low; 5=high"):
    return {"facet_normalized": name, "category": category, "scoring_anchors": anchor}


def _fake_ollama_response(content: str):
    """Shape a fake response the way ollama.Client.chat() returns it."""
    return {"message": {"content": content}}


# ---------------------------------------------------------------------------
# 3. Malformed JSON returns parse_error, does not crash the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_content",
    [
        "I'm sorry, I cannot provide scores for these facets right now.",  # no JSON at all
        "{invalid json here",  # truncated/broken JSON
        "",  # empty response
    ],
)
def test_malformed_llm_output_returns_parse_error_not_crash(monkeypatch, bad_content):
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response(bad_content)
    monkeypatch.setattr(scorer, "_client", fake_client)

    facets = [_fake_facet("Risktaking"), _fake_facet("Compassion")]

    # The whole point of the defensive parsing: this must NOT raise.
    results = scorer.score_facet_batch(facets, "some conversation text")

    assert len(results) == 2
    for r in results:
        assert r["status"] == "parse_error"
        assert r["score"] is None
        assert r["facet"] in {"Risktaking", "Compassion"}


def test_ollama_connection_failure_returns_parse_error_not_crash(monkeypatch):
    """If the Ollama call itself raises (e.g. connection refused because
    Ollama isn't running, or the llama-server worker crashed -- see
    docs/DEBUGGING.md #3), score_facet_batch must catch it, not propagate."""
    fake_client = MagicMock()
    fake_client.chat.side_effect = ConnectionError("connection refused")
    monkeypatch.setattr(scorer, "_client", fake_client)

    results = scorer.score_facet_batch([_fake_facet("Risktaking")], "some conversation text")

    assert len(results) == 1
    assert results[0]["status"] == "parse_error"
    assert "connection refused" in results[0]["evidence"]


def test_empty_batch_returns_empty_list():
    assert scorer.score_facet_batch([], "some conversation text") == []


def test_batch_larger_than_limit_raises_valueerror():
    facets = [_fake_facet(f"Facet{i}") for i in range(scorer.BATCH_SIZE + 1)]
    with pytest.raises(ValueError):
        scorer.score_facet_batch(facets, "some conversation text")


# ---------------------------------------------------------------------------
# Bonus: "messy but recoverable" JSON shapes should still parse correctly
# (mirrors the manual verification behind docs/DEBUGGING.md #2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapped_content",
    [
        '```json\n[{"facet": "Risktaking", "score": 4, "status": "scored", "confidence": "high", "evidence": "test"}]\n```',
        'Sure, here you go: [{"facet": "Risktaking", "score": 4, "status": "scored", "confidence": "high", "evidence": "test"}] Hope this helps!',
        '{"results": [{"facet": "Risktaking", "score": 4, "status": "scored", "confidence": "high", "evidence": "test"}]}',
    ],
)
def test_recoverable_malformed_shapes_still_parse(monkeypatch, wrapped_content):
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response(wrapped_content)
    monkeypatch.setattr(scorer, "_client", fake_client)

    results = scorer.score_facet_batch([_fake_facet("Risktaking")], "some conversation text")

    assert results[0]["status"] == "scored"
    assert results[0]["score"] == 4


# ---------------------------------------------------------------------------
# Bonus: medical facets are hard-blocked -- the LLM is never even called
# for them, regardless of what the model might have said.
# ---------------------------------------------------------------------------


def test_medical_facet_never_sent_to_llm(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(scorer, "_client", fake_client)

    results = scorer.score_facet_batch(
        [_fake_facet("Basophil count", category="medical_biological")],
        "I've been feeling tired lately",
    )

    assert results[0]["status"] == "not_observable"
    assert results[0]["score"] is None
    fake_client.chat.assert_not_called()


def test_out_of_range_score_is_forced_to_abstain(monkeypatch):
    """A score outside 1-5 (a model slip-up) must never pass through as-is."""
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response(
        '[{"facet": "Risktaking", "score": 9, "status": "scored", "confidence": "high", "evidence": "test"}]'
    )
    monkeypatch.setattr(scorer, "_client", fake_client)

    results = scorer.score_facet_batch([_fake_facet("Risktaking")], "some conversation text")

    assert results[0]["score"] is None
    assert results[0]["status"] == "insufficient_evidence"
