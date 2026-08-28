"""
Unit tests for src/scorer.py.

src.scorer._client (the Ollama client) is mocked/monkeypatched in every
test here -- no real Ollama connection is ever made, so these tests run
fully offline and don't depend on Ollama being installed or running.

Since scorer.py added hybrid Ollama/Groq backend detection, score_facet_batch
now routes through detect_backend() before it ever touches _client. The
autouse `force_ollama_backend` fixture below pins _active_backend to
"ollama" for every test in this file except the dedicated backend-detection
tests at the bottom (which explicitly reset it) -- this keeps all the
existing Ollama-path tests deterministic regardless of whether real Ollama
happens to be running on whatever machine executes the suite.
"""
from unittest.mock import MagicMock

import pytest

from src import scorer


def _fake_facet(name, category="personality_trait", anchor="1=low; 5=high"):
    return {"facet_normalized": name, "category": category, "scoring_anchors": anchor}


def _fake_ollama_response(content: str):
    """Shape a fake response the way ollama.Client.chat() returns it."""
    return {"message": {"content": content}}


@pytest.fixture(autouse=True)
def force_ollama_backend(monkeypatch):
    """Pin backend detection to "ollama" so every test in this file (other
    than the backend-detection tests, which override this) exercises the
    mocked _client regardless of the real machine's Ollama/GROQ_API_KEY
    state."""
    monkeypatch.setattr(scorer, "_active_backend", "ollama")


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


# ---------------------------------------------------------------------------
# Hybrid backend detection (src/scorer.py: detect_backend / _call_llm)
#
# These tests override the autouse force_ollama_backend fixture's effect by
# resetting _active_backend to None themselves, then control
# _check_ollama_available() and the GROQ_API_KEY env var directly -- so
# they're just as offline/deterministic as the rest of the file, they just
# test the detection logic itself instead of assuming Ollama is picked.
# ---------------------------------------------------------------------------


def test_detect_backend_prefers_ollama_when_available(monkeypatch):
    monkeypatch.setattr(scorer, "_active_backend", None)
    monkeypatch.setattr(scorer, "_check_ollama_available", lambda: True)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert scorer.detect_backend() == "ollama"


def test_detect_backend_falls_back_to_groq_when_ollama_unavailable(monkeypatch):
    monkeypatch.setattr(scorer, "_active_backend", None)
    monkeypatch.setattr(scorer, "_check_ollama_available", lambda: False)
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    assert scorer.detect_backend() == "groq"


def test_detect_backend_raises_clear_error_when_neither_available(monkeypatch):
    monkeypatch.setattr(scorer, "_active_backend", None)
    monkeypatch.setattr(scorer, "_check_ollama_available", lambda: False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No LLM backend available"):
        scorer.detect_backend()


def test_detect_backend_result_is_cached(monkeypatch):
    """Once resolved, detect_backend() shouldn't re-probe Ollama on every
    call -- only when force_refresh=True is passed."""
    monkeypatch.setattr(scorer, "_active_backend", None)
    probe = MagicMock(return_value=True)
    monkeypatch.setattr(scorer, "_check_ollama_available", probe)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert scorer.detect_backend() == "ollama"
    assert scorer.detect_backend() == "ollama"
    probe.assert_called_once()  # second call used the cache, not a re-probe

    probe.return_value = False
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    assert scorer.detect_backend(force_refresh=True) == "groq"


def test_call_llm_routes_to_groq_when_backend_is_groq(monkeypatch):
    """When the active backend is groq, _call_llm must hit the Groq client,
    not Ollama's -- verified by mocking Groq's SDK shape directly (no real
    API key or network call)."""
    monkeypatch.setattr(scorer, "_active_backend", "groq")

    fake_groq_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='[{"facet": "Risktaking", "score": 3, "status": "scored", "confidence": "high", "evidence": "test"}]'))]
    fake_groq_client.chat.completions.create.return_value = fake_response
    monkeypatch.setattr(scorer, "_get_groq_client", lambda: fake_groq_client)

    # _client (Ollama) must never be touched on this path.
    fake_ollama_client = MagicMock()
    monkeypatch.setattr(scorer, "_client", fake_ollama_client)

    results = scorer.score_facet_batch([_fake_facet("Risktaking")], "some conversation text")

    assert results[0]["status"] == "scored"
    assert results[0]["score"] == 3
    fake_groq_client.chat.completions.create.assert_called_once()
    fake_ollama_client.chat.assert_not_called()
    call_kwargs = fake_groq_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == scorer.GROQ_MODEL_NAME
