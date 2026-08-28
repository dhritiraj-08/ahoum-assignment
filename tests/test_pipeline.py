"""
Unit tests for src/pipeline.py (retrieve -> batch -> score wiring) and
src/embeddings.py (retrieve_relevant_facets / FAISS retrieval).

No Ollama and no real sentence-transformers download is required. The
embedding model is stubbed out with a small deterministic fake (see
_FakeSentenceTransformer below), so these tests build a tiny, fully offline
FAISS index instead of touching outputs/faiss_index.bin -- fast, and not
dependent on the real 399-facet dataset or an internet connection.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src import embeddings, pipeline


class _FakeSentenceTransformer:
    """Deterministic stand-in for SentenceTransformer.encode(): maps each
    input string to a fixed-size unit vector derived from a hash of the
    string, so the same text always encodes to the same vector without
    downloading or running a real model."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vec = rng.rand(16).astype("float32")
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype="float32")


FAKE_FACET_NAMES = ["Risktaking", "Compassion", "Hesitation", "Aloofness", "Merriness"]


@pytest.fixture
def fake_embedding_index(tmp_path, monkeypatch):
    """Builds a tiny, fully offline FAISS index (5 fake observable facets)
    and points embeddings.py's module-level paths + caches at it, so
    retrieve_relevant_facets() reads from this temporary index instead of
    the real outputs/ directory."""
    enriched_csv = tmp_path / "enriched_facets.csv"
    pd.DataFrame(
        {
            "facet_normalized": FAKE_FACET_NAMES,
            "category": ["personality_trait"] * 5,
            "conversation_observable": [True] * 5,
            "sensitivity": ["low"] * 5,
            "scoring_anchors": ["1=low; 5=high"] * 5,
            "abstention_reason": [""] * 5,
        }
    ).to_csv(enriched_csv, index=False)

    monkeypatch.setattr(embeddings, "FAISS_INDEX_PATH", tmp_path / "faiss_index.bin")
    monkeypatch.setattr(embeddings, "FACETS_JSON_PATH", tmp_path / "observable_facets.json")
    monkeypatch.setattr(embeddings, "_model", _FakeSentenceTransformer())
    # Reset module-level caches so a previously-loaded real/other-test index
    # can't leak into this test via the shared module object.
    monkeypatch.setattr(embeddings, "_index", None)
    monkeypatch.setattr(embeddings, "_facets_meta", None)

    embeddings.build_index(enriched_csv_path=enriched_csv)
    return tmp_path


# ---------------------------------------------------------------------------
# 5. retrieve_relevant_facets returns results
# ---------------------------------------------------------------------------


def test_retrieve_relevant_facets_returns_results(fake_embedding_index):
    results = embeddings.retrieve_relevant_facets("some conversation about taking risks", top_k=3)

    assert len(results) == 3
    for r in results:
        assert r["facet_normalized"] in FAKE_FACET_NAMES
        assert "similarity" in r
    # results should come back ranked best-match-first
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_retrieve_relevant_facets_empty_conversation_returns_empty(fake_embedding_index):
    assert embeddings.retrieve_relevant_facets("", top_k=5) == []
    assert embeddings.retrieve_relevant_facets("   ", top_k=5) == []


def test_retrieve_relevant_facets_missing_index_raises_clear_error(monkeypatch, tmp_path):
    """If --embed hasn't been run yet, fail with a clear, actionable
    message instead of a raw FileNotFoundError from inside faiss."""
    monkeypatch.setattr(embeddings, "FAISS_INDEX_PATH", tmp_path / "does_not_exist.bin")
    monkeypatch.setattr(embeddings, "FACETS_JSON_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(embeddings, "_index", None)
    monkeypatch.setattr(embeddings, "_facets_meta", None)

    with pytest.raises(FileNotFoundError):
        embeddings.retrieve_relevant_facets("some conversation", top_k=3)


# ---------------------------------------------------------------------------
# 4. top_k returns the correct number of facets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k,expected", [(1, 1), (3, 3), (5, 5), (100, 5)])
def test_retrieve_relevant_facets_respects_top_k(fake_embedding_index, top_k, expected):
    """top_k=100 against only 5 indexed facets should clip to 5, not error
    or pad -- exercises the max(1, min(top_k, index.ntotal)) clamp."""
    results = embeddings.retrieve_relevant_facets("a conversation", top_k=top_k)
    assert len(results) == expected


def test_run_pipeline_retrieves_exactly_top_k_facets(monkeypatch):
    """Pipeline-level test: run_pipeline should ask retrieval for exactly
    the requested top_k, and total_facets_retrieved should reflect however
    many retrieval actually returned. Both retrieval and scoring are
    mocked, so this test has zero dependency on FAISS, the embedding
    model, or Ollama -- it only tests pipeline.py's own wiring."""
    fake_facets = [
        {"facet_normalized": f"Facet{i}", "category": "personality_trait", "scoring_anchors": "1=low; 5=high"}
        for i in range(7)
    ]
    mock_retrieve = MagicMock(return_value=fake_facets)
    monkeypatch.setattr(pipeline, "retrieve_relevant_facets", mock_retrieve)

    mock_score = MagicMock(
        return_value=[
            {"facet": f["facet_normalized"], "score": 3, "status": "scored", "confidence": "medium", "evidence": "test"}
            for f in fake_facets
        ]
    )
    monkeypatch.setattr(pipeline, "score_facets", mock_score)

    result = pipeline.run_pipeline("some conversation", top_k=7, save_output=False)

    mock_retrieve.assert_called_once_with("some conversation", top_k=7)
    assert result["total_facets_retrieved"] == 7
    assert result["scored"] == 7
    assert result["abstained"] == 0
    assert len(result["results"]) == 7


@pytest.mark.parametrize("requested_top_k", [10, 25, 40])
def test_run_pipeline_passes_through_different_top_k_values(monkeypatch, requested_top_k):
    """Same wiring test as above, but sweeping the actual top_k values used
    elsewhere in this project (10 was the original default, 40 is the
    value src/pipeline.py was tuned to after the retrieval-miss
    investigation in docs/DECISIONS.md #1)."""
    fake_facets = [
        {"facet_normalized": f"Facet{i}", "category": "personality_trait", "scoring_anchors": "x"}
        for i in range(requested_top_k)
    ]
    mock_retrieve = MagicMock(return_value=fake_facets)
    monkeypatch.setattr(pipeline, "retrieve_relevant_facets", mock_retrieve)
    monkeypatch.setattr(pipeline, "score_facets", MagicMock(return_value=[]))

    result = pipeline.run_pipeline("some conversation", top_k=requested_top_k, save_output=False)

    mock_retrieve.assert_called_once_with("some conversation", top_k=requested_top_k)
    assert result["total_facets_retrieved"] == requested_top_k


def test_run_pipeline_handles_empty_conversation():
    result = pipeline.run_pipeline("   ", save_output=False)
    assert result["total_facets_retrieved"] == 0
    assert "error" in result


def test_run_pipeline_deduplicates_repeated_facets(monkeypatch):
    """If the same facet name shows up twice (shouldn't normally happen,
    but pipeline.py explicitly guards against it), the result list should
    only contain it once, keeping the first occurrence."""
    fake_facets = [{"facet_normalized": "Risktaking", "category": "personality_trait", "scoring_anchors": "x"}]
    monkeypatch.setattr(pipeline, "retrieve_relevant_facets", MagicMock(return_value=fake_facets))
    monkeypatch.setattr(
        pipeline,
        "score_facets",
        MagicMock(
            return_value=[
                {"facet": "Risktaking", "score": 4, "status": "scored", "confidence": "high", "evidence": "first"},
                {"facet": "Risktaking", "score": 2, "status": "scored", "confidence": "low", "evidence": "second"},
            ]
        ),
    )

    result = pipeline.run_pipeline("some conversation", save_output=False)

    assert len(result["results"]) == 1
    assert result["results"][0]["evidence"] == "first"


def test_run_pipeline_survives_retrieval_failure(monkeypatch):
    """A retrieval-side exception (e.g. index missing) should come back as
    a reported error, not an unhandled crash -- important since main.py and
    app.py both call run_pipeline() directly from user-facing code paths."""
    monkeypatch.setattr(pipeline, "retrieve_relevant_facets", MagicMock(side_effect=FileNotFoundError("no index")))

    result = pipeline.run_pipeline("some conversation", save_output=False)

    assert "error" in result
    assert result["results"] == []
