"""
test_vectorstore.py
===================
Unit tests for VectorStoreManager.

These tests cover the components most likely to be asked about
in technical interviews: duplicate detection, ingestion correctness,
retrieval with filters, and the hallucination guard threshold.

Run with: uv run pytest tests/ -v

PEP 8 | OOP
"""

from __future__ import annotations

import uuid

import pytest

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Helpers — isolated ChromaDB per test (does not touch ./data/chroma_db)
# ---------------------------------------------------------------------------


def _make_test_store(tmp_path, monkeypatch) -> VectorStoreManager:
    """Vector store backed by a temporary directory and unique collection name."""
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", f"pytest_{uuid.uuid4().hex[:8]}")
    return VectorStoreManager(settings=Settings())


def _make_chunk(
    *,
    source: str,
    topic: str,
    chunk_text: str,
    difficulty: str = "intermediate",
    chunk_type: str = "concept_explanation",
    related_topics: list[str] | None = None,
    is_bonus: bool = False,
) -> DocumentChunk:
    """Build a DocumentChunk with a consistent content-derived ID."""
    metadata = ChunkMetadata(
        topic=topic,
        difficulty=difficulty,
        type=chunk_type,
        source=source,
        related_topics=related_topics or [],
        is_bonus=is_bonus,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id(source, chunk_text),
        chunk_text=chunk_text,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """A single valid DocumentChunk for use across tests."""
    text = (
        "Long Short-Term Memory networks solve the vanishing gradient problem "
        "through gated mechanisms: the forget gate, input gate, and output gate. "
        "These gates control information flow through the cell state, allowing "
        "the network to maintain relevant information across long sequences."
    )
    return _make_chunk(
        source="test_lstm.md",
        topic="LSTM",
        chunk_text=text,
        related_topics=["RNN", "vanishing_gradient"],
    )


@pytest.fixture
def bonus_chunk() -> DocumentChunk:
    """A bonus topic chunk (GAN) for testing topic filtering."""
    text = (
        "Generative Adversarial Networks consist of two competing neural networks: "
        "a generator that produces synthetic data and a discriminator that "
        "distinguishes real from generated samples. Training is a minimax game."
    )
    return _make_chunk(
        source="test_gan.md",
        topic="GAN",
        chunk_text=text,
        difficulty="advanced",
        chunk_type="architecture",
        related_topics=["autoencoder", "generative_models"],
        is_bonus=True,
    )


# ---------------------------------------------------------------------------
# Chunk ID Generation Tests
# ---------------------------------------------------------------------------


class TestChunkIdGeneration:
    """Tests for the deterministic chunk ID generation logic."""

    def test_same_content_produces_same_id(self) -> None:
        """Identical source and text must always produce the same ID."""
        id1 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
        id2 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
        assert id1 == id2

    def test_different_content_produces_different_id(self) -> None:
        """Different text must produce different IDs."""
        id1 = VectorStoreManager.generate_chunk_id("lstm.md", "content one")
        id2 = VectorStoreManager.generate_chunk_id("lstm.md", "content two")
        assert id1 != id2

    def test_different_source_produces_different_id(self) -> None:
        """Same text from different sources must produce different IDs."""
        id1 = VectorStoreManager.generate_chunk_id("file_a.md", "same text")
        id2 = VectorStoreManager.generate_chunk_id("file_b.md", "same text")
        assert id1 != id2

    def test_id_is_16_characters(self) -> None:
        """Generated IDs must be exactly 16 hex characters."""
        chunk_id = VectorStoreManager.generate_chunk_id("source.md", "text")
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)


# ---------------------------------------------------------------------------
# Duplicate Detection Tests
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """
    Tests for the check_duplicate method.

    Interview talking point: these tests verify the core invariant
    of the duplicate guard — the system must never silently ingest
    the same content twice.
    """

    def test_new_chunk_is_not_duplicate(
        self, tmp_path, monkeypatch, sample_chunk: DocumentChunk
    ) -> None:
        """A chunk that has never been ingested must not be flagged as duplicate."""
        store = _make_test_store(tmp_path, monkeypatch)
        assert store.check_duplicate(sample_chunk.chunk_id) is False

    def test_ingested_chunk_is_duplicate(
        self, tmp_path, monkeypatch, sample_chunk: DocumentChunk
    ) -> None:
        """A chunk that has been ingested must be flagged as duplicate on re-check."""
        store = _make_test_store(tmp_path, monkeypatch)
        result = store.ingest([sample_chunk])
        assert result.ingested == 1
        assert store.check_duplicate(sample_chunk.chunk_id) is True

    def test_ingestion_skips_duplicate(
        self, tmp_path, monkeypatch, sample_chunk: DocumentChunk
    ) -> None:
        """Ingesting the same chunk twice must result in skipped=1 on second call."""
        store = _make_test_store(tmp_path, monkeypatch)
        first = store.ingest([sample_chunk])
        assert first.ingested == 1
        assert first.skipped == 0

        second = store.ingest([sample_chunk])
        assert second.ingested == 0
        assert second.skipped == 1


# ---------------------------------------------------------------------------
# Retrieval Tests
# ---------------------------------------------------------------------------


class TestRetrieval:
    """
    Tests for the query method.

    These cover the hallucination guard threshold and metadata filtering,
    both of which are common interview discussion topics.
    """

    def test_relevant_query_returns_results(
        self, tmp_path, monkeypatch, sample_chunk: DocumentChunk
    ) -> None:
        """A query semantically similar to an ingested chunk must return results."""
        store = _make_test_store(tmp_path, monkeypatch)
        store.ingest([sample_chunk])

        results = store.query("LSTM forget gate input output mechanism")
        assert len(results) > 0
        returned_ids = {chunk.chunk_id for chunk in results}
        assert sample_chunk.chunk_id in returned_ids
        assert all(
            chunk.score >= store._settings.similarity_threshold for chunk in results
        )

    def test_irrelevant_query_returns_empty(
        self, tmp_path, monkeypatch, sample_chunk: DocumentChunk
    ) -> None:
        """
        A query with no semantic similarity to the corpus must return empty list.

        This tests the hallucination guard threshold. The system must return
        an empty list — not low-quality chunks — when nothing matches.
        """
        store = _make_test_store(tmp_path, monkeypatch)
        store.ingest([sample_chunk])

        results = store.query("history of the roman empire ancient politics")
        assert results == []

    def test_topic_filter_restricts_results(
        self,
        tmp_path,
        monkeypatch,
        sample_chunk: DocumentChunk,
        bonus_chunk: DocumentChunk,
    ) -> None:
        """Results with topic_filter='LSTM' must not include GAN chunks."""
        store = _make_test_store(tmp_path, monkeypatch)
        store.ingest([sample_chunk, bonus_chunk])

        results = store.query(
            "neural network architecture training",
            topic_filter="LSTM",
        )
        assert len(results) > 0
        assert all(chunk.metadata.topic == "LSTM" for chunk in results)

    def test_results_sorted_by_score_descending(
        self, tmp_path, monkeypatch, sample_chunk: DocumentChunk
    ) -> None:
        """Retrieved chunks must be sorted with highest similarity first."""
        second_lstm = _make_chunk(
            source="test_lstm_extra.md",
            topic="LSTM",
            chunk_text=(
                "The forget gate in an LSTM decides what information to remove "
                "from the cell state. The input gate controls new information "
                "written into the cell state. The output gate selects what the "
                "hidden state exposes at each time step."
            ),
            related_topics=["RNN", "gates"],
        )
        store = _make_test_store(tmp_path, monkeypatch)
        store.ingest([sample_chunk, second_lstm])

        results = store.query("LSTM forget gate input output cell state", k=4)
        assert len(results) >= 2
        scores = [chunk.score for chunk in results]
        assert scores == sorted(scores, reverse=True)
