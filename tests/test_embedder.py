"""Tests for the embedding pipeline."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.embeddings.embedder import (
    embed_text,
    embed_batch,
    build_embedding_text,
    get_model,
)


# ── build_embedding_text ─────────────────────────────────────────

class TestBuildEmbeddingText:
    def test_basic_fields(self):
        text = build_embedding_text(
            activity_name="Brainstorming 101",
            stage="Identifying and Ideating",
            grade_band="K-2",
        )
        assert "Brainstorming 101" in text
        assert "Grade level: K-2" in text
        assert "Stage: Identifying and Ideating" in text

    def test_with_description(self):
        text = build_embedding_text(
            activity_name="Prototype Testing",
            stage="Engineering Design",
            grade_band="3-5",
            description="Students build and test prototypes",
        )
        assert "Students build and test prototypes" in text

    def test_with_keywords(self):
        text = build_embedding_text(
            activity_name="Patent Basics",
            stage="Intro",
            grade_band="6-8",
            keywords=["patent", "intellectual property"],
        )
        assert "Keywords: patent, intellectual property" in text

    def test_pipe_separator(self):
        text = build_embedding_text(
            activity_name="Act",
            stage="Stage",
            grade_band="K-2",
        )
        assert " | " in text

    def test_all_fields(self):
        text = build_embedding_text(
            activity_name="Full Activity",
            stage="Design",
            grade_band="9-12",
            description="A complete activity",
            keywords=["design", "build"],
        )
        parts = text.split(" | ")
        assert len(parts) == 5


# ── embed_text (mocked model) ────────────────────────────────────

class TestEmbedText:
    @patch("src.embeddings.embedder.get_model")
    def test_returns_list_of_floats(self, mock_get_model):
        fake_embedding = np.random.rand(384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embedding
        mock_get_model.return_value = mock_model

        result = embed_text("test query")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)
        mock_model.encode.assert_called_once_with("test query", normalize_embeddings=True)

    @patch("src.embeddings.embedder.get_model")
    def test_empty_string(self, mock_get_model):
        fake_embedding = np.zeros(384, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embedding
        mock_get_model.return_value = mock_model

        result = embed_text("")
        assert len(result) == 384


# ── embed_batch (mocked model) ───────────────────────────────────

class TestEmbedBatch:
    @patch("src.embeddings.embedder.get_model")
    def test_returns_list_of_lists(self, mock_get_model):
        fake_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        texts = ["text1", "text2", "text3"]
        result = embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(e) == 384 for e in result)
        mock_model.encode.assert_called_once_with(texts, normalize_embeddings=True, show_progress_bar=True)

    @patch("src.embeddings.embedder.get_model")
    def test_empty_batch(self, mock_get_model):
        fake_embeddings = np.array([]).reshape(0, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        result = embed_batch([])
        assert result == []

    @patch("src.embeddings.embedder.get_model")
    def test_single_item_batch(self, mock_get_model):
        fake_embeddings = np.random.rand(1, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        result = embed_batch(["single text"])
        assert len(result) == 1
        assert len(result[0]) == 384
