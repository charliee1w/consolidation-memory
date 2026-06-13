"""Tests for disk-backed incremental embedding cache."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from consolidation_memory.config import override_config, reset_config
from consolidation_memory.embedding_disk_cache import (
    clear_namespace,
    embed_items_incremental,
)


@pytest.fixture
def disk_cache_dir(tmp_path):
    data_dir = tmp_path / "data" / "projects" / "default"
    data_dir.mkdir(parents=True, exist_ok=True)
    reset_config(
        _base_data_dir=tmp_path / "data",
        active_project="default",
        EMBEDDING_DIMENSION=384,
        EMBEDDING_BACKEND="fastembed",
        EMBEDDING_DISK_CACHE_ENABLED=True,
    )
    clear_namespace("records")
    yield data_dir
    clear_namespace("records")
    reset_config()


def test_embed_items_incremental_persists_and_reuses_vectors(disk_cache_dir):
    vectors = np.random.randn(3, 384).astype(np.float32)
    items = [("a", "alpha"), ("b", "beta"), ("c", "gamma")]

    with patch("consolidation_memory.backends.encode_documents", return_value=vectors) as mock_encode:
        first = embed_items_incremental(items, namespace="records")
        assert first is not None
        assert first.shape == (3, 384)
        assert mock_encode.call_count == 1

    with patch("consolidation_memory.backends.encode_documents") as mock_encode:
        second = embed_items_incremental(items, namespace="records")
        assert second is not None
        np.testing.assert_allclose(second, first)
        mock_encode.assert_not_called()

    cache_path = disk_cache_dir / "embedding_cache" / "records" / "vectors.npz"
    assert cache_path.exists()


def test_embed_items_incremental_only_embeds_changed_items(disk_cache_dir):
    initial = np.random.randn(2, 384).astype(np.float32)
    updated = np.random.randn(1, 384).astype(np.float32)
    first_items = [("a", "alpha"), ("b", "beta")]

    with patch("consolidation_memory.backends.encode_documents", return_value=initial):
        embed_items_incremental(first_items, namespace="records")

    second_items = [("a", "alpha"), ("b", "beta changed")]
    with patch("consolidation_memory.backends.encode_documents", return_value=updated) as mock_encode:
        result = embed_items_incremental(second_items, namespace="records")
        assert result is not None
        assert result.shape == (2, 384)
        mock_encode.assert_called_once()
        assert mock_encode.call_args.args[0] == ["beta changed"]


def test_clear_namespace_forces_reembed(disk_cache_dir):
    vectors = np.random.randn(1, 384).astype(np.float32)
    items = [("a", "alpha")]

    with patch("consolidation_memory.backends.encode_documents", return_value=vectors) as mock_encode:
        embed_items_incremental(items, namespace="records")
        clear_namespace("records")
        embed_items_incremental(items, namespace="records")
        assert mock_encode.call_count == 2


def test_fingerprint_mismatch_rebuilds_cache(disk_cache_dir):
    vectors = np.random.randn(1, 384).astype(np.float32)
    items = [("a", "alpha")]

    with patch("consolidation_memory.backends.encode_documents", return_value=vectors) as mock_encode:
        embed_items_incremental(items, namespace="records")
        with override_config(EMBEDDING_MODEL_NAME="different-model"):
            from consolidation_memory import embedding_disk_cache

            embedding_disk_cache._runtime_stores.pop("records", None)
            embed_items_incremental(items, namespace="records")
        assert mock_encode.call_count == 2