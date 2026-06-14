"""Tests for recall time-budget helpers and bounded embedding behavior."""

from __future__ import annotations

import time

import numpy as np
import pytest

from consolidation_memory import recall_budget
from consolidation_memory.backends import encode_documents
from consolidation_memory.embedding_disk_cache import embed_items_incremental


class TestRecallBudget:
    def test_deadline_is_inactive_by_default(self):
        assert recall_budget.is_active() is False
        assert recall_budget.deadline_exceeded() is False

    def test_deadline_exceeded_inside_context(self):
        deadline = time.monotonic() - 0.01
        with recall_budget.recall_deadline(deadline):
            assert recall_budget.is_active() is True
            assert recall_budget.deadline_exceeded() is True

    def test_deadline_resets_after_context(self):
        with recall_budget.recall_deadline(time.monotonic() + 5.0):
            assert recall_budget.is_active() is True
        assert recall_budget.is_active() is False

    def test_encode_documents_raises_when_budget_exhausted(self, monkeypatch):
        monkeypatch.setattr(
            "consolidation_memory.config.get_config",
            lambda: type("Cfg", (), {"EMBEDDING_ENCODE_BATCH_SIZE": 1})(),
        )

        def _fake_chunk(texts: list[str]) -> np.ndarray:
            time.sleep(0.02)
            return np.ones((len(texts), 4), dtype=np.float32)

        monkeypatch.setattr(
            "consolidation_memory.backends._encode_documents_chunk",
            _fake_chunk,
        )
        monkeypatch.setattr(
            "consolidation_memory.backends.get_dimension",
            lambda: 4,
        )

        deadline = time.monotonic() + 0.01
        with recall_budget.recall_deadline(deadline):
            with pytest.raises(recall_budget.RecallBudgetExceeded):
                encode_documents(["one", "two", "three"])

    def test_embed_items_incremental_returns_none_when_budget_exhausted(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CONSOLIDATION_MEMORY_EMBEDDING_DISK_CACHE_ENABLED", "0")
        monkeypatch.setattr(
            "consolidation_memory.config.get_config",
            lambda: type(
                "Cfg",
                (),
                {
                    "EMBEDDING_ENCODE_BATCH_SIZE": 1,
                    "EMBEDDING_DISK_CACHE_ENABLED": False,
                    "EMBEDDING_CACHE_DIR": tmp_path,
                    "EMBEDDING_BACKEND": "fastembed",
                    "EMBEDDING_MODEL_NAME": "test",
                    "EMBEDDING_DIMENSION": 4,
                    "EMBEDDING_CACHE_WRITE_LOCK_PATH": tmp_path / "lock",
                    "EMBEDDING_CACHE_WRITE_LOCK_TIMEOUT_SECONDS": 5.0,
                },
            )(),
        )

        call_count = {"n": 0}

        def _fake_encode(texts: list[str]) -> np.ndarray:
            call_count["n"] += 1
            time.sleep(0.03)
            return np.ones((len(texts), 4), dtype=np.float32)

        monkeypatch.setattr(
            "consolidation_memory.backends.encode_documents",
            _fake_encode,
        )
        monkeypatch.setattr(
            "consolidation_memory.embedding_disk_cache.get_dimension",
            lambda: 4,
        )

        items = [(f"id-{i}", f"text {i}") for i in range(5)]
        deadline = time.monotonic() + 0.04
        with recall_budget.recall_deadline(deadline):
            result = embed_items_incremental(items, namespace="records")

        assert result is None
        assert call_count["n"] >= 1