"""Regression tests for vector-store persistence consistency."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers import make_normalized_batch as _make_normalized_batch
from tests.helpers import make_normalized_vec as _make_normalized_vec


def _snapshot_store_state(vs):
    return vs.size, list(vs._id_map), dict(vs._uuid_to_pos)


def _assert_rejected_without_mutation(vs, operation, *, match):
    before = _snapshot_store_state(vs)
    with pytest.raises(ValueError, match=match):
        operation()
    assert _snapshot_store_state(vs) == before


class TestVectorStorePersistenceRollback:
    def test_add_rolls_back_in_memory_state_when_save_fails(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        vec = _make_normalized_vec(seed=42)

        with patch.object(vs, "_save", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                vs.add("ep-1", vec)

        assert vs.size == 0
        assert vs._id_map == []
        assert vs._uuid_to_pos == {}

    def test_add_batch_rolls_back_in_memory_state_when_save_fails(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        ids = ["ep-1", "ep-2", "ep-3"]
        vecs = _make_normalized_batch(len(ids), seed=42)

        with patch.object(vs, "_save", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                vs.add_batch(ids, vecs)

        assert vs.size == 0
        assert vs._id_map == []
        assert vs._uuid_to_pos == {}


class TestVectorStoreInputValidation:
    def test_add_batch_rejects_id_vector_count_mismatch_without_mutation(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        ids = ["ep-1", "ep-2"]
        vecs = _make_normalized_batch(1, seed=42)

        _assert_rejected_without_mutation(
            vs,
            lambda: vs.add_batch(ids, vecs),
            match="received 1 vectors for 2 episode IDs",
        )

    def test_add_rejects_wrong_dimension_without_mutation(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        vec = _make_normalized_vec(dim=383, seed=42)

        _assert_rejected_without_mutation(
            vs,
            lambda: vs.add("ep-wrong-dim", vec),
            match="embedding dimension mismatch",
        )

    def test_add_batch_rejects_wrong_dimension_without_mutation(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        vecs = _make_normalized_batch(2, dim=383, seed=42)

        _assert_rejected_without_mutation(
            vs,
            lambda: vs.add_batch(["ep-1", "ep-2"], vecs),
            match="embedding dimension mismatch",
        )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
    def test_add_rejects_non_finite_values_without_mutation(self, bad_value):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        vec = np.full((384,), bad_value, dtype=np.float32)

        _assert_rejected_without_mutation(
            vs,
            lambda: vs.add("ep-non-finite", vec),
            match="embeddings must be finite",
        )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
    def test_add_batch_rejects_non_finite_values_without_mutation(self, bad_value):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        vecs = np.full((2, 384), bad_value, dtype=np.float32)

        _assert_rejected_without_mutation(
            vs,
            lambda: vs.add_batch(["ep-1", "ep-2"], vecs),
            match="embeddings must be finite",
        )


class TestVectorStoreReloadSignals:
    def test_add_emits_reload_signal_for_other_processes(self):
        from consolidation_memory.vector_store import VectorStore

        vs1 = VectorStore()
        vs2 = VectorStore()
        vec = _make_normalized_vec(seed=7)

        vs1.add("ep-1", vec)

        # Force a stale view so the test depends on the signal file, not mtime granularity.
        vs2._last_load_time = 0.0
        assert vs2.reload_if_stale() is True
        assert vs2.size == 1
        assert vs2.search(vec, k=1)[0][0] == "ep-1"

    def test_remove_and_compact_emit_reload_signal_for_other_processes(self):
        from consolidation_memory.vector_store import VectorStore

        ids = ["ep-a", "ep-b", "ep-c"]
        vecs = _make_normalized_batch(len(ids), seed=11)

        vs1 = VectorStore()
        vs2 = VectorStore()
        vs1.add_batch(ids, vecs)

        vs2._last_load_time = 0.0
        assert vs2.reload_if_stale() is True
        assert vs2.size == 3

        assert vs1.remove("ep-b") is True
        vs2._last_load_time = 0.0
        assert vs2.reload_if_stale() is True
        assert vs2.size == 2
        assert "ep-b" not in {uid for uid, _ in vs2.search(vecs[1], k=3)}
        assert vs2.tombstone_count == 1

        assert vs1.compact() == 1
        vs2._last_load_time = 0.0
        assert vs2.reload_if_stale() is True
        assert vs2.size == 2
        assert vs2.tombstone_count == 0
        assert set(vs2._id_map) == {"ep-a", "ep-c"}


class TestVectorStoreWriteLease:
    def test_add_uses_process_write_lease(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        vec = _make_normalized_vec(seed=19)
        calls = {"count": 0}

        @contextmanager
        def counted_acquire():
            calls["count"] += 1
            yield

        vs._write_lease = SimpleNamespace(acquire=counted_acquire)
        vs.add("ep-lease-1", vec)
        assert calls["count"] == 1

    def test_mutation_lease_is_reentrant_for_nested_saves(self):
        from consolidation_memory.vector_store import VectorStore

        vs = VectorStore()
        calls = {"count": 0}

        @contextmanager
        def counted_acquire():
            calls["count"] += 1
            yield

        vs._write_lease = SimpleNamespace(acquire=counted_acquire)
        with vs._mutation_lease():
            vs._save()
            vs._save_tombstones()

        assert calls["count"] == 1
