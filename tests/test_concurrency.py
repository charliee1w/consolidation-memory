"""Concurrency stress tests for consolidation-memory.

Verifies thread safety of store, recall, and vector_store operations
under concurrent access. All backends are mocked.

Run with: python -m pytest tests/test_concurrency.py -v
"""

import threading
from unittest.mock import patch

from helpers import make_normalized_vec as _make_vec


class TestConcurrentStore:
    @patch("consolidation_memory.backends.encode_documents")
    def test_20_threads_store(self, mock_embed, tmp_data_dir):
        """20 threads storing simultaneously — no corruption, no errors."""
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema, get_stats

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        _call = [0]
        _lock = threading.Lock()

        def embed_side_effect(texts):
            with _lock:
                _call[0] += 1
                seed = _call[0]
            return _make_vec(seed=seed).reshape(1, -1)

        mock_embed.side_effect = embed_side_effect

        errors = []

        def worker(idx):
            try:
                result = client.store(f"Concurrent episode {idx}", content_type="fact")
                assert result.status == "stored"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), f"Thread {t.name} still alive (possible deadlock)"

        assert not errors, f"Store errors: {errors}"
        stats = get_stats()
        assert stats["episodic_buffer"]["total"] == 20
        assert client._vector_store.size == 20
        client.close()


class TestConcurrentRecall:
    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_10_threads_recall(self, mock_embed, mock_query, tmp_data_dir):
        """10 threads recalling simultaneously — no deadlocks."""
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)
        mock_query.return_value = vec.reshape(1, -1)

        client.store("Shared episode for concurrent recall")

        errors = []

        def worker():
            try:
                result = client.recall("test query", include_knowledge=False)
                assert len(result.episodes) >= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), f"Thread {t.name} still alive (possible deadlock)"

        assert not errors, f"Recall errors: {errors}"
        client.close()


class TestStoreDuringRecall:
    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_interleaved_store_recall(self, mock_embed, mock_query, tmp_data_dir):
        """Store and recall running concurrently — no deadlock or corruption."""
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        _call = [0]
        _lock = threading.Lock()

        def embed_side_effect(texts):
            with _lock:
                _call[0] += 1
                seed = _call[0]
            return _make_vec(seed=seed).reshape(1, -1)

        mock_embed.side_effect = embed_side_effect
        mock_query.return_value = _make_vec(seed=1).reshape(1, -1)

        errors = []

        def store_worker(idx):
            try:
                client.store(f"Concurrent store {idx}")
            except Exception as e:
                errors.append(("store", e))

        def recall_worker():
            try:
                client.recall("test", include_knowledge=False)
            except Exception as e:
                errors.append(("recall", e))

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=store_worker, args=(i,)))
            threads.append(threading.Thread(target=recall_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), f"Thread {t.name} still alive (possible deadlock)"

        assert not errors, f"Errors: {errors}"
        client.close()


class TestCacheVersionRace:
    @patch("consolidation_memory.topic_cache.encode_documents")
    @patch("consolidation_memory.topic_cache.get_all_knowledge_topics")
    def test_topic_cache_invalidate_during_fetch(self, mock_topics, mock_embed, tmp_data_dir):
        """Concurrent invalidate during fetch returns dimensionally consistent data."""
        import numpy as np
        from consolidation_memory import topic_cache

        topic_cache.invalidate()

        mock_topics.return_value = [
            {"title": f"Topic {i}", "summary": f"Summary {i}", "id": i}
            for i in range(5)
        ]
        mock_embed.return_value = np.random.randn(5, 384).astype(np.float32)

        # Populate cache first
        topics, vecs = topic_cache.get_topic_vecs()
        assert len(topics) == 5
        assert vecs.shape[0] == 5

        # Now set up: embed will be slow, and we invalidate mid-fetch
        original_embed = mock_embed.side_effect

        def slow_embed(texts):
            # Invalidate while "embedding" is in progress
            topic_cache.invalidate()
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_embed.side_effect = slow_embed

        topics2, vecs2 = topic_cache.get_topic_vecs()
        # Must be dimensionally consistent
        assert vecs2 is not None
        assert len(topics2) == vecs2.shape[0]

    @patch("consolidation_memory.record_cache.encode_documents")
    @patch("consolidation_memory.record_cache.get_all_active_records")
    def test_record_cache_invalidate_during_fetch(self, mock_records, mock_embed, tmp_data_dir):
        """Concurrent invalidate during record fetch returns dimensionally consistent data."""
        import numpy as np
        from consolidation_memory import record_cache

        record_cache.invalidate()

        mock_records.return_value = [
            {"embedding_text": f"Record {i}", "id": i}
            for i in range(5)
        ]
        mock_embed.return_value = np.random.randn(5, 384).astype(np.float32)

        # Populate cache first
        records, vecs = record_cache.get_record_vecs(include_expired=True)
        assert len(records) == 5
        assert vecs.shape[0] == 5

        # Invalidate during embed
        def slow_embed(texts):
            record_cache.invalidate()
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_embed.side_effect = slow_embed

        records2, vecs2 = record_cache.get_record_vecs(include_expired=True)
        assert vecs2 is not None
        assert len(records2) == vecs2.shape[0]


class TestStoreDuringConsolidation:
    @patch("consolidation_memory.backends.encode_documents")
    def test_store_while_consolidation_locked(self, mock_embed, tmp_data_dir):
        """Store should succeed even when consolidation lock is held."""
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        # Simulate consolidation holding the lock
        client._consolidation_lock.acquire()

        result = client.store("Store during consolidation")
        assert result.status == "stored"

        client._consolidation_lock.release()
        client.close()
