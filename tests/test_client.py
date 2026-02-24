"""Tests for MemoryClient — the pure Python API.

Run with: python -m pytest tests/test_client.py -v
"""

import json
from unittest.mock import patch

import numpy as np



def _make_normalized_vec(dim=384, seed=None):
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_normalized_batch(n, dim=384, seed=None):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestClientLifecycle:
    def test_construct_and_close(self):
        from consolidation_memory.database import ensure_schema
        ensure_schema()

        from consolidation_memory.client import MemoryClient
        client = MemoryClient(auto_consolidate=False)
        assert client._vector_store is not None
        client.close()

    def test_context_manager(self):
        from consolidation_memory.database import ensure_schema
        ensure_schema()

        from consolidation_memory.client import MemoryClient
        with MemoryClient(auto_consolidate=False) as client:
            assert client._vector_store is not None
        # after exit, thread should be stopped
        assert client._consolidation_thread is None

    def test_multiple_instances_safe(self):
        from consolidation_memory.database import ensure_schema
        ensure_schema()

        from consolidation_memory.client import MemoryClient
        c1 = MemoryClient(auto_consolidate=False)
        c2 = MemoryClient(auto_consolidate=False)
        c1.close()
        c2.close()


class TestClientStore:
    @patch("consolidation_memory.backends.encode_documents")
    def test_basic_store(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        result = client.store("test content", content_type="fact", tags=["python"])
        assert result.status == "stored"
        assert result.id is not None
        assert result.content_type == "fact"
        assert result.tags == ["python"]

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_surprise_clamping(self, mock_embed):
        from consolidation_memory.database import ensure_schema, get_episode
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        # Surprise > 1.0 should clamp to 1.0
        r1 = client.store("high surprise", surprise=5.0)
        ep = get_episode(r1.id)
        assert ep["surprise_score"] == 1.0

        # Negative should clamp to 0.0
        vec2 = _make_normalized_vec(seed=99)
        mock_embed.return_value = vec2.reshape(1, -1)
        r2 = client.store("low surprise", surprise=-1.0)
        ep2 = get_episode(r2.id)
        assert ep2["surprise_score"] == 0.0

        client.close()


class TestClientRecall:
    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_basic_recall(self, mock_embed_docs, mock_embed_query):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed_docs.return_value = vec.reshape(1, -1)
        mock_embed_query.return_value = vec.reshape(1, -1)

        client.store("recall test content", content_type="fact")

        result = client.recall("recall test", n_results=5, include_knowledge=False)
        assert result.total_episodes == 1
        assert len(result.episodes) >= 1
        assert result.episodes[0]["content"] == "recall test content"

        client.close()

    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_empty_recall(self, mock_embed_docs, mock_embed_query):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed_query.return_value = vec.reshape(1, -1)

        result = client.recall("nothing here")
        assert result.total_episodes == 0
        assert len(result.episodes) == 0

        client.close()


class TestClientForget:
    @patch("consolidation_memory.backends.encode_documents")
    def test_forget_existing(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        stored = client.store("forgettable content")
        result = client.forget(stored.id)
        assert result.status == "forgotten"
        assert result.id == stored.id

        client.close()

    def test_forget_nonexistent(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        result = client.forget("nonexistent-uuid")
        assert result.status == "not_found"

        client.close()


class TestClientStatus:
    @patch("consolidation_memory.backends.encode_documents")
    def test_status_counts(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client.store("status test episode")

        status = client.status()
        assert status.episodic_buffer["total"] == 1
        assert status.faiss_index_size == 1
        assert status.version == "0.1.0"
        assert status.embedding_backend != ""

        client.close()


class TestClientExport:
    @patch("consolidation_memory.backends.encode_documents")
    def test_export_round_trip(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client.store("export test", content_type="fact", tags=["test"])

        result = client.export()
        assert result.status == "exported"
        assert result.episodes == 1

        from pathlib import Path
        export_data = json.loads(Path(result.path).read_text(encoding="utf-8"))
        assert export_data["stats"]["episode_count"] == 1

        client.close()


class TestClientConsolidate:
    def test_consolidate_lock(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        # Acquire lock externally
        client._consolidation_lock.acquire()

        result = client.consolidate()
        assert result["status"] == "already_running"

        client._consolidation_lock.release()
        client.close()
