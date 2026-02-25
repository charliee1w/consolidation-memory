"""Integration tests exercising the full MemoryClient flow.

These tests mock the embedding backend but exercise the real database,
vector store, and client methods end-to-end.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from consolidation_memory.client import MemoryClient
from consolidation_memory.database import ensure_schema, search_episodes


def _make_vec(dim=384, seed=None):
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _mock_encode(texts):
    """Deterministic mock: hash text to seed a reproducible vector."""
    vecs = []
    for t in texts:
        seed = hash(t) % (2**31)
        vecs.append(_make_vec(seed=seed))
    return np.array(vecs, dtype=np.float32)


@pytest.fixture()
def client():
    """Create a MemoryClient with mocked embedding backend."""
    ensure_schema()
    with (
        patch("consolidation_memory.backends.encode_documents", side_effect=_mock_encode),
        patch("consolidation_memory.backends.encode_query", side_effect=lambda q: _mock_encode([q])[0]),
        patch("consolidation_memory.backends.get_dimension", return_value=384),
    ):
        c = MemoryClient(auto_consolidate=False)
        yield c
        c.close()


# ── Store → Recall round-trip ────────────────────────────────────────────────

class TestStoreRecallRoundTrip:
    def test_store_and_recall(self, client):
        result = client.store("Python uses indentation for blocks", content_type="fact", tags=["python"])
        assert result.status == "stored"
        assert result.id is not None

        recall = client.recall("Python syntax")
        assert len(recall.episodes) >= 1
        assert any("indentation" in ep["content"] for ep in recall.episodes)
        assert recall.total_episodes >= 1

    def test_store_returns_duplicate(self, client):
        client.store("fact about deduplication", content_type="fact")
        result = client.store("fact about deduplication", content_type="fact")
        assert result.status == "duplicate_detected"

    def test_recall_with_filters(self, client):
        client.store("fix the bug by restarting", content_type="solution", tags=["debug"])
        client.store("user prefers dark mode", content_type="preference", tags=["ui"])

        # Filter by content_type
        result = client.recall("preferences", content_types=["preference"])
        for ep in result.episodes:
            assert ep["content_type"] == "preference"

        # Filter by tag
        result = client.recall("debug", tags=["debug"])
        for ep in result.episodes:
            assert "debug" in ep["tags"]

    def test_recall_empty_db(self, client):
        result = client.recall("anything")
        assert result.episodes == []
        assert result.total_episodes == 0


# ── Batch Store ──────────────────────────────────────────────────────────────

class TestBatchStore:
    def test_batch_store(self, client):
        episodes = [
            {"content": "batch item one", "content_type": "fact", "tags": ["test"]},
            {"content": "batch item two", "content_type": "solution"},
            {"content": "batch item three"},
        ]
        result = client.store_batch(episodes)
        assert result.status == "stored"
        assert result.stored == 3
        assert result.duplicates == 0
        assert len(result.results) == 3

    def test_batch_store_empty(self, client):
        result = client.store_batch([])
        assert result.stored == 0

    def test_batch_store_dedup(self, client):
        client.store("already exists", content_type="fact")
        result = client.store_batch([{"content": "already exists"}])
        assert result.duplicates == 1
        assert result.stored == 0


# ── Search (keyword/metadata) ───────────────────────────────────────────────

class TestSearch:
    def test_search_by_keyword(self, client):
        client.store("Python uses GIL for thread safety", content_type="fact", tags=["python"])
        client.store("Rust has no garbage collector", content_type="fact", tags=["rust"])

        result = client.search(query="GIL")
        assert result.total_matches >= 1
        assert all("GIL" in ep["content"] for ep in result.episodes)

    def test_search_by_content_type(self, client):
        client.store("a solution to the problem", content_type="solution")
        client.store("a random exchange", content_type="exchange")

        result = client.search(content_types=["solution"])
        assert result.total_matches >= 1
        for ep in result.episodes:
            assert ep["content_type"] == "solution"

    def test_search_by_tag(self, client):
        client.store("tagged episode", content_type="fact", tags=["special"])
        client.store("untagged episode", content_type="fact")

        result = client.search(tags=["special"])
        assert result.total_matches >= 1
        for ep in result.episodes:
            assert "special" in ep["tags"]

    def test_search_no_filters_returns_all(self, client):
        client.store("episode one", content_type="fact")
        client.store("episode two", content_type="fact")
        result = client.search()
        assert result.total_matches >= 2

    def test_search_with_date_filter(self, client):
        client.store("old episode", content_type="fact")
        result = client.search(after="2099-01-01")
        assert result.total_matches == 0

    def test_search_with_limit(self, client):
        for i in range(5):
            client.store(f"episode number {i}", content_type="fact")
        result = client.search(limit=2)
        assert len(result.episodes) == 2


# ── Search database function ────────────────────────────────────────────────

class TestSearchDB:
    def test_search_episodes_db(self, client):
        client.store("database level test", content_type="fact", tags=["db"])
        results = search_episodes(query="database level")
        assert len(results) >= 1

    def test_search_episodes_content_type_filter(self, client):
        client.store("solution content", content_type="solution")
        client.store("fact content", content_type="fact")
        results = search_episodes(content_types=["solution"])
        assert all(r["content_type"] == "solution" for r in results)


# ── Forget ───────────────────────────────────────────────────────────────────

class TestForget:
    def test_forget_episode(self, client):
        stored = client.store("to be forgotten", content_type="fact")
        result = client.forget(stored.id)
        assert result.status == "forgotten"

        # Should not appear in search
        search = client.search(query="forgotten")
        assert all(stored.id != ep["id"] for ep in search.episodes)

    def test_forget_nonexistent(self, client):
        result = client.forget("nonexistent-id")
        assert result.status == "not_found"


# ── Status ───────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_basic(self, client):
        result = client.status()
        assert result.version == "0.2.0"
        assert "status" in result.health
        assert isinstance(result.consolidation_quality, dict)

    def test_status_after_store(self, client):
        client.store("test episode", content_type="fact")
        result = client.status()
        assert result.episodic_buffer["total"] >= 1
        assert result.faiss_index_size >= 1


# ── Export ───────────────────────────────────────────────────────────────────

class TestExport:
    def test_export_basic(self, client):
        client.store("export test", content_type="fact")
        result = client.export()
        assert result.status == "exported"
        assert result.episodes >= 1
        assert result.path  # non-empty path

        # Verify JSON is valid
        with open(result.path) as f:
            data = json.load(f)
        assert "episodes" in data
        assert "knowledge_topics" in data
        assert data["stats"]["episode_count"] >= 1
