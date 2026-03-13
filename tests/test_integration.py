"""Integration tests exercising the full MemoryClient flow.

These tests mock the embedding backend but exercise the real database,
vector store, and client methods end-to-end.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from consolidation_memory.client import MemoryClient
from consolidation_memory.database import ensure_schema, fts_search, insert_episode, search_episodes
from tests.helpers import mock_encode as _mock_encode


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

    def test_search_episodes_tag_filter_returns_full_limit(self, client):
        """Tag filtering should not reduce result count below limit.

        Regression test: previously, SQL LIMIT was applied before Python
        tag filtering, so requesting limit=10 with tags could return fewer
        results than available.
        """
        # Store 10 episodes with tag "a" and 10 with tag "b"
        for i in range(10):
            client.store(f"episode a{i}", content_type="fact", tags=["a"])
        for i in range(10):
            client.store(f"episode b{i}", content_type="fact", tags=["b"])
        results = search_episodes(tags=["a"], limit=10)
        assert len(results) == 10
        for r in results:
            ep_tags = json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"]
            assert "a" in ep_tags

    def test_search_episodes_tag_filter_paginates_past_newer_non_matches(self, client):
        """Tagged matches should be found even when many newer rows do not match tags."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(12):
            ts = (base + timedelta(minutes=i)).isoformat()
            insert_episode(
                content=f"older tagged {i}",
                content_type="fact",
                tags=["target"],
                created_at=ts,
                updated_at=ts,
            )

        for i in range(120):
            ts = (base + timedelta(days=1, minutes=i)).isoformat()
            insert_episode(
                content=f"newer non-match {i}",
                content_type="fact",
                tags=["noise"],
                created_at=ts,
                updated_at=ts,
            )

        results = search_episodes(tags=["target"], limit=10)
        assert len(results) == 10
        for row in results:
            ep_tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
            assert "target" in ep_tags


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
        from consolidation_memory import __version__
        assert result.version == __version__
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


# ── Hybrid Search Integration ────────────────────────────────────────────────

class TestHybridRecall:
    def test_specific_term_ranks_high(self, client):
        """Episode with a specific term should rank high when recalled by that term."""
        client.store("Fix the CORS bug in AuthService", content_type="solution", tags=["cors"])
        client.store("General auth token refresh logic", content_type="solution", tags=["auth"])
        client.store("Database migration for users table", content_type="fact", tags=["db"])

        result = client.recall("CORS AuthService")
        assert len(result.episodes) >= 1
        # CORS episode should be in the results (boosted by FTS5)
        contents = [ep["content"] for ep in result.episodes]
        assert any("CORS" in c for c in contents)

        # Also verify it's in the FTS index directly
        fts_results = fts_search("CORS AuthService")
        assert len(fts_results) >= 1


class TestClaimDriftEndToEnd:
    def test_store_consolidate_claim_retrieval_drift_recall(self, client, monkeypatch, tmp_data_dir):
        from consolidation_memory.database import (
            get_all_episodes,
            get_connection,
            insert_claim_event,
            insert_claim_sources,
            upsert_claim,
        )

        stored = client.store(
            "Deployment note: src/app.py sets APP_MODE=legacy until migration is done.",
            content_type="fact",
            tags=["deploy", "app"],
        )
        assert stored.status == "stored"

        claim_id = "integration-claim-app-mode"

        def _fake_run_consolidation(vector_store=None):
            del vector_store
            episodes = get_all_episodes(include_deleted=False)
            assert episodes, "expected at least one episode before consolidation"
            source_episode_id = episodes[0]["id"]

            upsert_claim(
                claim_id=claim_id,
                claim_type="fact",
                canonical_text="src/app.py sets APP_MODE=legacy",
                payload={"path": "src/app.py", "app_mode": "legacy"},
                status="active",
                confidence=0.9,
                valid_from="2026-01-01T00:00:00+00:00",
            )
            insert_claim_sources(
                claim_id,
                [{"source_episode_id": source_episode_id}],
            )
            insert_claim_event(claim_id, event_type="create", details={"source": "integration-test"})

            return {"status": "completed", "episodes_processed": 1}

        with patch("consolidation_memory.consolidation.run_consolidation", side_effect=_fake_run_consolidation):
            consolidation_result = client.consolidate()
        assert consolidation_result["status"] == "completed"

        pre_drift_claims = client.search_claims(query="APP_MODE legacy", claim_type="fact")
        assert any(claim["id"] == claim_id for claim in pre_drift_claims.claims)

        monkeypatch.setattr(
            "consolidation_memory.drift.get_changed_files",
            lambda base_ref=None, repo_path=None: ["src/app.py"],
        )
        drift_result = client.detect_drift(base_ref="origin/main", repo_path=tmp_data_dir)
        assert drift_result["impacted_claim_ids"] == [claim_id]
        assert drift_result["challenged_claim_ids"] == [claim_id]

        post_drift_claims = client.search_claims(query="APP_MODE legacy", claim_type="fact")
        assert all(claim["id"] != claim_id for claim in post_drift_claims.claims)

        recall_after_drift = client.recall("What mode is src/app.py using right now?")
        assert all(claim["id"] != claim_id for claim in recall_after_drift.claims)

        with get_connection() as conn:
            claim_row = conn.execute(
                "SELECT status FROM claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            drift_event = conn.execute(
                """SELECT 1
                   FROM claim_events
                   WHERE claim_id = ? AND event_type = 'code_drift_detected'
                   LIMIT 1""",
                (claim_id,),
            ).fetchone()

        assert claim_row is not None and claim_row["status"] == "challenged"
        assert drift_event is not None
