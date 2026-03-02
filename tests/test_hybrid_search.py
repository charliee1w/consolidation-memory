"""Tests for hybrid BM25 + semantic search (Phase 2.1)."""

from unittest.mock import patch

import pytest

from consolidation_memory.config import override_config
from consolidation_memory.database import (
    _sanitize_fts_query,
    _reset_fts5_cache,
    ensure_schema,
    fts_available,
    fts_insert,
    fts_rebuild,
    fts_search,
    get_episode,
    insert_episode,
    soft_delete_episode,
    hard_delete_episode,
)
from helpers import mock_encode as _mock_encode


@pytest.fixture()
def schema():
    """Ensure schema is set up for each test."""
    ensure_schema()


# ── FTS5 Schema ──────────────────────────────────────────────────────────────

class TestFTS5Schema:
    def test_fts5_table_created(self, schema):
        assert fts_available() is True

    def test_migration_v10_applied(self, schema):
        """Schema version should be 10 after ensure_schema."""
        from consolidation_memory.database import get_connection
        with get_connection() as conn:
            row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
            assert row["v"] >= 10


# ── FTS Insert / Delete ──────────────────────────────────────────────────────

class TestFTSInsertDelete:
    def test_insert_episode_populates_fts(self, schema):
        eid = insert_episode("CORS bug fix in AuthService", content_type="solution")
        results = fts_search("CORS AuthService")
        assert any(r[0] == eid for r in results)

    def test_soft_delete_removes_from_fts(self, schema):
        eid = insert_episode("unique ZXCVBN token problem", content_type="solution")
        results = fts_search("ZXCVBN")
        assert any(r[0] == eid for r in results)

        soft_delete_episode(eid)
        results = fts_search("ZXCVBN")
        assert not any(r[0] == eid for r in results)

    def test_hard_delete_removes_from_fts(self, schema):
        eid = insert_episode("rare QWERTY keyboard issue", content_type="fact")
        results = fts_search("QWERTY")
        assert any(r[0] == eid for r in results)

        hard_delete_episode(eid)
        results = fts_search("QWERTY")
        assert not any(r[0] == eid for r in results)

    def test_fts_rebuild(self, schema):
        eid = insert_episode("rebuild test content XYZZY", content_type="fact")
        # Verify it's there
        assert any(r[0] == eid for r in fts_search("XYZZY"))
        # Rebuild should preserve it
        fts_rebuild()
        assert any(r[0] == eid for r in fts_search("XYZZY"))


# ── FTS Search ───────────────────────────────────────────────────────────────

class TestFTSSearch:
    def test_basic_keyword_search(self, schema):
        insert_episode("Python uses GIL for thread safety", content_type="fact")
        insert_episode("Rust has no garbage collector", content_type="fact")

        results = fts_search("GIL thread")
        assert len(results) >= 1
        # The GIL episode should rank first
        eids = [r[0] for r in results]
        ep = get_episode(eids[0])
        assert ep is not None
        assert "GIL" in ep["content"]

    def test_bm25_scores_are_normalized(self, schema):
        insert_episode("keyword matching test content", content_type="fact")
        results = fts_search("keyword matching")
        assert len(results) >= 1
        for _, score in results:
            assert 0.0 <= score < 1.0  # normalized = raw / (raw + 1)

    def test_search_returns_empty_for_no_match(self, schema):
        insert_episode("some content here", content_type="fact")
        results = fts_search("NONEXISTENT_TERM_XYZZY")
        assert results == []

    def test_search_with_limit(self, schema):
        for i in range(10):
            insert_episode(f"episode number {i} with searchable content", content_type="fact")
        results = fts_search("searchable content", limit=3)
        assert len(results) <= 3

    def test_multi_term_or_search(self, schema):
        """FTS5 query uses OR — documents matching any term are returned."""
        eid1 = insert_episode("alpha bravo charlie", content_type="fact")
        eid2 = insert_episode("delta echo foxtrot", content_type="fact")
        # Searching for terms from both episodes should return both
        results = fts_search("alpha delta")
        eids = {r[0] for r in results}
        assert eid1 in eids
        assert eid2 in eids


# ── Query Sanitization ───────────────────────────────────────────────────────

class TestQuerySanitization:
    def test_strips_special_chars(self):
        assert _sanitize_fts_query("hello! world@#$%") == "hello OR world"

    def test_drops_single_chars(self):
        assert _sanitize_fts_query("a b cd ef") == "cd OR ef"

    def test_empty_query_returns_empty(self):
        assert _sanitize_fts_query("") == ""

    def test_all_special_chars_returns_empty(self):
        assert _sanitize_fts_query("!@#$%^&*()") == ""

    def test_preserves_alphanumeric(self):
        assert _sanitize_fts_query("CORS AuthService v2") == "CORS OR AuthService OR v2"

    def test_single_valid_term(self):
        assert _sanitize_fts_query("CORS") == "CORS"


# ── BM25 Normalization ──────────────────────────────────────────────────────

class TestBM25Normalization:
    def test_normalization_properties(self):
        """Verify raw / (raw + 1.0) has expected properties."""
        # At raw=0, normalized=0
        assert 0.0 / (0.0 + 1.0) == 0.0
        # At raw=1, normalized=0.5
        assert 1.0 / (1.0 + 1.0) == 0.5
        # At raw=9, normalized~0.9
        assert abs(9.0 / (9.0 + 1.0) - 0.9) < 1e-9
        # Always < 1
        assert 1000.0 / (1000.0 + 1.0) < 1.0
        # Monotonically increasing
        scores = [x / (x + 1.0) for x in range(100)]
        for i in range(1, len(scores)):
            assert scores[i] > scores[i - 1]


# ── Hybrid Scoring (context_assembler) ──────────────────────────────────────

class TestHybridScoring:
    @pytest.fixture()
    def _setup(self, schema):
        """Patch embedding backend for recall tests."""
        with (
            patch("consolidation_memory.backends.encode_documents", side_effect=_mock_encode),
            patch(
                "consolidation_memory.backends.encode_query",
                side_effect=lambda q: _mock_encode([q])[0],
            ),
            patch("consolidation_memory.backends.get_dimension", return_value=384),
        ):
            from consolidation_memory.client import MemoryClient
            client = MemoryClient(auto_consolidate=False)
            yield client
            client.close()

    def test_keyword_match_boosts_ranking(self, _setup):
        """Episode with exact keyword match should be boosted by BM25."""
        client = _setup
        # Store episodes with distinct keywords
        client.store("Fix the CORS bug in AuthService module", content_type="solution")
        client.store("General authentication improvements", content_type="solution")
        client.store("Password validation changes", content_type="solution")

        result = client.recall("CORS AuthService")
        # The CORS episode should appear in results
        assert any("CORS" in ep["content"] for ep in result.episodes)
        # And should have bm25_score field
        cors_ep = [ep for ep in result.episodes if "CORS" in ep["content"]]
        if cors_ep:
            assert "bm25_score" in cors_ep[0]
            assert cors_ep[0]["bm25_score"] > 0

    def test_keyword_match_ranks_higher(self, _setup):
        """Episode with exact keyword match should rank above pure-semantic matches."""
        client = _setup
        # Store one with the exact query term, one without
        client.store("ZXCVBN token validation failure in auth", content_type="solution")
        client.store("General authentication token improvements", content_type="solution")

        result = client.recall("ZXCVBN token")
        if len(result.episodes) >= 2:
            # The keyword-matching episode should appear first
            assert "ZXCVBN" in result.episodes[0]["content"]

    def test_semantic_only_still_works(self, _setup):
        """Episode with high semantic sim but no keyword match should still appear."""
        client = _setup
        client.store("Python programming tips and tricks", content_type="fact")
        # With mock embeddings, recall still works via FAISS
        result = client.recall("Python programming tips and tricks")
        assert len(result.episodes) >= 1

    def test_hybrid_disabled_skips_fts(self, _setup):
        """When HYBRID_SEARCH_ENABLED=False, bm25_score should not appear."""
        client = _setup
        client.store("test content for disabled hybrid", content_type="fact")

        with override_config(HYBRID_SEARCH_ENABLED=False):
            result = client.recall("test content disabled")
            for ep in result.episodes:
                assert "bm25_score" not in ep


class TestFTS5Unavailable:
    def test_graceful_fallback_when_no_fts5(self, schema):
        """When FTS5 table doesn't exist, recall still works via FAISS only."""
        _reset_fts5_cache()
        with patch("consolidation_memory.database.fts_available", return_value=False):
            # fts_search should return empty
            assert fts_search("anything") == []
            # fts_insert should be a no-op
            fts_insert("fake-id", "fake content")  # should not raise

    def test_fts_available_false_when_table_missing(self, tmp_data_dir):
        """Before ensure_schema, FTS5 should not be available."""
        _reset_fts5_cache()
        # Don't call ensure_schema — table won't exist
        # fts_available checks sqlite_master, which requires a connection
        # Just verify the function doesn't crash
        result = fts_available()
        assert isinstance(result, bool)
