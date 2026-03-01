"""Tests for tag co-occurrence graph (Option D — intent motifs)."""

from unittest.mock import patch

import numpy as np

from consolidation_memory.database import (
    ensure_schema,
    get_cooccurring_tags,
    get_tag_pairs_in_set,
    update_tag_cooccurrence,
)

from helpers import make_normalized_vec as _make_normalized_vec


# ── Schema migration ──────────────────────────────────────────────────────────


class TestSchemaMigrationV9:
    def test_migration_creates_tag_cooccurrence_table(self, tmp_data_dir):
        ensure_schema()
        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_cooccurrence'"
            )
            assert cursor.fetchone() is not None

    def test_tag_cooccurrence_columns(self, tmp_data_dir):
        ensure_schema()
        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(tag_cooccurrence)")
            columns = {row["name"] for row in cursor.fetchall()}

        assert {"tag_a", "tag_b", "count", "last_seen"}.issubset(columns)


# ── update_tag_cooccurrence ──────────────────────────────────────────────────


class TestUpdateTagCooccurrence:
    def test_single_pair(self, tmp_data_dir):
        ensure_schema()
        update_tag_cooccurrence(["diet", "exercise"])

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tag_cooccurrence WHERE tag_a = ? AND tag_b = ?",
                ("diet", "exercise"),
            ).fetchone()
        assert row is not None
        assert row["count"] == 1

    def test_increments_on_repeat(self, tmp_data_dir):
        ensure_schema()
        update_tag_cooccurrence(["diet", "exercise"])
        update_tag_cooccurrence(["diet", "exercise"])
        update_tag_cooccurrence(["diet", "exercise"])

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            row = conn.execute(
                "SELECT count FROM tag_cooccurrence WHERE tag_a = ? AND tag_b = ?",
                ("diet", "exercise"),
            ).fetchone()
        assert row["count"] == 3

    def test_orders_tags_alphabetically(self, tmp_data_dir):
        """tag_a < tag_b invariant is maintained."""
        ensure_schema()
        # Pass in reverse order
        update_tag_cooccurrence(["exercise", "diet"])

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tag_cooccurrence WHERE tag_a = ? AND tag_b = ?",
                ("diet", "exercise"),
            ).fetchone()
        assert row is not None
        assert row["count"] == 1

    def test_multiple_pairs(self, tmp_data_dir):
        ensure_schema()
        update_tag_cooccurrence(["diet", "exercise", "weight"])

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM tag_cooccurrence").fetchall()
        # 3 tags -> 3 pairs: (diet, exercise), (diet, weight), (exercise, weight)
        assert len(rows) == 3

    def test_single_tag_is_noop(self, tmp_data_dir):
        ensure_schema()
        update_tag_cooccurrence(["diet"])

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM tag_cooccurrence").fetchall()
        assert len(rows) == 0

    def test_deduplicates_tags(self, tmp_data_dir):
        ensure_schema()
        update_tag_cooccurrence(["diet", "diet", "exercise"])

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM tag_cooccurrence").fetchall()
        # Only 1 unique pair: (diet, exercise)
        assert len(rows) == 1


# ── get_cooccurring_tags ─────────────────────────────────────────────────────


class TestGetCooccurringTags:
    def test_returns_cooccurring_tags(self, tmp_data_dir):
        ensure_schema()
        # Build co-occurrence: diet-exercise appears 3x, diet-weight 2x
        for _ in range(3):
            update_tag_cooccurrence(["diet", "exercise"])
        for _ in range(2):
            update_tag_cooccurrence(["diet", "weight"])

        result = get_cooccurring_tags(["diet"], min_count=2)
        assert "exercise" in result
        assert result["exercise"] == 3
        assert "weight" in result
        assert result["weight"] == 2

    def test_excludes_input_tags(self, tmp_data_dir):
        ensure_schema()
        for _ in range(3):
            update_tag_cooccurrence(["diet", "exercise"])

        result = get_cooccurring_tags(["diet"], min_count=2)
        assert "diet" not in result

    def test_respects_min_count(self, tmp_data_dir):
        ensure_schema()
        update_tag_cooccurrence(["diet", "exercise"])  # count=1
        for _ in range(3):
            update_tag_cooccurrence(["diet", "weight"])  # count=3

        result = get_cooccurring_tags(["diet"], min_count=2)
        assert "exercise" not in result
        assert "weight" in result

    def test_empty_tags(self, tmp_data_dir):
        ensure_schema()
        assert get_cooccurring_tags([]) == {}


# ── get_tag_pairs_in_set ─────────────────────────────────────────────────────


class TestGetTagPairsInSet:
    def test_finds_pairs_in_set(self, tmp_data_dir):
        ensure_schema()
        for _ in range(3):
            update_tag_cooccurrence(["diet", "exercise"])
        update_tag_cooccurrence(["diet", "python"])  # count=1, below min

        pairs = get_tag_pairs_in_set(["diet", "exercise", "python"], min_count=2)
        # Only (diet, exercise) with count >= 2
        assert len(pairs) == 1
        assert pairs[0][0] == "diet"
        assert pairs[0][1] == "exercise"
        assert pairs[0][2] == 3

    def test_returns_empty_for_single_tag(self, tmp_data_dir):
        ensure_schema()
        assert get_tag_pairs_in_set(["diet"]) == []

    def test_returns_empty_when_no_cooccurrence(self, tmp_data_dir):
        ensure_schema()
        pairs = get_tag_pairs_in_set(["diet", "exercise"], min_count=2)
        assert pairs == []


# ── Co-occurrence boost during recall ────────────────────────────────────────


class TestCooccurrenceBoostInRecall:
    """Test that recall boosts episodes with co-occurring tags."""

    def _make_vec(self, seed=42):
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def test_boost_applied_to_cooccurring_episodes(self, tmp_data_dir):
        """Episodes whose tags participate in co-occurrence connections
        should get a ~10% boost compared to episodes without."""
        ensure_schema()

        # Build co-occurrence: diet and exercise frequently appear together
        for _ in range(5):
            update_tag_cooccurrence(["diet", "exercise"])

        from consolidation_memory.context_assembler import _apply_cooccurrence_boost

        # Create scored candidates
        ep_diet = {"id": "1", "tags": '["diet"]', "surprise_score": 0.5,
                   "created_at": "2026-01-01", "access_count": 0}
        ep_exercise = {"id": "2", "tags": '["exercise"]', "surprise_score": 0.5,
                       "created_at": "2026-01-01", "access_count": 0}
        ep_python = {"id": "3", "tags": '["python"]', "surprise_score": 0.5,
                     "created_at": "2026-01-01", "access_count": 0}

        scored = [
            (ep_diet, 0.8, 0.8),      # has "diet" - co-occurs with "exercise"
            (ep_exercise, 0.7, 0.7),   # has "exercise" - co-occurs with "diet"
            (ep_python, 0.75, 0.75),   # has "python" - no co-occurrence
        ]

        boosted = _apply_cooccurrence_boost(scored)

        # diet and exercise should be boosted (they co-occur)
        assert boosted[0][1] == 0.8 * 1.10   # diet boosted
        assert boosted[1][1] == 0.7 * 1.10   # exercise boosted
        assert boosted[2][1] == 0.75          # python NOT boosted

    def test_no_boost_without_cooccurrence(self, tmp_data_dir):
        """Without co-occurrence data, scores should be unchanged."""
        ensure_schema()

        from consolidation_memory.context_assembler import _apply_cooccurrence_boost

        ep_a = {"id": "1", "tags": '["a"]'}
        ep_b = {"id": "2", "tags": '["b"]'}

        scored = [(ep_a, 0.8, 0.8), (ep_b, 0.7, 0.7)]
        boosted = _apply_cooccurrence_boost(scored)

        assert boosted[0][1] == 0.8
        assert boosted[1][1] == 0.7

    def test_no_boost_with_single_candidate(self, tmp_data_dir):
        """Single candidate should not be boosted (need 2+ for cluster)."""
        ensure_schema()
        for _ in range(5):
            update_tag_cooccurrence(["diet", "exercise"])

        from consolidation_memory.context_assembler import _apply_cooccurrence_boost

        ep = {"id": "1", "tags": '["diet"]'}
        # _apply_cooccurrence_boost only called when len(scored) >= 2
        # but let's test with 2 where only one has co-occurring tags
        ep2 = {"id": "2", "tags": '[]'}
        scored = [(ep, 0.8, 0.8), (ep2, 0.5, 0.5)]
        boosted = _apply_cooccurrence_boost(scored)

        # Only "diet" is in co-occurrence, but there's no matching partner
        # because ep2 has no tags. So no boost.
        assert boosted[0][1] == 0.8
        assert boosted[1][1] == 0.5


class TestCooccurrenceInStore:
    """Test that store() updates the co-occurrence table."""

    def test_store_updates_cooccurrence(self, tmp_data_dir):
        """Storing an episode with multiple tags should update co-occurrence."""
        ensure_schema()

        def mock_encode(texts):
            return np.array([_make_normalized_vec(seed=i) for i, _ in enumerate(texts)])

        with patch("consolidation_memory.backends.encode_documents", mock_encode):
            from consolidation_memory.client import MemoryClient

            client = MemoryClient(auto_consolidate=False)
            try:
                client.store("healthy meal plan", tags=["diet", "exercise", "health"])
            finally:
                client.close()

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM tag_cooccurrence").fetchall()

        # 3 tags -> 3 pairs
        assert len(rows) == 3
        tags_in_pairs = set()
        for row in rows:
            tags_in_pairs.add(row["tag_a"])
            tags_in_pairs.add(row["tag_b"])
        assert tags_in_pairs == {"diet", "exercise", "health"}

    def test_store_with_single_tag_no_cooccurrence(self, tmp_data_dir):
        """Single tag should not create co-occurrence entries."""
        ensure_schema()

        def mock_encode(texts):
            return np.array([_make_normalized_vec(seed=42) for _ in texts])

        with patch("consolidation_memory.backends.encode_documents", mock_encode):
            from consolidation_memory.client import MemoryClient

            client = MemoryClient(auto_consolidate=False)
            try:
                client.store("python tips", tags=["python"])
            finally:
                client.close()

        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM tag_cooccurrence").fetchall()
        assert len(rows) == 0
