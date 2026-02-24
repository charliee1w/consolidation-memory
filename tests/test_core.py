"""Tests for consolidation memory core components.

Run with: python -m pytest tests/ -v
No external embedding server required — embedding backend is mocked.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────
# Shared tmp_data_dir fixture is in conftest.py (autouse=True)


def _make_normalized_vec(dim=384, seed=None):
    """Create a random L2-normalized vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_normalized_batch(n, dim=384, seed=None):
    """Create n random L2-normalized vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ── Database tests ────────────────────────────────────────────────────────────

class TestDatabase:
    def test_schema_creation(self):
        from consolidation_memory.database import ensure_schema, get_stats
        ensure_schema()
        stats = get_stats()
        assert stats["episodic_buffer"]["total"] == 0
        assert stats["knowledge_base"]["total_topics"] == 0

    def test_episode_crud(self):
        from consolidation_memory.database import ensure_schema, insert_episode, get_episode, soft_delete_episode
        ensure_schema()

        ep_id = insert_episode(
            content="Test memory content",
            content_type="fact",
            tags=["test"],
            surprise_score=0.7,
        )
        assert ep_id is not None

        ep = get_episode(ep_id)
        assert ep is not None
        assert ep["content"] == "Test memory content"
        assert ep["content_type"] == "fact"
        assert ep["surprise_score"] == 0.7
        assert json.loads(ep["tags"]) == ["test"]

        deleted = soft_delete_episode(ep_id)
        assert deleted is True
        assert get_episode(ep_id) is None

    def test_access_increment(self):
        from consolidation_memory.database import ensure_schema, insert_episode, get_episode, increment_access
        ensure_schema()

        ep_id = insert_episode(content="Access test", tags=["test"])
        assert get_episode(ep_id)["access_count"] == 0

        increment_access([ep_id])
        assert get_episode(ep_id)["access_count"] == 1

        increment_access([ep_id])
        assert get_episode(ep_id)["access_count"] == 2

    def test_consolidation_tracking(self):
        from consolidation_memory.database import (
            ensure_schema, insert_episode, mark_consolidated,
            get_unconsolidated_episodes, get_episode,
        )
        ensure_schema()

        ep_id = insert_episode(content="Will be consolidated", tags=[])
        assert len(get_unconsolidated_episodes()) == 1

        mark_consolidated([ep_id], "test_topic.md")
        assert len(get_unconsolidated_episodes()) == 0
        assert get_episode(ep_id)["consolidated"] == 1

    def test_knowledge_topic_upsert(self):
        from consolidation_memory.database import ensure_schema, upsert_knowledge_topic, get_all_knowledge_topics
        ensure_schema()

        upsert_knowledge_topic(
            filename="test.md",
            title="Test Topic",
            summary="A test topic",
            source_episodes=["ep1"],
            fact_count=3,
            confidence=0.85,
        )
        topics = get_all_knowledge_topics()
        assert len(topics) == 1
        assert topics[0]["title"] == "Test Topic"

        upsert_knowledge_topic(
            filename="test.md",
            title="Updated Topic",
            summary="Updated summary",
            source_episodes=["ep2"],
        )
        topics = get_all_knowledge_topics()
        assert len(topics) == 1
        assert topics[0]["title"] == "Updated Topic"
        sources = json.loads(topics[0]["source_episodes"])
        assert set(sources) == {"ep1", "ep2"}

    def test_schema_version(self):
        from consolidation_memory.database import ensure_schema, get_connection, CURRENT_SCHEMA_VERSION
        ensure_schema()
        with get_connection() as conn:
            row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
        assert row["v"] == CURRENT_SCHEMA_VERSION


# ── Vector store tests ────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()
        assert vs.size == 0

        vec = _make_normalized_vec(seed=42)
        vs.add("ep-1", vec)
        assert vs.size == 1

        results = vs.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "ep-1"
        assert results[0][1] > 0.99

    def test_add_batch(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(5, seed=42)
        ids = [f"ep-{i}" for i in range(5)]
        vs.add_batch(ids, vecs)
        assert vs.size == 5

    def test_remove(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vec = _make_normalized_vec(seed=42)
        vs.add("ep-remove", vec)
        assert vs.size == 1

        removed = vs.remove("ep-remove")
        assert removed is True
        assert vs.size == 0

    def test_remove_batch(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(4, seed=42)
        ids = ["a", "b", "c", "d"]
        vs.add_batch(ids, vecs)
        assert vs.size == 4

        removed = vs.remove_batch(["b", "d"])
        assert removed == 2
        assert vs.size == 2

        results = vs.search(vecs[0], k=10)
        found_ids = [r[0] for r in results]
        assert "a" in found_ids
        assert "c" in found_ids
        assert "b" not in found_ids

    def test_reconstruct_batch(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(3, seed=42)
        vs.add_batch(["x", "y", "z"], vecs)

        result = vs.reconstruct_batch(["y", "missing", "x"])
        assert result is not None
        found_ids, found_vecs = result
        assert set(found_ids) == {"x", "y"}
        assert found_vecs.shape == (2, 384)

    def test_persistence(self):
        from consolidation_memory.vector_store import VectorStore
        vs1 = VectorStore()
        vec = _make_normalized_vec(seed=42)
        vs1.add("persist-test", vec)

        vs2 = VectorStore()
        assert vs2.size == 1
        results = vs2.search(vec, k=1)
        assert results[0][0] == "persist-test"

    def test_integrity_check_mismatch(self):
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.config import FAISS_INDEX_PATH, FAISS_ID_MAP_PATH, EMBEDDING_DIMENSION
        import faiss

        idx = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        vecs = np.random.randn(2, EMBEDDING_DIMENSION).astype(np.float32)
        idx.add(vecs)
        faiss.write_index(idx, str(FAISS_INDEX_PATH))
        with open(FAISS_ID_MAP_PATH, "w") as f:
            json.dump(["only-one-id"], f)

        vs = VectorStore()
        assert vs.size == 0


# ── Tombstone tests ──────────────────────────────────────────────────────────

class TestTombstones:
    def test_tombstone_excludes_from_search(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(3, seed=42)
        vs.add_batch(["a", "b", "c"], vecs)
        vs.remove("b")

        results = vs.search(vecs[1], k=10)
        found_ids = [r[0] for r in results]
        assert "b" not in found_ids
        assert vs.size == 2

    def test_tombstone_persistence(self):
        from consolidation_memory.vector_store import VectorStore
        vs1 = VectorStore()

        vec = _make_normalized_vec(seed=42)
        vs1.add("tombstone-persist", vec)
        vs1.remove("tombstone-persist")

        vs2 = VectorStore()
        assert vs2.size == 0
        assert "tombstone-persist" in vs2._tombstones

    def test_compact_removes_tombstoned(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(5, seed=42)
        ids = [f"ep-{i}" for i in range(5)]
        vs.add_batch(ids, vecs)
        vs.remove_batch(["ep-1", "ep-3"])

        assert vs._index.ntotal == 5
        assert vs.size == 3

        removed = vs.compact()
        assert removed == 2
        assert vs._index.ntotal == 3
        assert vs.size == 3
        assert len(vs._tombstones) == 0

    def test_search_compensation(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(10, seed=42)
        ids = [f"ep-{i}" for i in range(10)]
        vs.add_batch(ids, vecs)

        vs.remove_batch([f"ep-{i}" for i in range(5)])

        results = vs.search(vecs[5], k=5)
        assert len(results) == 5
        for rid, _ in results:
            assert rid not in {f"ep-{i}" for i in range(5)}

    def test_reconstruct_excludes_tombstoned(self):
        from consolidation_memory.vector_store import VectorStore
        vs = VectorStore()

        vecs = _make_normalized_batch(3, seed=42)
        vs.add_batch(["x", "y", "z"], vecs)
        vs.remove("y")

        result = vs.reconstruct_batch(["x", "y", "z"])
        assert result is not None
        found_ids, _ = result
        assert "y" not in found_ids
        assert set(found_ids) == {"x", "z"}


# ── Reload signal tests ──────────────────────────────────────────────────────

class TestReloadSignal:
    def test_signal_triggers_reload(self):
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.config import FAISS_RELOAD_SIGNAL
        vs = VectorStore()

        vec = _make_normalized_vec(seed=42)
        vs.add("before-reload", vec)

        vs._last_load_time = time.time() - 100
        FAISS_RELOAD_SIGNAL.write_text(str(time.time()), encoding="utf-8")

        reloaded = vs.reload_if_stale()
        assert reloaded is True

    def test_old_signal_ignored(self):
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.config import FAISS_RELOAD_SIGNAL
        vs = VectorStore()

        FAISS_RELOAD_SIGNAL.write_text(str(time.time() - 200), encoding="utf-8")
        vs._last_load_time = time.time()

        reloaded = vs.reload_if_stale()
        assert reloaded is False

    def test_signal_file_written(self):
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.config import FAISS_RELOAD_SIGNAL

        assert not FAISS_RELOAD_SIGNAL.exists()
        VectorStore.signal_reload()
        assert FAISS_RELOAD_SIGNAL.exists()


# ── Knowledge versioning tests ───────────────────────────────────────────────

class TestVersioning:
    def test_creates_backup(self, tmp_data_dir):
        from consolidation_memory.consolidation import _version_knowledge_file
        from consolidation_memory.config import KNOWLEDGE_DIR, KNOWLEDGE_VERSIONS_DIR

        filepath = KNOWLEDGE_DIR / "test_topic.md"
        filepath.write_text("# Original Content", encoding="utf-8")

        _version_knowledge_file(filepath)

        versions = list(KNOWLEDGE_VERSIONS_DIR.glob("test_topic.*.md"))
        assert len(versions) == 1
        assert versions[0].read_text(encoding="utf-8") == "# Original Content"

    def test_preserves_content(self, tmp_data_dir):
        from consolidation_memory.consolidation import _version_knowledge_file
        from consolidation_memory.config import KNOWLEDGE_DIR, KNOWLEDGE_VERSIONS_DIR

        filepath = KNOWLEDGE_DIR / "preserve.md"
        original = "---\ntitle: Test\n---\n\n## Facts\n- Important fact"
        filepath.write_text(original, encoding="utf-8")

        _version_knowledge_file(filepath)

        versions = list(KNOWLEDGE_VERSIONS_DIR.glob("preserve.*.md"))
        assert versions[0].read_text(encoding="utf-8") == original

    def test_prunes_old_versions(self, tmp_data_dir):
        from consolidation_memory.consolidation import _version_knowledge_file
        from consolidation_memory.config import KNOWLEDGE_DIR, KNOWLEDGE_VERSIONS_DIR

        filepath = KNOWLEDGE_DIR / "pruned.md"

        for i in range(7):
            filepath.write_text(f"Version {i}", encoding="utf-8")
            _version_knowledge_file(filepath)
            time.sleep(0.05)

        versions = list(KNOWLEDGE_VERSIONS_DIR.glob("pruned.*.md"))
        assert len(versions) <= 5

    def test_noop_new_file(self, tmp_data_dir):
        from consolidation_memory.consolidation import _version_knowledge_file
        from consolidation_memory.config import KNOWLEDGE_DIR, KNOWLEDGE_VERSIONS_DIR

        filepath = KNOWLEDGE_DIR / "nonexistent.md"
        _version_knowledge_file(filepath)

        versions = list(KNOWLEDGE_VERSIONS_DIR.glob("nonexistent.*.md"))
        assert len(versions) == 0


# ── Topic embedding cache tests ──────────────────────────────────────────────

class TestTopicCache:
    def test_cache_invalidation(self):
        from consolidation_memory.topic_cache import _cache, invalidate

        _cache["topic_count"] = 5
        _cache["topics"] = [{"title": "Cached", "summary": "cached"}]
        _cache["vecs"] = np.zeros((1, 384), dtype=np.float32)

        invalidate()
        assert _cache["topic_count"] == -1

    @patch("consolidation_memory.topic_cache.encode_documents")
    def test_cache_reuse(self, mock_embed):
        from consolidation_memory.topic_cache import get_topic_vecs, invalidate
        from consolidation_memory.database import ensure_schema, upsert_knowledge_topic

        ensure_schema()
        upsert_knowledge_topic("t1.md", "Title 1", "Summary 1", ["ep1"])

        fake_vecs = _make_normalized_batch(1, seed=42)
        mock_embed.return_value = fake_vecs
        invalidate()
        topics1, vecs1 = get_topic_vecs()
        assert mock_embed.call_count == 1

        topics2, vecs2 = get_topic_vecs()
        assert mock_embed.call_count == 1


# ── Deduplication tests ──────────────────────────────────────────────────────

class TestDedup:
    @patch("consolidation_memory.backends.encode_documents")
    def test_blocks_identical_content(self, mock_embed):
        from consolidation_memory.database import ensure_schema

        ensure_schema()

        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        result1 = client.store("Test content for dedup", content_type="fact")
        assert result1.status == "stored"

        result2 = client.store("Test content for dedup", content_type="fact")
        assert result2.status == "duplicate_detected"

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_allows_different_content(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec1 = _make_normalized_vec(seed=42)
        vec2 = _make_normalized_vec(seed=99)

        call_count = [0]
        def side_effect(texts):
            call_count[0] += 1
            if call_count[0] <= 1:
                return vec1.reshape(1, -1)
            return vec2.reshape(1, -1)

        mock_embed.side_effect = side_effect

        result1 = client.store("First content", content_type="fact")
        assert result1.status == "stored"

        result2 = client.store("Completely different content", content_type="fact")
        assert result2.status == "stored"

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_dedup_ignores_deleted(self, mock_embed):
        from consolidation_memory.database import ensure_schema, soft_delete_episode
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        result1 = client.store("Deletable content", content_type="fact")
        ep_id = result1.id

        soft_delete_episode(ep_id)
        client._vector_store.remove(ep_id)

        result2 = client.store("Deletable content", content_type="fact")
        assert result2.status == "stored"

        client.close()


# ── LLM validation tests ────────────────────────────────────────────────────

class TestValidation:
    def _make_episodes(self, n=2):
        return [
            {"content": f"Episode {i} at C:\\Users\\test\\path{i} version 1.2.{i}"}
            for i in range(n)
        ]

    def test_good_output(self):
        from consolidation_memory.consolidation import _validate_llm_output
        text = """---
title: Good Topic
summary: VR stack uses SteamVR 2.1.0 with path C:\\Users\\test\\path0.
tags: [vr, steamvr]
confidence: 0.85
---

## Facts
- SteamVR version 1.2.0 is installed at C:\\Users\\test\\path0
- Another fact about version 1.2.1

## Solutions
### Fix calibration
1. Run calibration at C:\\Users\\test\\path1
"""
        episodes = self._make_episodes()
        valid, failures = _validate_llm_output(text, episodes)
        assert valid is True
        assert failures == []

    def test_missing_title(self):
        from consolidation_memory.consolidation import _validate_llm_output
        text = """---
summary: Some summary
tags: [test]
confidence: 0.8
---

## Facts
- A fact
"""
        valid, failures = _validate_llm_output(text, [])
        assert valid is False
        assert any("title" in f.lower() for f in failures)

    def test_vague_summary(self):
        from consolidation_memory.consolidation import _validate_llm_output
        text = """---
title: Some Topic
summary: This document discusses the VR setup process.
tags: [test]
confidence: 0.8
---

## Facts
- A fact
"""
        valid, failures = _validate_llm_output(text, [])
        assert valid is False
        assert any("vague" in f.lower() or "meta" in f.lower() for f in failures)

    def test_no_sections(self):
        from consolidation_memory.consolidation import _validate_llm_output
        text = """---
title: Flat Topic
summary: Dense factual summary with specifics.
tags: [test]
confidence: 0.8
---

Just plain text without any sections or bullets.
"""
        valid, failures = _validate_llm_output(text, [])
        assert valid is False
        assert any("section" in f.lower() or "heading" in f.lower() for f in failures)

    def test_specifics_preservation(self):
        from consolidation_memory.consolidation import _validate_llm_output
        episodes = [
            {"content": "Found bug at C:\\Users\\gore\\project\\main.py version 3.14.2"},
            {"content": "Fix applied at C:\\Users\\gore\\project\\fix.py version 3.14.3"},
        ]
        text = """---
title: Bug Fix
summary: A bug was found and fixed.
tags: [bug]
confidence: 0.8
---

## Facts
- A bug was identified
- A fix was applied
"""
        valid, failures = _validate_llm_output(text, episodes)
        assert valid is False
        assert any("specifics" in f.lower() or "preservation" in f.lower() for f in failures)


# ── Adaptive surprise scoring tests ──────────────────────────────────────────

class TestSurpriseAdjustment:
    def _setup_episodes(self, specs):
        from consolidation_memory.database import ensure_schema, insert_episode
        import consolidation_memory.database as database

        ensure_schema()
        ids = []
        now = datetime.now(timezone.utc)
        for access, surprise, days_ago in specs:
            ep_id = insert_episode(
                content=f"Episode access={access} surprise={surprise}",
                surprise_score=surprise,
            )
            updated = (now - timedelta(days=days_ago)).isoformat()
            with database.get_connection() as conn:
                conn.execute(
                    "UPDATE episodes SET access_count = ?, updated_at = ? WHERE id = ?",
                    (access, updated, ep_id),
                )
            ids.append(ep_id)
        return ids

    def test_boost_high_access(self):
        from consolidation_memory.consolidation import _adjust_surprise_scores
        from consolidation_memory.database import get_episode

        ids = self._setup_episodes([
            (0, 0.5, 0),
            (1, 0.5, 0),
            (10, 0.5, 0),
        ])
        adjusted = _adjust_surprise_scores()
        assert adjusted >= 1
        ep = get_episode(ids[2])
        assert ep["surprise_score"] > 0.5

    def test_decay_inactive(self):
        from consolidation_memory.consolidation import _adjust_surprise_scores
        from consolidation_memory.database import get_episode, mark_consolidated
        import consolidation_memory.database as database

        ids = self._setup_episodes([
            (0, 0.5, 30),
            (5, 0.5, 0),
        ])
        # Decay only applies to consolidated episodes
        mark_consolidated([ids[0]], "test_topic.md")
        # mark_consolidated resets updated_at — restore the 30-day-old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with database.get_connection() as conn:
            conn.execute("UPDATE episodes SET updated_at = ? WHERE id = ?", (old_ts, ids[0]))
        adjusted = _adjust_surprise_scores()
        assert adjusted >= 1
        ep = get_episode(ids[0])
        assert ep["surprise_score"] < 0.5

    def test_clamped_to_range(self):
        from consolidation_memory.consolidation import _adjust_surprise_scores
        from consolidation_memory.database import get_episode, mark_consolidated
        import consolidation_memory.database as database

        ids = self._setup_episodes([
            (0, 0.12, 30),
            (100, 0.98, 0),
        ])
        # Decay only applies to consolidated episodes
        mark_consolidated([ids[0]], "test_topic.md")
        # mark_consolidated resets updated_at — restore the 30-day-old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with database.get_connection() as conn:
            conn.execute("UPDATE episodes SET updated_at = ? WHERE id = ?", (old_ts, ids[0]))
        _adjust_surprise_scores()
        ep_low = get_episode(ids[0])
        ep_high = get_episode(ids[1])
        assert ep_low["surprise_score"] >= 0.1
        assert ep_high["surprise_score"] <= 1.0

    def test_median_no_change(self):
        from consolidation_memory.consolidation import _adjust_surprise_scores
        from consolidation_memory.database import get_episode

        ids = self._setup_episodes([
            (3, 0.5, 0),
            (3, 0.5, 0),
            (3, 0.5, 0),
        ])
        adjusted = _adjust_surprise_scores()
        assert adjusted == 0
        for eid in ids:
            assert get_episode(eid)["surprise_score"] == 0.5

    def test_boost_capped_at_015(self):
        from consolidation_memory.consolidation import _adjust_surprise_scores
        from consolidation_memory.database import get_episode

        ids = self._setup_episodes([
            (0, 0.5, 0),
            (1000, 0.5, 0),
        ])
        _adjust_surprise_scores()
        ep = get_episode(ids[1])
        assert ep["surprise_score"] <= 0.65 + 0.01


# ── Export tests ─────────────────────────────────────────────────────────────

class TestExport:
    def test_creates_export_file(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema, insert_episode
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import BACKUP_DIR

        ensure_schema()
        insert_episode(content="Export test episode", tags=["test"])

        client = MemoryClient(auto_consolidate=False)
        result = client.export()
        assert result.status == "exported"
        assert result.episodes == 1

        exports = list(BACKUP_DIR.glob("memory_export_*.json"))
        assert len(exports) == 1
        client.close()

    def test_includes_knowledge(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema, upsert_knowledge_topic
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import KNOWLEDGE_DIR

        ensure_schema()

        kf = KNOWLEDGE_DIR / "export_topic.md"
        kf.write_text("# Test Knowledge", encoding="utf-8")
        upsert_knowledge_topic("export_topic.md", "Export Topic", "Test", ["ep1"])

        client = MemoryClient(auto_consolidate=False)
        result = client.export()
        assert result.knowledge_topics == 1

        from consolidation_memory.config import BACKUP_DIR
        export_file = list(BACKUP_DIR.glob("memory_export_*.json"))[0]
        data = json.loads(export_file.read_text(encoding="utf-8"))
        assert data["knowledge_topics"][0]["file_content"] == "# Test Knowledge"
        client.close()

    def test_prunes_old_exports(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import BACKUP_DIR

        ensure_schema()

        client = MemoryClient(auto_consolidate=False)
        for i in range(7):
            client.export()
            time.sleep(0.05)

        exports = list(BACKUP_DIR.glob("memory_export_*.json"))
        assert len(exports) <= 5
        client.close()


# ── Frontmatter parsing tests ────────────────────────────────────────────────

class TestFrontmatter:
    def test_proper_frontmatter(self):
        from consolidation_memory.consolidation import _parse_frontmatter
        text = """---
title: Test Title
summary: A test summary
tags: [foo, bar]
confidence: 0.9
---

## Facts
- Fact 1"""
        result = _parse_frontmatter(text)
        assert result["meta"]["title"] == "Test Title"
        assert result["meta"]["summary"] == "A test summary"
        assert result["meta"]["tags"] == ["foo", "bar"]
        assert result["meta"]["confidence"] == 0.9
        assert "## Facts" in result["body"]

    def test_missing_closing_delimiter(self):
        from consolidation_memory.consolidation import _parse_frontmatter
        text = """---
title: No Closing
summary: Missing closing delimiter
tags: [test]
confidence: 0.7

## Facts
- A fact"""
        result = _parse_frontmatter(text)
        assert result["meta"]["title"] == "No Closing"
        assert "## Facts" in result["body"]

    def test_no_frontmatter(self):
        from consolidation_memory.consolidation import _parse_frontmatter
        text = "Just a plain document\n\nWith some text."
        result = _parse_frontmatter(text)
        assert result["meta"] == {}
        assert "Just a plain" in result["body"]

    def test_code_fences_stripped(self):
        from consolidation_memory.consolidation import _parse_frontmatter
        text = """```markdown
---
title: Fenced
summary: Has code fences
tags: [test]
confidence: 0.8
---

## Body
```"""
        result = _parse_frontmatter(text)
        assert result["meta"]["title"] == "Fenced"

    def test_normalize_fixes_missing_closing(self):
        from consolidation_memory.consolidation import _normalize_output
        text = """---
title: Broken
summary: No closing
tags: [test]
confidence: 0.8

## Facts
- Something"""
        fixed = _normalize_output(text)
        assert fixed.count("---") >= 2

    def test_quoted_tags(self):
        from consolidation_memory.consolidation import _parse_fm_lines
        meta = _parse_fm_lines("tags: ['foo', \"bar\", baz]")
        assert meta["tags"] == ["foo", "bar", "baz"]


# ── Cluster confidence tests ─────────────────────────────────────────────────

class TestClusterConfidence:
    def test_high_coherence_high_surprise(self):
        from consolidation_memory.consolidation import _compute_cluster_confidence
        sim_matrix = np.ones((3, 3), dtype=np.float32)
        episodes = [
            {"surprise_score": 0.9},
            {"surprise_score": 0.8},
            {"surprise_score": 0.85},
        ]
        conf = _compute_cluster_confidence(episodes, sim_matrix, [0, 1, 2])
        assert conf >= 0.9

    def test_low_coherence_low_surprise(self):
        from consolidation_memory.consolidation import _compute_cluster_confidence
        sim_matrix = np.full((3, 3), 0.3, dtype=np.float32)
        np.fill_diagonal(sim_matrix, 1.0)
        episodes = [
            {"surprise_score": 0.2},
            {"surprise_score": 0.1},
            {"surprise_score": 0.15},
        ]
        conf = _compute_cluster_confidence(episodes, sim_matrix, [0, 1, 2])
        assert conf <= 0.6
        assert conf >= 0.5

    def test_clamped_range(self):
        from consolidation_memory.consolidation import _compute_cluster_confidence
        sim_matrix = np.zeros((2, 2), dtype=np.float32)
        episodes = [{"surprise_score": 0.0}, {"surprise_score": 0.0}]
        conf = _compute_cluster_confidence(episodes, sim_matrix, [0, 1])
        assert conf >= 0.5
        assert conf <= 0.95


# ── Database export / bulk query tests ────────────────────────────────────────

class TestDatabaseExport:
    def test_get_all_episodes(self):
        from consolidation_memory.database import ensure_schema, insert_episode, get_all_episodes, soft_delete_episode
        ensure_schema()

        id1 = insert_episode(content="Ep 1", tags=[])
        id2 = insert_episode(content="Ep 2", tags=[])
        soft_delete_episode(id2)

        all_eps = get_all_episodes(include_deleted=False)
        assert len(all_eps) == 1
        assert all_eps[0]["id"] == id1

        all_with_deleted = get_all_episodes(include_deleted=True)
        assert len(all_with_deleted) == 2

    def test_get_all_active_episodes(self):
        from consolidation_memory.database import (
            ensure_schema, insert_episode, get_all_active_episodes,
            mark_pruned, soft_delete_episode,
        )
        ensure_schema()

        id1 = insert_episode(content="Active", tags=[])
        id2 = insert_episode(content="Will prune", tags=[])
        id3 = insert_episode(content="Will delete", tags=[])

        mark_pruned([id2])
        soft_delete_episode(id3)

        active = get_all_active_episodes()
        active_ids = [ep["id"] for ep in active]
        assert id1 in active_ids
        assert id2 not in active_ids
        assert id3 not in active_ids

    def test_update_surprise_scores(self):
        from consolidation_memory.database import ensure_schema, insert_episode, get_episode, update_surprise_scores
        ensure_schema()

        id1 = insert_episode(content="Score test", surprise_score=0.5, tags=[])
        update_surprise_scores([(0.75, id1)])

        ep = get_episode(id1)
        assert ep["surprise_score"] == 0.75


# ── Backend factory tests ────────────────────────────────────────────────────

class TestBackendFactory:
    def test_reset_backends(self):
        from consolidation_memory.backends import reset_backends
        reset_backends()
        # After reset, globals should be None
        import consolidation_memory.backends as backends
        assert backends._embedding_backend is None
        assert backends._llm_backend is None

    @patch("consolidation_memory.config.EMBEDDING_BACKEND", "fastembed")
    def test_invalid_backend_raises(self):
        from consolidation_memory.backends import _create_embedding_backend
        with patch("consolidation_memory.config.EMBEDDING_BACKEND", "nonexistent"):
            with pytest.raises(ValueError, match="Unknown embedding backend"):
                _create_embedding_backend()
