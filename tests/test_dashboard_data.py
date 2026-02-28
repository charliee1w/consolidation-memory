"""Tests for dashboard data-fetching layer."""

import json

import pytest

from consolidation_memory.database import (
    ensure_schema,
    insert_episode,
    upsert_knowledge_topic,
    insert_knowledge_records,
    start_consolidation_run,
    complete_consolidation_run,
)
from consolidation_memory.dashboard_data import DashboardData


@pytest.fixture
def data():
    """Create DashboardData instance with schema initialized."""
    ensure_schema()
    return DashboardData()


class TestGetEpisodes:
    def test_default_returns_episodes_sorted_by_created_at_desc(self, data):
        ensure_schema()
        id1 = insert_episode("first episode", content_type="fact")
        id2 = insert_episode("second episode", content_type="exchange")
        id3 = insert_episode("third episode", content_type="solution")

        episodes = data.get_episodes()
        assert len(episodes) == 3
        # Most recent first
        assert episodes[0]["id"] == id3
        assert episodes[1]["id"] == id2
        assert episodes[2]["id"] == id1

    def test_returns_expected_fields(self, data):
        insert_episode(
            "test content here",
            content_type="fact",
            tags=["python", "tips"],
            surprise_score=0.7,
        )

        episodes = data.get_episodes()
        assert len(episodes) == 1
        ep = episodes[0]
        assert "id" in ep
        assert "content_preview" in ep
        assert "content_type" in ep
        assert ep["content_type"] == "fact"
        assert "tags" in ep
        assert ep["tags"] == ["python", "tips"]
        assert "surprise_score" in ep
        assert ep["surprise_score"] == pytest.approx(0.7)
        assert "created_at" in ep
        assert "consolidated" in ep

    def test_content_preview_truncated(self, data):
        long_content = "x" * 200
        insert_episode(long_content)

        episodes = data.get_episodes()
        assert len(episodes[0]["content_preview"]) <= 83  # 80 + "..."

    def test_filter_by_content_type(self, data):
        insert_episode("fact episode", content_type="fact")
        insert_episode("exchange episode", content_type="exchange")
        insert_episode("another fact", content_type="fact")

        facts = data.get_episodes(content_type="fact")
        assert len(facts) == 2
        assert all(e["content_type"] == "fact" for e in facts)

    def test_sort_by_surprise_score(self, data):
        insert_episode("low surprise", surprise_score=0.1)
        insert_episode("high surprise", surprise_score=0.9)
        insert_episode("mid surprise", surprise_score=0.5)

        episodes = data.get_episodes(sort_by="surprise_score", desc=True)
        scores = [e["surprise_score"] for e in episodes]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_content_type(self, data):
        insert_episode("z episode", content_type="solution")
        insert_episode("a episode", content_type="exchange")
        insert_episode("m episode", content_type="fact")

        episodes = data.get_episodes(sort_by="content_type", desc=False)
        types = [e["content_type"] for e in episodes]
        assert types == sorted(types)

    def test_limit(self, data):
        for i in range(10):
            insert_episode(f"episode {i}")

        episodes = data.get_episodes(limit=3)
        assert len(episodes) == 3

    def test_excludes_deleted(self, data):
        from consolidation_memory.database import soft_delete_episode

        id1 = insert_episode("keep me")
        id2 = insert_episode("delete me")
        soft_delete_episode(id2)

        episodes = data.get_episodes()
        assert len(episodes) == 1
        assert episodes[0]["id"] == id1


class TestGetKnowledgeTopics:
    def test_returns_topics_with_fields(self, data):
        topic_id = upsert_knowledge_topic(
            filename="python_tips.md",
            title="Python Tips",
            summary="Tips for Python development",
            source_episodes=["ep1", "ep2", "ep3"],
            fact_count=5,
            confidence=0.85,
        )

        topics = data.get_knowledge_topics()
        assert len(topics) == 1
        t = topics[0]
        assert t["id"] == topic_id
        assert t["filename"] == "python_tips.md"
        assert t["title"] == "Python Tips"
        assert t["summary"] == "Tips for Python development"
        assert t["fact_count"] == 5
        assert t["confidence"] == pytest.approx(0.85)
        assert t["source_episode_count"] == 3
        assert "created_at" in t
        assert "updated_at" in t

    def test_ordered_by_updated_at_desc(self, data):
        upsert_knowledge_topic("a.md", "A", "summary A", ["ep1"])
        upsert_knowledge_topic("b.md", "B", "summary B", ["ep2"])
        # Update A to make it most recent
        upsert_knowledge_topic("a.md", "A Updated", "new summary", ["ep3"])

        topics = data.get_knowledge_topics()
        assert topics[0]["filename"] == "a.md"
        assert topics[1]["filename"] == "b.md"


class TestGetRecordsForTopic:
    def test_returns_records(self, data):
        topic_id = upsert_knowledge_topic("test.md", "Test", "summary", ["ep1"])
        insert_knowledge_records(topic_id, [
            {"record_type": "fact", "content": '{"key": "value"}', "embedding_text": "test fact",
             "confidence": 0.9},
            {"record_type": "solution", "content": '{"fix": "it"}', "embedding_text": "test solution"},
        ], source_episodes=["ep1"])

        records = data.get_records_for_topic(topic_id)
        assert len(records) == 2
        assert records[0]["record_type"] in ("fact", "solution")
        assert "content" in records[0]
        assert "confidence" in records[0]
        assert "created_at" in records[0]

    def test_empty_for_nonexistent_topic(self, data):
        records = data.get_records_for_topic("nonexistent-id")
        assert records == []


class TestGetConsolidationRuns:
    def test_returns_runs_newest_first(self, data):
        run1 = start_consolidation_run()
        complete_consolidation_run(
            run1, episodes_processed=10, clusters_formed=3,
            topics_created=2, topics_updated=1, episodes_pruned=5,
        )
        run2 = start_consolidation_run()
        complete_consolidation_run(
            run2, episodes_processed=20, clusters_formed=5,
            topics_created=1, topics_updated=3,
        )

        runs = data.get_consolidation_runs()
        assert len(runs) == 2
        assert runs[0]["id"] == run2  # newest first

        r = runs[0]
        assert r["episodes_processed"] == 20
        assert r["clusters_formed"] == 5
        assert r["topics_created"] == 1
        assert r["topics_updated"] == 3
        assert r["status"] == "completed"
        assert "started_at" in r
        assert "completed_at" in r

    def test_limit(self, data):
        for _ in range(5):
            rid = start_consolidation_run()
            complete_consolidation_run(rid)

        runs = data.get_consolidation_runs(limit=2)
        assert len(runs) == 2

    def test_includes_error_runs(self, data):
        rid = start_consolidation_run()
        complete_consolidation_run(rid, status="error", error_message="LLM timeout")

        runs = data.get_consolidation_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "error"
        assert runs[0]["error_message"] == "LLM timeout"


class TestGetStats:
    def test_episode_counts_by_type(self, data):
        insert_episode("e1", content_type="exchange")
        insert_episode("e2", content_type="exchange")
        insert_episode("e3", content_type="fact")
        insert_episode("e4", content_type="solution")
        insert_episode("e5", content_type="preference")

        stats = data.get_stats()
        assert stats["episodes_by_type"]["exchange"] == 2
        assert stats["episodes_by_type"]["fact"] == 1
        assert stats["episodes_by_type"]["solution"] == 1
        assert stats["episodes_by_type"]["preference"] == 1
        assert stats["total_episodes"] == 5

    def test_knowledge_and_record_counts(self, data):
        tid = upsert_knowledge_topic("t.md", "T", "s", ["ep1"], fact_count=3)
        insert_knowledge_records(tid, [
            {"record_type": "fact", "content": "{}", "embedding_text": "f"},
            {"record_type": "solution", "content": "{}", "embedding_text": "s"},
        ])

        stats = data.get_stats()
        assert stats["knowledge_topic_count"] == 1
        assert stats["record_count"] == 2

    def test_db_size_present(self, data):
        stats = data.get_stats()
        assert "db_size_mb" in stats
        assert isinstance(stats["db_size_mb"], float)

    def test_last_consolidation(self, data):
        rid = start_consolidation_run()
        complete_consolidation_run(rid)

        stats = data.get_stats()
        assert stats["last_consolidation"] is not None

    def test_empty_db(self, data):
        stats = data.get_stats()
        assert stats["total_episodes"] == 0
        assert stats["knowledge_topic_count"] == 0
        assert stats["record_count"] == 0
        assert stats["last_consolidation"] is None


class TestGetFaissStats:
    def test_no_index_files(self, data):
        stats = data.get_faiss_stats()
        assert stats["index_size"] == 0
        assert stats["tombstone_count"] == 0
        assert stats["tombstone_ratio"] == 0.0

    def test_reads_from_metadata_files(self, data, tmp_data_dir):
        from consolidation_memory.config import get_config
        cfg = get_config()

        # Write mock FAISS metadata files
        id_map = ["id1", "id2", "id3", "id4", "id5"]
        cfg.FAISS_ID_MAP_PATH.write_text(json.dumps(id_map))

        tombstones = ["id2", "id4"]
        cfg.FAISS_TOMBSTONE_PATH.write_text(json.dumps(tombstones))

        stats = data.get_faiss_stats()
        assert stats["index_size"] == 3  # 5 total - 2 tombstones
        assert stats["tombstone_count"] == 2
        assert stats["tombstone_ratio"] == pytest.approx(2 / 5)


class TestEmptyDatabase:
    def test_all_methods_return_empty(self, data):
        assert data.get_episodes() == []
        assert data.get_knowledge_topics() == []
        assert data.get_records_for_topic("any-id") == []
        assert data.get_consolidation_runs() == []

        stats = data.get_stats()
        assert stats["total_episodes"] == 0

        faiss = data.get_faiss_stats()
        assert faiss["index_size"] == 0
