"""Tests for deterministic consolidation fast-path behavior."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from consolidation_memory.claim_graph import claim_from_record
from consolidation_memory.config import get_config, override_config
from consolidation_memory.consolidation.engine import _process_cluster
from consolidation_memory.consolidation.fast_path import try_fast_path_extraction
from consolidation_memory.database import (
    ensure_schema,
    get_connection,
    get_records_by_topic,
    insert_episode,
    insert_knowledge_records,
    upsert_claim,
    upsert_knowledge_topic,
)
from consolidation_memory.knowledge_consistency import build_knowledge_consistency_report
from consolidation_memory.markdown_records import parse_markdown_records


class TestFastPathExtraction:
    def test_preference_episode_parses_without_llm(self, tmp_data_dir):
        ensure_schema()
        episodes = [
            {
                "id": "ep-pref-1",
                "content": "User prefers short PR summaries with concrete file paths.",
                "content_type": "preference",
                "tags": '["workflow", "reviews"]',
            }
        ]
        result = try_fast_path_extraction(episodes)
        assert result is not None
        records = result["extraction_data"]["records"]
        assert len(records) == 1
        assert records[0]["type"] == "preference"
        assert records[0]["key"] == "workflow, reviews"
        assert "short PR summaries" in str(records[0]["value"])

    def test_solution_episode_requires_path_anchor(self, tmp_data_dir):
        ensure_schema()
        episodes = [
            {
                "id": "ep-sol-1",
                "content": (
                    "Tests fail in tests/test_auth.py when JWT secret is missing. "
                    "Fix: set AUTH_JWT_SECRET in .env and run pytest tests/test_auth.py"
                ),
                "content_type": "solution",
                "tags": "[]",
            }
        ]
        result = try_fast_path_extraction(episodes)
        assert result is not None
        records = result["extraction_data"]["records"]
        assert records[0]["type"] == "solution"
        assert "tests/test_auth.py" in str(records[0]["context"])

    def test_unstructured_episode_returns_none(self, tmp_data_dir):
        ensure_schema()
        episodes = [
            {
                "id": "ep-freeform",
                "content": "We talked about many unrelated things over coffee.",
                "content_type": "exchange",
                "tags": "[]",
            }
        ]
        assert try_fast_path_extraction(episodes) is None

    def test_structured_json_fact_parses_without_llm(self, tmp_data_dir):
        ensure_schema()
        record = {"type": "fact", "subject": "Auth service", "info": "Uses JWT bearer tokens"}
        episodes = [
            {
                "id": "ep-fact-json",
                "content": json.dumps(record),
                "content_type": "fact",
                "tags": "[]",
            }
        ]
        result = try_fast_path_extraction(episodes)
        assert result is not None
        assert result["path_kind"] == "structured"
        records = result["extraction_data"]["records"]
        assert records[0] == record

    def test_structured_json_procedure_parses_list_steps(self, tmp_data_dir):
        ensure_schema()
        episodes = [
            {
                "id": "ep-proc-json",
                "content": json.dumps(
                    {
                        "type": "procedure",
                        "trigger": "before merging a PR",
                        "steps": ["run ruff", "run targeted pytest", "update changelog"],
                        "context": "release workflow",
                    }
                ),
                "content_type": "procedure",
                "tags": "[]",
            }
        ]
        result = try_fast_path_extraction(episodes)
        assert result is not None
        records = result["extraction_data"]["records"]
        assert records[0]["type"] == "procedure"
        assert records[0]["trigger"] == "before merging a PR"
        assert records[0]["steps"] == "run ruff | run targeted pytest | update changelog"
        assert records[0]["context"] == "release workflow"

    def test_procedure_text_episode_parses_without_llm(self, tmp_data_dir):
        ensure_schema()
        episodes = [
            {
                "id": "ep-proc-text",
                "content": (
                    "Trigger: CI fails on pull request\n"
                    "Steps: inspect logs, reproduce locally, add regression test"
                ),
                "content_type": "procedure",
                "tags": '["ci", "debugging"]',
            }
        ]
        result = try_fast_path_extraction(episodes)
        assert result is not None
        assert result["path_kind"] == "procedure"
        records = result["extraction_data"]["records"]
        assert records[0]["trigger"] == "CI fails on pull request"
        assert "reproduce locally" in records[0]["steps"]
        assert records[0]["context"] == "ci, debugging"

    def test_incomplete_structured_json_returns_none(self, tmp_data_dir):
        ensure_schema()
        episodes = [
            {
                "id": "ep-incomplete-fact",
                "content": json.dumps({"type": "fact", "subject": "Missing info field"}),
                "content_type": "fact",
                "tags": "[]",
            }
        ]
        assert try_fast_path_extraction(episodes) is None


class TestFastPathClusterProcessing:
    def test_preference_cluster_created_without_llm(self, tmp_data_dir):
        ensure_schema()
        insert_episode(
            content="User prefers concise commit messages.",
            content_type="preference",
            tags=["git"],
            episode_id="ep-fast-pref",
        )
        episode = {
            "id": "ep-fast-pref",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content_type": "preference",
            "content": "User prefers concise commit messages.",
            "tags": '["git"]',
            "surprise_score": 0.7,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM extraction should not run"),
            ),
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value=None,
            ),
            override_config(RENDER_MARKDOWN=False),
        ):
            result = _process_cluster(
                cluster_id=1,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "created"
        assert result["fast_path"] is True
        assert result["api_calls"] == 0

        with get_connection() as conn:
            topic_row = conn.execute(
                "SELECT id, title FROM knowledge_topics WHERE title LIKE 'Preference:%'"
            ).fetchone()
            assert topic_row is not None
            record_rows = conn.execute(
                "SELECT content FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_row["id"],),
            ).fetchall()
            assert len(record_rows) == 1
            content = json.loads(record_rows[0]["content"])
            assert content["type"] == "preference"
            assert content["key"] == "git"

    def test_existing_claim_merge_without_llm(self, tmp_data_dir):
        ensure_schema()
        record = {"type": "fact", "subject": "Service X runtime", "info": "Python 3.12"}
        claim = claim_from_record(record)
        topic_id = upsert_knowledge_topic(
            filename="existing_fact_topic.md",
            title="Service X runtime",
            summary="Runtime fact",
            source_episodes=[],
            fact_count=1,
        )
        insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": record,
                    "embedding_text": "Service X runtime Python 3.12",
                    "confidence": 0.8,
                }
            ],
            source_episodes=[],
        )
        upsert_claim(
            claim_id=claim["id"],
            claim_type=claim["claim_type"],
            canonical_text=claim["canonical_text"],
            payload=claim["payload"],
        )

        insert_episode(
            content=json.dumps(record),
            content_type="fact",
            tags=[],
            episode_id="ep-existing-claim",
        )
        episode = {
            "id": "ep-existing-claim",
            "created_at": "2026-01-02T00:00:00+00:00",
            "content_type": "fact",
            "content": json.dumps(record),
            "tags": "[]",
            "surprise_score": 0.6,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM extraction should not run"),
            ),
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value={
                    "id": topic_id,
                    "filename": "existing_fact_topic.md",
                    "title": "Service X runtime",
                    "summary": "Runtime fact",
                },
            ),
            override_config(RENDER_MARKDOWN=False, CONTRADICTION_LLM_ENABLED=False),
        ):
            result = _process_cluster(
                cluster_id=2,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "updated"
        assert result["fast_path"] is True
        assert result["api_calls"] == 0

    def test_unstructured_cluster_falls_back_to_llm(self, tmp_data_dir):
        ensure_schema()
        episode = {
            "id": "ep-llm-fallback",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content_type": "exchange",
            "content": "A long unstructured conversation with no stable structure.",
            "tags": "[]",
            "surprise_score": 0.5,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)
        extraction_data = {
            "title": "Fallback Topic",
            "summary": "Created by LLM fallback.",
            "tags": ["fallback"],
            "records": [{"type": "fact", "subject": "Topic", "info": "fallback"}],
        }

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                return_value=(extraction_data, 1),
            ) as mock_llm,
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value=None,
            ),
            override_config(RENDER_MARKDOWN=False),
        ):
            result = _process_cluster(
                cluster_id=3,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "created"
        assert result["fast_path"] is False
        assert result["api_calls"] == 1
        mock_llm.assert_called_once()

    def test_fact_cluster_created_without_llm(self, tmp_data_dir):
        ensure_schema()
        record = {
            "type": "fact",
            "subject": "Embedding backend",
            "info": "Default is fastembed with BAAI/bge-small-en-v1.5",
        }
        insert_episode(
            content=json.dumps(record),
            content_type="fact",
            tags=["config"],
            episode_id="ep-fast-fact",
        )
        episode = {
            "id": "ep-fast-fact",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content_type": "fact",
            "content": json.dumps(record),
            "tags": '["config"]',
            "surprise_score": 0.6,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM extraction should not run"),
            ),
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value=None,
            ),
            override_config(RENDER_MARKDOWN=False),
        ):
            result = _process_cluster(
                cluster_id=4,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "created"
        assert result["fast_path"] is True
        assert result["api_calls"] == 0

        with get_connection() as conn:
            topic_row = conn.execute(
                "SELECT id, title FROM knowledge_topics WHERE title LIKE 'Fact:%'"
            ).fetchone()
            assert topic_row is not None
            record_rows = conn.execute(
                "SELECT content FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_row["id"],),
            ).fetchall()
            assert len(record_rows) == 1
            content = json.loads(record_rows[0]["content"])
            assert content["type"] == "fact"
            assert content["subject"] == "Embedding backend"

    def test_procedure_cluster_created_without_llm(self, tmp_data_dir):
        ensure_schema()
        insert_episode(
            content=(
                "Trigger: before cutting a release\n"
                "Steps: run release gates, bump version, publish changelog"
            ),
            content_type="procedure",
            tags=["release"],
            episode_id="ep-fast-proc",
        )
        episode = {
            "id": "ep-fast-proc",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content_type": "procedure",
            "content": (
                "Trigger: before cutting a release\n"
                "Steps: run release gates, bump version, publish changelog"
            ),
            "tags": '["release"]',
            "surprise_score": 0.55,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM extraction should not run"),
            ),
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value=None,
            ),
            override_config(RENDER_MARKDOWN=False),
        ):
            result = _process_cluster(
                cluster_id=5,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "created"
        assert result["fast_path"] is True
        assert result["api_calls"] == 0

        with get_connection() as conn:
            topic_row = conn.execute(
                "SELECT id, title FROM knowledge_topics WHERE title LIKE 'Procedure:%'"
            ).fetchone()
            assert topic_row is not None
            record_rows = conn.execute(
                "SELECT content FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_row["id"],),
            ).fetchall()
            assert len(record_rows) == 1
            content = json.loads(record_rows[0]["content"])
            assert content["type"] == "procedure"
            assert content["trigger"] == "before cutting a release"

    def test_fast_path_create_renders_markdown_from_db_records(self, tmp_data_dir):
        ensure_schema()
        cfg = get_config()
        record = {
            "type": "fact",
            "subject": "Embedding backend",
            "info": "Default is fastembed with BAAI/bge-small-en-v1.5",
        }
        insert_episode(
            content=json.dumps(record),
            content_type="fact",
            tags=["config"],
            episode_id="ep-md-create",
        )
        episode = {
            "id": "ep-md-create",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content_type": "fact",
            "content": json.dumps(record),
            "tags": '["config"]',
            "surprise_score": 0.6,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM extraction should not run"),
            ),
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value=None,
            ),
            override_config(RENDER_MARKDOWN=True),
        ):
            result = _process_cluster(
                cluster_id=6,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "created"
        assert result["fast_path"] is True
        assert result["api_calls"] == 0

        with get_connection() as conn:
            topic_row = conn.execute(
                "SELECT id, filename, title FROM knowledge_topics WHERE title LIKE 'Fact:%'"
            ).fetchone()
            assert topic_row is not None
            topic_id = topic_row["id"]
            filename = topic_row["filename"]
            db_rows = conn.execute(
                "SELECT content FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_id,),
            ).fetchall()
            assert len(db_rows) == 1

        md_path = cfg.KNOWLEDGE_DIR / filename
        assert md_path.exists()
        markdown = md_path.read_text(encoding="utf-8")
        assert "Embedding backend" in markdown
        assert "fastembed" in markdown

        parsed = parse_markdown_records(markdown.split("---", 2)[-1])
        assert len(parsed) == 1
        assert parsed[0]["subject"] == "Embedding backend"

        report = build_knowledge_consistency_report()
        assert report["consistent_topics"] == 1
        assert report["issue_count"] == 0

    def test_fast_path_merge_renders_markdown_from_db_records(self, tmp_data_dir):
        ensure_schema()
        cfg = get_config()
        # Use unrelated subjects so embedding similarity stays below the
        # contradiction threshold when CONTRADICTION_LLM_ENABLED=False.
        existing_record = {
            "type": "fact",
            "subject": "Database engine",
            "info": "PostgreSQL 16",
        }
        new_record = {"type": "fact", "subject": "Service X port", "info": "8080"}
        topic_id = upsert_knowledge_topic(
            filename="merge_markdown_topic.md",
            title="Infrastructure facts",
            summary="Database and networking",
            source_episodes=[],
            fact_count=1,
        )
        insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": existing_record,
                    "embedding_text": "Database engine PostgreSQL 16",
                    "confidence": 0.8,
                }
            ],
            source_episodes=[],
        )
        (cfg.KNOWLEDGE_DIR / "merge_markdown_topic.md").write_text(
            "# stale markdown that should be replaced\n",
            encoding="utf-8",
        )

        insert_episode(
            content=json.dumps(new_record),
            content_type="fact",
            tags=[],
            episode_id="ep-md-merge",
        )
        episode = {
            "id": "ep-md-merge",
            "created_at": "2026-01-02T00:00:00+00:00",
            "content_type": "fact",
            "content": json.dumps(new_record),
            "tags": "[]",
            "surprise_score": 0.6,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM merge should not run"),
            ),
            patch(
                "consolidation_memory.consolidation.engine._find_similar_topic",
                return_value={
                    "id": topic_id,
                    "filename": "merge_markdown_topic.md",
                    "title": "Infrastructure facts",
                    "summary": "Database and networking",
                },
            ),
            override_config(RENDER_MARKDOWN=True, CONTRADICTION_LLM_ENABLED=False),
        ):
            result = _process_cluster(
                cluster_id=7,
                cluster_items=[(episode, 0)],
                sim_matrix=sim_matrix,
                cluster_confidences=[],
            )

        assert result["status"] == "updated"
        assert result["fast_path"] is True
        assert result["api_calls"] == 0

        db_rows = get_records_by_topic(topic_id, include_expired=False)
        assert len(db_rows) == 2

        markdown = (cfg.KNOWLEDGE_DIR / "merge_markdown_topic.md").read_text(encoding="utf-8")
        assert "PostgreSQL 16" in markdown
        assert "8080" in markdown
        assert "stale markdown" not in markdown

        parsed = parse_markdown_records(markdown.split("---", 2)[-1])
        assert len(parsed) == 2
        subjects = {rec["subject"] for rec in parsed}
        assert subjects == {"Database engine", "Service X port"}

        report = build_knowledge_consistency_report()
        assert report["consistent_topics"] == 1
        assert report["issue_count"] == 0