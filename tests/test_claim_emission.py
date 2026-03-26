"""Tests for claim emission during consolidation create/merge flows."""

import json
from unittest.mock import patch
from typing import Any, cast

import numpy as np

from consolidation_memory.claim_graph import claim_from_record
from consolidation_memory.config import override_config
from consolidation_memory.consolidation.engine import (
    _build_deterministic_merge_payload,
    _merge_into_existing,
    _process_cluster,
)
from consolidation_memory.database import (
    ensure_schema,
    get_connection,
    get_records_by_topic,
    insert_episode,
    insert_knowledge_records,
    upsert_knowledge_topic,
)


class TestClaimEmission:
    @staticmethod
    def _content_dict(row: dict[str, Any]) -> dict[str, Any]:
        content = row.get("content", {})
        if isinstance(content, str):
            loaded = cast(dict[str, Any], json.loads(content))
            return loaded
        return dict(content)

    @staticmethod
    def _unit_vec(seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        vec = rng.randn(8).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def test_deterministic_merge_fallback_keeps_existing_topic_identity(self, tmp_data_dir):
        title, summary, tags, records = _build_deterministic_merge_payload(
            existing_records=[{"type": "fact", "subject": "Old", "info": "A"}],
            new_records=[{"type": "fact", "subject": "New", "info": "B"}],
            existing_title="Stable Topic",
            existing_summary="Stable summary",
            existing_tags=["stable"],
            extraction_data={
                "title": "Drifted Topic",
                "summary": "Drifted summary",
                "tags": ["new-tag"],
            },
        )
        assert title == "Stable Topic"
        assert summary == "Stable summary"
        assert set(tags) == {"stable", "new-tag"}
        assert len(records) == 2

    def test_claims_emitted_on_topic_create(self, tmp_data_dir):
        ensure_schema()

        insert_episode(
            content="We should use Python 3.12 for service X.",
            tags=[],
            episode_id="ep-create-1",
        )
        episode = {
            "id": "ep-create-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content_type": "exchange",
            "content": "We should use Python 3.12 for service X.",
            "tags": "[]",
            "surprise_score": 0.7,
        }
        sim_matrix = np.array([[1.0]], dtype=np.float32)
        extraction_data = {
            "title": "Claim Create Topic",
            "summary": "Python runtime decisions for service X.",
            "tags": ["python"],
            "records": [
                {"type": "fact", "subject": "Service X runtime", "info": "Python 3.12"},
                {
                    "type": "solution",
                    "problem": "Local startup error",
                    "fix": "pin uvicorn to 0.32.0",
                    "context": "dev env",
                },
            ],
        }

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                return_value=(extraction_data, 1),
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

        expected_claim_ids = {claim_from_record(rec)["id"] for rec in extraction_data["records"]}
        with get_connection() as conn:
            topic_row = conn.execute(
                "SELECT id FROM knowledge_topics WHERE title = ?",
                (extraction_data["title"],),
            ).fetchone()
            assert topic_row is not None
            topic_id = topic_row["id"]

            for claim_id in expected_claim_ids:
                claim_row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
                assert claim_row is not None
                assert claim_row["status"] == "active"

                source_count = conn.execute(
                    """SELECT COUNT(*) AS c FROM claim_sources
                       WHERE claim_id = ?
                         AND source_episode_id = ?
                         AND source_topic_id = ?
                         AND source_record_id IS NOT NULL""",
                    (claim_id, episode["id"], topic_id),
                ).fetchone()["c"]
                assert source_count == 1

                event_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM claim_events WHERE claim_id = ? AND event_type = 'create'",
                    (claim_id,),
                ).fetchone()["c"]
                assert event_count == 1

    def test_claims_updated_on_merge(self, tmp_data_dir):
        ensure_schema()

        topic_id = upsert_knowledge_topic(
            filename="claim_merge.md",
            title="Claim Merge Topic",
            summary="S",
            source_episodes=["ep-old"],
        )
        old_record = {"type": "fact", "subject": "Runtime", "info": "Python 3.11"}
        insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": old_record,
                    "embedding_text": "Runtime: Python 3.11",
                    "confidence": 0.8,
                },
            ],
        )

        new_record = {"type": "fact", "subject": "Runtime", "info": "Python 3.12"}
        extraction_data = {
            "title": "Claim Merge Topic",
            "summary": "S2",
            "tags": [],
            "records": [new_record],
        }
        merged_data = {
            "title": "Claim Merge Topic",
            "summary": "S2",
            "tags": [],
            "records": [old_record, new_record],
        }
        existing = {
            "id": topic_id,
            "filename": "claim_merge.md",
            "title": "Claim Merge Topic",
            "summary": "S",
        }
        insert_episode(content="merge source", tags=[], episode_id="ep-merge-1")

        # No contradiction: orthogonal vectors.
        new_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
        existing_vecs = np.array([[0.0, 1.0]], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                return_value=(merged_data, 1),
            ),
            patch(
                "consolidation_memory.consolidation.engine.encode_documents",
                side_effect=[new_vecs, existing_vecs],
            ),
            override_config(
                RENDER_MARKDOWN=False,
                MERGE_DROP_DETECTION_ENABLED=False,
                CONTRADICTION_LLM_ENABLED=False,
                CONTRADICTION_SIMILARITY_THRESHOLD=0.99,
            ),
        ):
            status, _ = _merge_into_existing(
                existing=existing,
                extraction_data=extraction_data,
                cluster_episodes=[{"id": "ep-merge-1", "content": "c", "tags": "[]"}],
                cluster_ep_ids=["ep-merge-1"],
                confidence=0.8,
            )

        assert status == "updated"

        expected_claim_ids = {claim_from_record(rec)["id"] for rec in merged_data["records"]}
        with get_connection() as conn:
            edge_count = conn.execute("SELECT COUNT(*) AS c FROM claim_edges").fetchone()["c"]
            assert edge_count == 0

            for claim_id in expected_claim_ids:
                assert conn.execute(
                    "SELECT 1 FROM claims WHERE id = ?",
                    (claim_id,),
                ).fetchone() is not None

                source_count = conn.execute(
                    """SELECT COUNT(*) AS c FROM claim_sources
                       WHERE claim_id = ?
                         AND source_episode_id = ?
                         AND source_topic_id = ?
                         AND source_record_id IS NOT NULL""",
                    (claim_id, "ep-merge-1", topic_id),
                ).fetchone()["c"]
                assert source_count >= 1

                update_events = conn.execute(
                    """SELECT COUNT(*) AS c FROM claim_events
                       WHERE claim_id = ? AND event_type = 'update'""",
                    (claim_id,),
                ).fetchone()["c"]
                assert update_events >= 1

    def test_contradiction_creates_claim_edge_and_event(self, tmp_data_dir):
        ensure_schema()

        topic_id = upsert_knowledge_topic(
            filename="claim_contradiction.md",
            title="Claim Contradiction",
            summary="S",
            source_episodes=["ep-old"],
        )
        old_record = {"type": "fact", "subject": "Python", "info": "version 3.11"}
        old_record_ids = insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": old_record,
                    "embedding_text": "Python version 3.11",
                    "confidence": 0.8,
                },
            ],
        )
        assert len(old_record_ids) == 1

        new_record = {"type": "fact", "subject": "Python", "info": "version 3.12"}
        extraction_data = {
            "title": "Claim Contradiction",
            "summary": "S2",
            "tags": [],
            "records": [new_record],
        }
        merged_data = {
            "title": "Claim Contradiction",
            "summary": "S2",
            "tags": [],
            "records": [new_record],
        }
        existing = {
            "id": topic_id,
            "filename": "claim_contradiction.md",
            "title": "Claim Contradiction",
            "summary": "S",
        }
        insert_episode(content="contradiction source", tags=[], episode_id="ep-contradict-1")

        def mock_encode(texts):
            return np.array([self._unit_vec(42) for _ in texts], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                return_value=(merged_data, 1),
            ),
            patch("consolidation_memory.consolidation.engine.encode_documents", mock_encode),
            override_config(
                RENDER_MARKDOWN=False,
                MERGE_DROP_DETECTION_ENABLED=False,
                CONTRADICTION_LLM_ENABLED=False,
                CONTRADICTION_SIMILARITY_THRESHOLD=0.5,
            ),
        ):
            status, _ = _merge_into_existing(
                existing=existing,
                extraction_data=extraction_data,
                cluster_episodes=[{"id": "ep-contradict-1", "content": "c", "tags": "[]"}],
                cluster_ep_ids=["ep-contradict-1"],
                confidence=0.8,
            )

        assert status == "updated"

        old_claim_id = claim_from_record(old_record)["id"]
        new_claim_id = claim_from_record(new_record)["id"]
        with get_connection() as conn:
            edge_row = conn.execute(
                """SELECT * FROM claim_edges
                   WHERE from_claim_id = ?
                     AND to_claim_id = ?
                     AND edge_type = 'contradicts'""",
                (new_claim_id, old_claim_id),
            ).fetchone()
            assert edge_row is not None

            event_rows = conn.execute(
                """SELECT claim_id, details FROM claim_events
                   WHERE event_type = 'contradiction'
                     AND claim_id IN (?, ?)""",
                (new_claim_id, old_claim_id),
            ).fetchall()
            assert len(event_rows) >= 2

        roles = set()
        for row in event_rows:
            details = json.loads(row["details"]) if row["details"] else {}
            if "role" in details:
                roles.add(details["role"])
        assert {"new", "old"}.issubset(roles)

    def test_temporal_validity_matches_record_windows(self, tmp_data_dir):
        ensure_schema()

        topic_id = upsert_knowledge_topic(
            filename="claim_temporal.md",
            title="Claim Temporal",
            summary="S",
            source_episodes=["ep-old"],
        )
        old_payload = {"type": "fact", "subject": "Interpreter", "info": "old version"}
        insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": old_payload,
                    "embedding_text": "Interpreter old version",
                    "confidence": 0.8,
                },
            ],
        )

        new_payload = {"type": "fact", "subject": "Interpreter", "info": "new version"}
        extraction_data = {
            "title": "Claim Temporal",
            "summary": "S2",
            "tags": [],
            "records": [new_payload],
        }
        merged_data = {
            "title": "Claim Temporal",
            "summary": "S2",
            "tags": [],
            "records": [new_payload],
        }
        existing = {
            "id": topic_id,
            "filename": "claim_temporal.md",
            "title": "Claim Temporal",
            "summary": "S",
        }
        insert_episode(content="temporal source", tags=[], episode_id="ep-temporal-1")

        def mock_encode(texts):
            return np.array([self._unit_vec(21) for _ in texts], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                return_value=(merged_data, 1),
            ),
            patch("consolidation_memory.consolidation.engine.encode_documents", mock_encode),
            override_config(
                RENDER_MARKDOWN=False,
                MERGE_DROP_DETECTION_ENABLED=False,
                CONTRADICTION_LLM_ENABLED=False,
                CONTRADICTION_SIMILARITY_THRESHOLD=0.5,
            ),
        ):
            status, _ = _merge_into_existing(
                existing=existing,
                extraction_data=extraction_data,
                cluster_episodes=[{"id": "ep-temporal-1", "content": "c", "tags": "[]"}],
                cluster_ep_ids=["ep-temporal-1"],
                confidence=0.8,
            )

        assert status == "updated"

        records = get_records_by_topic(topic_id, include_expired=True)
        old_record_row = None
        new_record_row = None
        for row in records:
            payload = self._content_dict(row)
            if payload.get("info") == "old version":
                old_record_row = row
            elif payload.get("info") == "new version":
                new_record_row = row

        assert old_record_row is not None
        assert new_record_row is not None
        assert old_record_row["valid_until"] is not None
        assert new_record_row["valid_from"] is not None

        old_claim_id = claim_from_record(old_payload)["id"]
        new_claim_id = claim_from_record(new_payload)["id"]
        with get_connection() as conn:
            old_claim_row = conn.execute(
                "SELECT * FROM claims WHERE id = ?",
                (old_claim_id,),
            ).fetchone()
            new_claim_row = conn.execute(
                "SELECT * FROM claims WHERE id = ?",
                (new_claim_id,),
            ).fetchone()

        assert old_claim_row is not None
        assert new_claim_row is not None
        assert old_claim_row["status"] == "expired"
        assert old_claim_row["valid_until"] == old_record_row["valid_until"]
        assert new_claim_row["status"] == "active"
        assert new_claim_row["valid_from"] == new_record_row["valid_from"]
        assert new_claim_row["valid_until"] == new_record_row["valid_until"]
