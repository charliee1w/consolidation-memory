"""Tests for entity-centric recall."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from consolidation_memory.context_assembler import recall as assemble_recall
from consolidation_memory.database import (
    ensure_schema,
    insert_claim_sources,
    insert_episode,
    insert_episode_anchors,
    upsert_claim,
)
from consolidation_memory.entity_recall import (
    resolve_entity_anchors,
)
from consolidation_memory.schemas import dispatch_tool_call
from consolidation_memory.vector_store import VectorStore
from tests.helpers import mock_encode


class TestEntityResolution:
    def test_resolve_path_entity_anchors(self):
        anchors = resolve_entity_anchors("src/consolidation_memory/context_assembler.py")
        anchor_values = {a["anchor_value"] for a in anchors if a["anchor_type"] == "path"}
        assert "src/consolidation_memory/context_assembler.py" in anchor_values
        assert "context_assembler.py" in anchor_values

    def test_resolve_subject_entity_has_no_path_anchors(self):
        anchors = resolve_entity_anchors("memory_status")
        assert anchors == []


class TestEntityRecallIntegration:
    def test_entity_boosts_path_linked_episode(self, tmp_data_dir):
        ensure_schema()
        vs = VectorStore()

        linked_id = insert_episode(
            content="Problem: recall slowness. Fix: tuned context_assembler scoring.",
            content_type="solution",
            tags=["recall"],
            surprise_score=0.5,
        )
        insert_episode_anchors(
            linked_id,
            [{"anchor_type": "path", "anchor_value": "src/consolidation_memory/context_assembler.py"}],
        )
        noise_id = insert_episode(
            content="Unrelated tracker scope release automation notes.",
            content_type="fact",
            tags=["release"],
            surprise_score=0.5,
        )

        vs.add(noise_id, mock_encode(["release automation tracker scope"])[0])
        vs.add(linked_id, mock_encode(["context assembler recall fix"])[0])

        with (
            patch(
                "consolidation_memory.context_assembler.backends.encode_query",
                side_effect=lambda text: mock_encode(["release automation tracker scope"])[0],
            ),
            patch(
                "consolidation_memory.context_assembler.backends.encode_documents",
                side_effect=lambda texts: np.stack([mock_encode([t])[0] for t in texts]),
            ),
        ):
            generic = assemble_recall(
                "release automation",
                n_results=2,
                include_knowledge=False,
                vector_store=vs,
            )
            entity = assemble_recall(
                "release automation",
                n_results=2,
                include_knowledge=False,
                vector_store=vs,
                entity="context_assembler.py",
            )

        assert generic["episodes"][0]["id"] == noise_id
        assert entity["episodes"][0]["id"] == linked_id
        assert entity["entity_resolution"]["kind"] == "path"
        assert entity["entity_resolution"]["linked_episode_count"] >= 1

    def test_entity_boosts_subject_linked_claim(self, tmp_data_dir):
        ensure_schema()
        vs = VectorStore()

        ep_id = insert_episode(
            content="memory status trust profile",
            content_type="fact",
            tags=[],
            surprise_score=0.5,
        )
        vs.add(ep_id, mock_encode(["memory status trust profile"])[0])

        upsert_claim(
            claim_id="claim-memory-status",
            claim_type="fact",
            canonical_text="type=fact | subject=memory_status | info=explains scheduler",
            payload={"type": "fact", "subject": "memory_status", "info": "explains scheduler"},
            confidence=0.9,
        )
        insert_claim_sources("claim-memory-status", [{"source_episode_id": ep_id}])

        with (
            patch(
                "consolidation_memory.context_assembler.backends.encode_query",
                side_effect=lambda text: mock_encode([text])[0],
            ),
            patch(
                "consolidation_memory.context_assembler.backends.encode_documents",
                side_effect=lambda texts: np.stack([mock_encode(["memory status"])[0] for _ in texts]),
            ),
        ):
            result = assemble_recall(
                "scheduler explanation",
                n_results=3,
                include_knowledge=True,
                vector_store=vs,
                entity="memory_status",
            )

        claim_ids = {claim["id"] for claim in result["claims"]}
        assert "claim-memory-status" in claim_ids
        assert result["entity_resolution"]["kind"] == "subject"
        assert result["entity_resolution"]["linked_claim_count"] >= 1

    def test_entity_includes_off_vector_episode(self, tmp_data_dir):
        ensure_schema()
        vs = VectorStore()

        linked_id = insert_episode(
            content="Problem: entity recall miss. Fix: anchor lookup in db/anchors.py.",
            content_type="solution",
            tags=[],
            surprise_score=0.5,
        )
        insert_episode_anchors(
            linked_id,
            [{"anchor_type": "path", "anchor_value": "db/anchors.py"}],
        )

        with (
            patch(
                "consolidation_memory.context_assembler.backends.encode_query",
                side_effect=lambda text: mock_encode(["completely unrelated query"])[0],
            ),
            patch(
                "consolidation_memory.context_assembler.backends.encode_documents",
                side_effect=lambda texts: np.stack([mock_encode(["x"])[0] for _ in texts]),
            ),
        ):
            result = assemble_recall(
                "completely unrelated query",
                n_results=5,
                include_knowledge=False,
                vector_store=vs,
                entity="db/anchors.py",
            )

        returned_ids = {episode["id"] for episode in result["episodes"]}
        assert linked_id in returned_ids


class TestEntityToolDispatch:
    def test_dispatch_passes_entity_to_client(self):
        from unittest.mock import MagicMock

        from consolidation_memory.types import RecallResult

        client = MagicMock()
        client.query_recall.return_value = RecallResult(
            episodes=[],
            entity_resolution={"entity": "context_assembler.py", "kind": "path"},
        )

        dispatch_tool_call(
            client,
            "memory_recall",
            {"query": "recall fixes", "entity": "context_assembler.py"},
        )

        client.query_recall.assert_called_once()
        assert client.query_recall.call_args.kwargs["entity"] == "context_assembler.py"