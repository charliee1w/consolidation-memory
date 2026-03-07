"""Tests for claim retrieval and public API wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from consolidation_memory.context_assembler import recall as assemble_recall
from consolidation_memory.database import ensure_schema, insert_episode, upsert_claim
from consolidation_memory.schemas import dispatch_tool_call
from consolidation_memory.types import ClaimBrowseResult, ClaimSearchResult
from consolidation_memory.vector_store import VectorStore
from tests.helpers import mock_encode

try:
    from fastapi.testclient import TestClient
    from consolidation_memory.rest import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TestClaimRecallPayload:
    def test_claims_included_in_recall_response(self, tmp_data_dir):
        ensure_schema()
        vs = VectorStore()

        ep_id = insert_episode(
            content="python runtime context",
            content_type="fact",
            tags=[],
            surprise_score=0.5,
        )
        episode_vec = mock_encode(["python runtime context"])[0]
        vs.add(ep_id, episode_vec)

        upsert_claim(
            claim_id="claim-runtime-312",
            claim_type="fact",
            canonical_text="python runtime is 3.12",
            payload={"subject": "python", "info": "3.12"},
            confidence=0.9,
            valid_from=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )

        with (
            patch(
                "consolidation_memory.context_assembler.backends.encode_query",
                side_effect=lambda text: mock_encode([text])[0],
            ),
            patch(
                "consolidation_memory.context_assembler.backends.encode_documents",
                side_effect=lambda texts: np.stack([mock_encode(["python runtime"])[0] for _ in texts]),
            ),
        ):
            result = assemble_recall(
                "python runtime",
                n_results=5,
                include_knowledge=True,
                vector_store=vs,
            )

        assert "episodes" in result
        assert "knowledge" in result
        assert "records" in result
        assert "warnings" in result
        assert "claims" in result
        assert isinstance(result["claims"], list)
        assert any(c["id"] == "claim-runtime-312" for c in result["claims"])

    def test_claim_as_of_behavior(self, tmp_data_dir):
        ensure_schema()
        vs = VectorStore()

        ep_id = insert_episode(
            content="python version history",
            content_type="fact",
            tags=[],
            surprise_score=0.5,
        )
        episode_vec = mock_encode(["python version history"])[0]
        vs.add(ep_id, episode_vec)

        upsert_claim(
            claim_id="claim-python-311",
            claim_type="fact",
            canonical_text="python runtime was 3.11",
            payload={"subject": "python", "info": "3.11"},
            confidence=0.8,
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-python-312",
            claim_type="fact",
            canonical_text="python runtime is 3.12",
            payload={"subject": "python", "info": "3.12"},
            confidence=0.9,
            valid_from="2025-07-01T00:00:00+00:00",
        )

        with (
            patch(
                "consolidation_memory.context_assembler.backends.encode_query",
                side_effect=lambda text: mock_encode([text])[0],
            ),
            patch(
                "consolidation_memory.context_assembler.backends.encode_documents",
                side_effect=lambda texts: np.stack([mock_encode(["python runtime"])[0] for _ in texts]),
            ),
        ):
            early = assemble_recall(
                "python runtime",
                n_results=5,
                include_knowledge=True,
                vector_store=vs,
                as_of="2025-03-01T00:00:00+00:00",
            )
            late = assemble_recall(
                "python runtime",
                n_results=5,
                include_knowledge=True,
                vector_store=vs,
                as_of="2025-08-01T00:00:00+00:00",
            )

        early_ids = {c["id"] for c in early["claims"]}
        late_ids = {c["id"] for c in late["claims"]}
        assert "claim-python-311" in early_ids
        assert "claim-python-312" not in early_ids
        assert "claim-python-311" not in late_ids
        assert "claim-python-312" in late_ids


class TestClaimToolDispatch:
    def test_mcp_dispatch_for_claim_calls(self):
        client = MagicMock()
        client.query_browse_claims.return_value = ClaimBrowseResult(
            claims=[{"id": "claim-a"}],
            total=1,
            claim_type="fact",
            as_of="2026-01-01T00:00:00+00:00",
        )
        client.query_search_claims.return_value = ClaimSearchResult(
            claims=[{"id": "claim-b"}],
            total_matches=1,
            query="uvicorn",
            claim_type="procedure",
            as_of="2026-01-01T00:00:00+00:00",
        )

        browse_out = dispatch_tool_call(
            client,
            "memory_claim_browse",
            {"claim_type": "fact", "as_of": "2026-01-01T00:00:00+00:00", "limit": 10},
        )
        search_out = dispatch_tool_call(
            client,
            "memory_claim_search",
            {"query": "uvicorn", "claim_type": "procedure", "as_of": "2026-01-01T00:00:00+00:00"},
        )

        assert browse_out["total"] == 1
        assert browse_out["claims"][0]["id"] == "claim-a"
        assert search_out["total_matches"] == 1
        assert search_out["claims"][0]["id"] == "claim-b"
        client.query_browse_claims.assert_called_once_with(
            claim_type="fact",
            as_of="2026-01-01T00:00:00+00:00",
            limit=10,
        )
        client.query_search_claims.assert_called_once_with(
            query="uvicorn",
            claim_type="procedure",
            as_of="2026-01-01T00:00:00+00:00",
            limit=50,
        )


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestClaimRestValidation:
    def test_claim_rest_request_response_validation(self, tmp_data_dir):
        ensure_schema()
        upsert_claim(
            claim_id="claim-rest-validation",
            claim_type="fact",
            canonical_text="python runtime is 3.12",
            payload={"subject": "python", "info": "3.12"},
            valid_from="2025-01-01T00:00:00+00:00",
        )

        app = create_app()
        with TestClient(app) as api:
            valid_resp = api.post("/memory/claims/search", json={"query": "python"})
            assert valid_resp.status_code == 200
            body = valid_resp.json()
            assert "claims" in body
            assert "total_matches" in body
            assert any(c["id"] == "claim-rest-validation" for c in body["claims"])

            missing_query = api.post("/memory/claims/search", json={})
            assert missing_query.status_code == 422

            invalid_limit = api.post("/memory/claims/browse", json={"limit": 0})
            assert invalid_limit.status_code == 422


class TestRecallBackwardCompatibility:
    def test_client_recall_compatible_when_claims_missing(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            legacy_payload = {
                "episodes": [{"id": "ep-1", "content": "x"}],
                "knowledge": [{"topic": "python"}],
                "records": [{"id": "rec-1"}],
                "warnings": ["legacy warning"],
            }
            fake_stats = {
                "episodic_buffer": {"total": 1},
                "knowledge_base": {"total_topics": 1},
            }

            with (
                patch("consolidation_memory.context_assembler.recall", return_value=legacy_payload),
                patch("consolidation_memory.database.get_stats", return_value=fake_stats),
            ):
                result = client.recall("python")

            assert result.episodes == legacy_payload["episodes"]
            assert result.knowledge == legacy_payload["knowledge"]
            assert result.records == legacy_payload["records"]
            assert result.warnings == legacy_payload["warnings"]
            assert result.claims == []
            assert result.total_episodes == 1
            assert result.total_knowledge_topics == 1
        finally:
            client.close()
