"""Tests for claim retrieval in recall flow."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

from consolidation_memory.context_assembler import _search_claims, recall
from consolidation_memory.database import (
    ensure_schema,
    insert_claim_event,
    insert_episode,
    insert_claim_sources,
    upsert_claim,
)
from consolidation_memory.vector_store import VectorStore
from tests.helpers import mock_encode


class TestClaimSearch:
    def test_search_claims_uses_temporal_source_when_as_of_set(self, tmp_data_dir):
        ensure_schema()
        query_vec = mock_encode(["python"])[0]
        claim_row = {
            "id": "claim-1",
            "claim_type": "fact",
            "canonical_text": "python runtime is 3.12",
            "payload": '{"subject":"python","info":"3.12"}',
            "status": "active",
            "confidence": 0.9,
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }

        with (
            patch("consolidation_memory.context_assembler.get_claims_as_of") as mock_as_of,
            patch("consolidation_memory.context_assembler.get_active_claims") as mock_active,
            patch("consolidation_memory.context_assembler.backends.encode_documents") as mock_docs,
        ):
            mock_as_of.return_value = [claim_row]
            mock_active.return_value = []
            mock_docs.return_value = np.stack([query_vec])

            claims, _warnings = _search_claims(
                "python",
                query_vec,
                as_of="2026-02-01T00:00:00+00:00",
            )

        mock_as_of.assert_called_once()
        mock_active.assert_not_called()
        assert len(claims) == 1
        assert claims[0]["id"] == "claim-1"

    def test_search_claims_adds_uncertainty_for_low_conf_and_contradiction(self, tmp_data_dir):
        ensure_schema()

        claim_id = "claim-low-contradicted"
        upsert_claim(
            claim_id=claim_id,
            claim_type="fact",
            canonical_text="python runtime is uncertain",
            payload={"subject": "python", "info": "maybe 3.11"},
            confidence=0.4,
            valid_from=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        )
        insert_claim_event(
            claim_id=claim_id,
            event_type="contradiction",
            details={"reason": "new evidence"},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        query_vec = mock_encode(["python runtime"])[0]
        with patch(
            "consolidation_memory.context_assembler.backends.encode_documents",
            side_effect=lambda texts: np.stack([query_vec for _ in texts]),
        ):
            claims, warnings = _search_claims("python runtime", query_vec)

        target = next(c for c in claims if c["id"] == claim_id)
        uncertainty = target.get("uncertainty", "")
        assert "Low confidence" in uncertainty
        assert "Recently contradicted" in uncertainty
        assert any("low confidence" in w for w in warnings)
        assert any("recently contradicted" in w for w in warnings)

    def test_search_claims_temporal_filtering(self, tmp_data_dir):
        ensure_schema()
        upsert_claim(
            claim_id="claim-old",
            claim_type="fact",
            canonical_text="python runtime was 3.11",
            payload={"subject": "python", "info": "3.11"},
            confidence=0.8,
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-new",
            claim_type="fact",
            canonical_text="python runtime is 3.12",
            payload={"subject": "python", "info": "3.12"},
            confidence=0.9,
            valid_from="2025-06-01T00:00:00+00:00",
        )

        query_vec = mock_encode(["python runtime"])[0]
        with patch(
            "consolidation_memory.context_assembler.backends.encode_documents",
            side_effect=lambda texts: np.stack([query_vec for _ in texts]),
        ):
            early_claims, _ = _search_claims(
                "python runtime",
                query_vec,
                as_of="2025-03-01T00:00:00+00:00",
            )
            late_claims, _ = _search_claims(
                "python runtime",
                query_vec,
                as_of="2025-07-01T00:00:00+00:00",
            )

        early_ids = {c["id"] for c in early_claims}
        late_ids = {c["id"] for c in late_claims}
        assert "claim-old" in early_ids
        assert "claim-new" not in early_ids
        assert "claim-new" in late_ids

    def test_search_claims_pages_until_scoped_matches_are_found(self, tmp_data_dir):
        ensure_schema()
        visible_episode_id = insert_episode(
            content="visible provenance",
            scope={
                "namespace_slug": "default",
                "project_slug": "default",
                "app_client_name": "visible-client",
                "app_client_type": "python_sdk",
            },
        )
        hidden_episode_id = insert_episode(
            content="hidden provenance",
            scope={
                "namespace_slug": "default",
                "project_slug": "default",
                "app_client_name": "hidden-client",
                "app_client_type": "python_sdk",
            },
        )
        upsert_claim(
            claim_id="claim-visible",
            claim_type="fact",
            canonical_text="shared claim text",
            payload={"subject": "visible"},
            confidence=0.9,
            valid_from="2026-01-01T00:00:00+00:00",
        )
        insert_claim_sources("claim-visible", [{"source_episode_id": visible_episode_id}])

        for i in range(260):
            claim_id = f"claim-hidden-{i:03d}"
            upsert_claim(
                claim_id=claim_id,
                claim_type="fact",
                canonical_text="shared claim text",
                payload={"subject": f"hidden-{i}"},
                confidence=0.9,
                valid_from="2026-01-01T00:00:00+00:00",
            )
            insert_claim_sources(claim_id, [{"source_episode_id": hidden_episode_id}])

        query_vec = mock_encode(["shared claim text"])[0]
        with patch(
            "consolidation_memory.context_assembler.backends.encode_documents",
            side_effect=lambda texts: np.stack([query_vec for _ in texts]),
        ):
            claims, _warnings = _search_claims(
                "shared claim text",
                query_vec,
                scope={
                    "namespace_slug": "default",
                    "project_slug": "default",
                    "app_client_name": "visible-client",
                    "app_client_type": "python_sdk",
                },
            )

        assert [claim["id"] for claim in claims] == ["claim-visible"]


class TestRecallClaimsIntegration:
    def test_recall_returns_claims_field(self, tmp_data_dir):
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
            claim_id="claim-runtime",
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
            result = recall(
                "python runtime",
                n_results=5,
                include_knowledge=True,
                vector_store=vs,
            )

        assert "episodes" in result
        assert "knowledge" in result
        assert "records" in result
        assert "claims" in result
        assert isinstance(result["claims"], list)
        assert any(c["id"] == "claim-runtime" for c in result["claims"])
