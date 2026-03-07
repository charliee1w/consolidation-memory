"""Tests for canonical query service semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from consolidation_memory.query_service import (
    CanonicalQueryService,
    ClaimSearchQuery,
    DriftQuery,
    RecallQuery,
)
from consolidation_memory.types import ClaimBrowseResult


class TestCanonicalQueryServiceRecall:
    def test_recall_preserves_temporal_and_trust_signals(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        payload = {
            "episodes": [{"id": "ep-1", "content": "history"}],
            "knowledge": [{
                "topic": "python",
                "source_summary": "Based on 2 conversations (Jan 01, 2025)",
                "uncertainty": "Evolving — this topic has had recent contradictions",
            }],
            "records": [{
                "id": "rec-1",
                "source_summary": "Based on 1 conversation (Jan 01, 2025)",
                "uncertainty": "Low confidence — based on limited or conflicting information",
            }],
            "claims": [{
                "id": "claim-1",
                "status": "challenged",
                "uncertainty": "Recently contradicted - verify against newer evidence",
            }],
            "warnings": ["1 claim was recently contradicted (last 30 days)"],
        }
        fake_stats = {
            "episodic_buffer": {"total": 5},
            "knowledge_base": {"total_topics": 2},
        }

        with (
            patch("consolidation_memory.context_assembler.recall", return_value=payload) as mock_recall,
            patch("consolidation_memory.database.get_stats", return_value=fake_stats),
        ):
            result = service.recall(
                RecallQuery(
                    query="python runtime",
                    as_of="2025-06-01T00:00:00+00:00",
                ),
                scope_filter={"namespace_slug": "team-a"},
            )

        assert result.total_episodes == 5
        assert result.total_knowledge_topics == 2
        assert result.knowledge[0]["source_summary"].startswith("Based on 2 conversations")
        assert "Evolving" in result.knowledge[0]["uncertainty"]
        assert "Low confidence" in result.records[0]["uncertainty"]
        assert "Recently contradicted" in result.claims[0]["uncertainty"]
        assert result.warnings == ["1 claim was recently contradicted (last 30 days)"]

        call_kwargs = mock_recall.call_args.kwargs
        assert call_kwargs["as_of"] == "2025-06-01T00:00:00+00:00"
        assert call_kwargs["scope"] == {"namespace_slug": "team-a"}


class TestCanonicalQueryServiceClaims:
    def test_search_claims_ranks_from_browse_snapshot(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        browse_result = ClaimBrowseResult(
            claims=[
                {
                    "id": "claim-strong",
                    "canonical_text": "start api with uvicorn main:app",
                    "payload": {"trigger": "run api", "steps": "uvicorn main:app"},
                    "confidence": 0.95,
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "id": "claim-weak",
                    "canonical_text": "run server",
                    "payload": {"steps": "python app.py"},
                    "confidence": 0.6,
                    "updated_at": "2025-01-01T00:00:00+00:00",
                },
            ],
            total=2,
            claim_type="procedure",
            as_of="2026-02-01T00:00:00+00:00",
        )

        with patch.object(service, "browse_claims", return_value=browse_result) as mock_browse:
            result = service.search_claims(
                ClaimSearchQuery(
                    query="uvicorn",
                    claim_type="procedure",
                    as_of="2026-02-01T00:00:00+00:00",
                    limit=1,
                )
            )

        assert result.total_matches == 1
        assert result.claims[0]["id"] == "claim-strong"
        assert result.claims[0]["relevance"] > 0
        mock_browse.assert_called_once()


class TestCanonicalQueryServiceDrift:
    def test_detect_drift_delegates_to_drift_engine(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        expected = {
            "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            "impacted_claim_ids": ["claim-1"],
            "challenged_claim_ids": ["claim-1"],
            "impacts": [{
                "claim_id": "claim-1",
                "previous_status": "active",
                "new_status": "challenged",
                "matched_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            }],
        }

        with patch("consolidation_memory.drift.detect_code_drift", return_value=expected) as mock_detect:
            result = service.detect_drift(
                DriftQuery(base_ref="origin/main", repo_path="C:/repo")
            )

        assert result == expected
        mock_detect.assert_called_once_with(base_ref="origin/main", repo_path="C:/repo")

