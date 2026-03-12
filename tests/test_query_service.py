"""Tests for canonical query service semantics."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from consolidation_memory.query_service import (
    CanonicalQueryService,
    ClaimBrowseQuery,
    ClaimSearchQuery,
    DriftQuery,
    RecallQuery,
)


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
            patch("consolidation_memory.database.get_stats", return_value=fake_stats) as mock_get_stats,
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
        mock_get_stats.assert_called_once_with(scope={"namespace_slug": "team-a"})


class TestCanonicalQueryServiceClaims:
    def test_browse_claims_applies_scope_filter_when_provided(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        rows = [{
            "id": "claim-a",
            "claim_type": "fact",
            "canonical_text": "python runtime is 3.12",
            "payload": "{\"subject\":\"python\",\"info\":\"3.12\"}",
            "status": "active",
            "confidence": 0.9,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }]

        with (
            patch("consolidation_memory.database.get_active_claims", return_value=rows),
            patch("consolidation_memory.query_service.filter_claims_for_scope", return_value=[]) as mock_scope_filter,
        ):
            result = service.browse_claims(
                ClaimBrowseQuery(claim_type="fact", as_of=None, limit=50),
                scope_filter={"project_slug": "repo-a"},
            )

        assert result.total == 0
        mock_scope_filter.assert_called_once()

    def test_search_claims_ranks_full_claim_pages(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        first_page = [{
            "id": f"claim-new-{i}",
            "claim_type": "procedure",
            "canonical_text": f"new claim {i}",
            "payload": "{}",
            "status": "active",
            "confidence": 0.5,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        } for i in range(250)]
        second_page = [
            {
                "id": "claim-strong",
                "claim_type": "procedure",
                "canonical_text": "start api with uvicorn main:app",
                "payload": "{\"trigger\": \"run api\", \"steps\": \"uvicorn main:app\"}",
                "status": "active",
                "confidence": 0.95,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
            },
            {
                "id": "claim-weak",
                "claim_type": "procedure",
                "canonical_text": "run server",
                "payload": "{\"steps\": \"python app.py\"}",
                "status": "active",
                "confidence": 0.6,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-02T00:00:00+00:00",
            },
        ]

        with patch(
            "consolidation_memory.database.get_active_claims",
            side_effect=[first_page, second_page, []],
        ) as mock_get_active_claims:
            result = service.search_claims(
                ClaimSearchQuery(
                    query="uvicorn",
                    claim_type="procedure",
                    as_of=None,
                    limit=1,
                )
            )

        assert result.total_matches == 1
        assert result.claims[0]["id"] == "claim-strong"
        assert result.claims[0]["relevance"] > 0
        assert mock_get_active_claims.call_args_list[0].kwargs["offset"] == 0
        assert mock_get_active_claims.call_args_list[1].kwargs["offset"] == 250

    def test_browse_claims_pages_until_scoped_limit_is_satisfied(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        first_page = [{
            "id": f"claim-out-{i}",
            "claim_type": "fact",
            "canonical_text": f"out of scope {i}",
            "payload": "{}",
            "status": "active",
            "confidence": 0.5,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": f"2025-01-01T00:00:{i:02d}+00:00",
        } for i in range(50)]
        second_page = [{
            "id": "claim-in-scope",
            "claim_type": "fact",
            "canonical_text": "in scope claim",
            "payload": "{}",
            "status": "active",
            "confidence": 0.9,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-02T00:00:00+00:00",
        }]

        def _filter_side_effect(claims, scope_filter):
            del scope_filter
            if claims and claims[0]["id"].startswith("claim-out-"):
                return []
            return claims

        with (
            patch(
                "consolidation_memory.database.get_active_claims",
                side_effect=[first_page, second_page],
            ) as mock_get_active_claims,
            patch(
                "consolidation_memory.query_service.filter_claims_for_scope",
                side_effect=_filter_side_effect,
            ),
        ):
            result = service.browse_claims(
                ClaimBrowseQuery(claim_type="fact", as_of=None, limit=1),
                scope_filter={"project_slug": "repo-a"},
            )

        assert result.total == 1
        assert [claim["id"] for claim in result.claims] == ["claim-in-scope"]
        assert mock_get_active_claims.call_args_list[0].kwargs["offset"] == 0
        assert mock_get_active_claims.call_args_list[1].kwargs["offset"] == 50


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
                DriftQuery(
                    base_ref="origin/main",
                    repo_path="C:/repo",
                    scope={"namespace_slug": "default", "project_slug": "repo-a"},
                )
            )

        assert result == expected
        mock_detect.assert_called_once_with(
            base_ref="origin/main",
            repo_path="C:/repo",
            scope={"namespace_slug": "default", "project_slug": "repo-a"},
        )
