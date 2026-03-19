"""Tests for canonical query service semantics."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np

from consolidation_memory.query_service import (
    CanonicalQueryService,
    ClaimBrowseQuery,
    ClaimSearchQuery,
    DriftQuery,
    OutcomeBrowseQuery,
    RecallQuery,
)
from tests.helpers import mock_encode


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
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value={}),
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
        ) as mock_get_active_claims, patch(
            "consolidation_memory.database.get_claim_outcome_evidence",
            return_value={},
        ):
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
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value={}),
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

    def test_search_claims_strategy_reuse_ranking_prefers_validated_strategies(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        strategy_rows = [
            {
                "id": "strategy-validated",
                "claim_type": "strategy",
                "canonical_text": "debug flaky ci tests with deterministic reruns",
                "payload": (
                    '{"problem_pattern":"flaky ci tests","strategy":"rerun deterministically",'
                    '"expected_signals":"same failure reproduces"}'
                ),
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "id": "strategy-degraded",
                "claim_type": "strategy",
                "canonical_text": "debug flaky ci tests with deterministic reruns",
                "payload": (
                    '{"problem_pattern":"flaky ci tests","strategy":"rerun deterministically",'
                    '"failure_modes":"infra outage"}'
                ),
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        ]

        with (
            patch("consolidation_memory.database.get_active_claims", side_effect=[strategy_rows, []]),
            patch(
                "consolidation_memory.database.get_claim_outcome_evidence",
                return_value={
                    "strategy-validated": {
                        "validation_count": 4,
                        "success_count": 3,
                        "partial_success_count": 1,
                        "failure_count": 0,
                        "contradiction_count": 0,
                        "challenged_count": 0,
                        "last_observed_at": "2026-01-03T00:00:00+00:00",
                    },
                    "strategy-degraded": {
                        "validation_count": 4,
                        "success_count": 1,
                        "partial_success_count": 0,
                        "failure_count": 3,
                        "contradiction_count": 1,
                        "challenged_count": 1,
                        "last_observed_at": "2026-01-03T00:00:00+00:00",
                    },
                },
            ),
        ):
            result = service.search_claims(
                ClaimSearchQuery(query="flaky ci tests", claim_type="strategy", limit=10)
            )

        assert [claim["id"] for claim in result.claims] == [
            "strategy-validated",
            "strategy-degraded",
        ]
        assert result.claims[0]["relevance"] > result.claims[1]["relevance"]
        assert result.claims[0]["strategy_evidence"]["reusability"] == "validated"
        assert result.claims[1]["strategy_evidence"]["reusability"] == "degraded"
        assert result.claims[0]["reliability"]["score"] > result.claims[1]["reliability"]["score"]

    def test_browse_claims_attaches_strategy_evidence(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        strategy_rows = [{
            "id": "strategy-1",
            "claim_type": "strategy",
            "canonical_text": "triage flaky tests",
            "payload": '{"problem_pattern":"flaky tests","strategy":"rerun deterministically"}',
            "status": "active",
            "confidence": 0.85,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }]

        with (
            patch("consolidation_memory.database.get_active_claims", return_value=strategy_rows),
            patch(
                "consolidation_memory.database.get_claim_outcome_evidence",
                return_value={
                    "strategy-1": {
                        "validation_count": 2,
                        "success_count": 1,
                        "partial_success_count": 1,
                        "failure_count": 0,
                        "contradiction_count": 0,
                        "challenged_count": 0,
                        "last_observed_at": "2026-01-01T00:00:00+00:00",
                    },
                },
            ),
        ):
            result = service.browse_claims(ClaimBrowseQuery(claim_type="strategy", limit=10))

        assert result.total == 1
        assert result.claims[0]["strategy_evidence"]["validation_count"] == 2
        assert result.claims[0]["strategy_evidence"]["reusability"] == "mixed"
        assert "reliability" in result.claims[0]

    def test_search_claims_reliability_prefers_supported_claims(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        rows = [
            {
                "id": "claim-supported",
                "claim_type": "fact",
                "canonical_text": "target flaky ci tests quickly",
                "payload": "{\"subject\":\"ci\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
            {
                "id": "claim-unsupported",
                "claim_type": "fact",
                "canonical_text": "target flaky ci tests quickly",
                "payload": "{\"subject\":\"ci\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
        ]
        evidence = {
            "claim-supported": {
                "validation_count": 4,
                "success_count": 4,
                "source_link_count": 3,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "source_record_count": 1,
                "source_anchor_count": 2,
                "outcome_anchor_count": 1,
                "outcomes_with_provenance_count": 2,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
            "claim-unsupported": {
                "validation_count": 0,
                "success_count": 0,
                "source_link_count": 0,
                "source_episode_count": 0,
                "source_topic_count": 0,
                "source_record_count": 0,
                "source_anchor_count": 0,
                "outcome_anchor_count": 0,
                "outcomes_with_provenance_count": 0,
                "last_observed_at": None,
            },
        }

        with (
            patch("consolidation_memory.database.get_active_claims", side_effect=[rows, []]),
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value=evidence),
        ):
            result = service.search_claims(
                ClaimSearchQuery(query="flaky ci tests", claim_type="fact", limit=10)
            )

        assert [claim["id"] for claim in result.claims] == ["claim-supported", "claim-unsupported"]
        assert result.claims[0]["reliability"]["score"] > result.claims[1]["reliability"]["score"]
        assert result.claims[0]["relevance"] > result.claims[1]["relevance"]

    def test_search_claims_reliability_penalizes_challenged_drifted_claims(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        rows = [
            {
                "id": "claim-stable",
                "claim_type": "fact",
                "canonical_text": "debug flaky ci tests",
                "payload": "{\"subject\":\"ci\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
            {
                "id": "claim-drifted",
                "claim_type": "fact",
                "canonical_text": "debug flaky ci tests",
                "payload": "{\"subject\":\"ci\"}",
                "status": "challenged",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
        ]
        evidence = {
            "claim-stable": {
                "validation_count": 3,
                "success_count": 3,
                "source_link_count": 2,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "source_record_count": 0,
                "source_anchor_count": 1,
                "outcome_anchor_count": 1,
                "outcomes_with_provenance_count": 1,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
            "claim-drifted": {
                "validation_count": 3,
                "success_count": 3,
                "challenged_count": 1,
                "drift_event_count": 2,
                "source_link_count": 2,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "source_record_count": 0,
                "source_anchor_count": 1,
                "outcome_anchor_count": 1,
                "outcomes_with_provenance_count": 1,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
        }

        with (
            patch("consolidation_memory.database.get_active_claims", side_effect=[rows, []]),
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value=evidence),
        ):
            result = service.search_claims(
                ClaimSearchQuery(query="flaky ci tests", claim_type="fact", limit=10)
            )

        assert [claim["id"] for claim in result.claims] == ["claim-stable", "claim-drifted"]
        assert result.claims[0]["reliability"]["score"] > result.claims[1]["reliability"]["score"]
        assert result.claims[1]["reliability"]["recommendation"] in {"reuse_with_caution", "avoid_reuse"}

    def test_search_claims_prefers_validated_strategy_when_semantics_are_equal(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        rows = [
            {
                "id": "strategy-validated",
                "claim_type": "strategy",
                "canonical_text": "stabilize flaky ci tests with deterministic reruns",
                "payload": "{\"problem_pattern\":\"flaky ci tests\",\"strategy\":\"rerun deterministically\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-10T00:00:00+00:00",
            },
            {
                "id": "strategy-weak",
                "claim_type": "strategy",
                "canonical_text": "stabilize flaky ci tests with deterministic reruns",
                "payload": "{\"problem_pattern\":\"flaky ci tests\",\"strategy\":\"rerun deterministically\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-10T00:00:00+00:00",
            },
        ]
        evidence = {
            "strategy-validated": {
                "validation_count": 4,
                "success_count": 4,
                "failure_count": 0,
                "source_link_count": 3,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "source_record_count": 1,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
            "strategy-weak": {
                "validation_count": 1,
                "success_count": 0,
                "failure_count": 1,
                "source_link_count": 1,
                "source_episode_count": 1,
                "source_topic_count": 0,
                "source_record_count": 0,
                "challenged_count": 1,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
        }
        query_vec = mock_encode(["flaky ci tests"])[0]

        with (
            patch("consolidation_memory.database.get_active_claims", side_effect=[rows, []]),
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value=evidence),
            patch("consolidation_memory.query_service.backends.encode_query", return_value=query_vec),
            patch(
                "consolidation_memory.query_service.claim_cache.get_claim_vecs",
                return_value=np.stack([query_vec, query_vec]),
            ),
        ):
            result = service.search_claims(
                ClaimSearchQuery(query="flaky ci tests", claim_type="strategy", limit=10)
            )

        assert [claim["id"] for claim in result.claims] == ["strategy-validated", "strategy-weak"]

    def test_search_claims_demotes_stale_or_challenged_matches(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        rows = [
            {
                "id": "claim-current",
                "claim_type": "fact",
                "canonical_text": "debug flaky ci tests",
                "payload": "{\"subject\":\"ci\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
            {
                "id": "claim-stale-challenged",
                "claim_type": "fact",
                "canonical_text": "debug flaky ci tests",
                "payload": "{\"subject\":\"ci\"}",
                "status": "challenged",
                "confidence": 0.9,
                "valid_from": "2024-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            },
        ]
        evidence = {
            "claim-current": {
                "validation_count": 3,
                "success_count": 3,
                "source_link_count": 2,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
            "claim-stale-challenged": {
                "validation_count": 3,
                "success_count": 1,
                "failure_count": 2,
                "challenged_count": 1,
                "drift_event_count": 2,
                "source_link_count": 2,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "last_observed_at": "2025-01-01T00:00:00+00:00",
            },
        }
        query_vec = mock_encode(["debug flaky ci tests"])[0]

        with (
            patch("consolidation_memory.database.get_active_claims", side_effect=[rows, []]),
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value=evidence),
            patch("consolidation_memory.query_service.backends.encode_query", return_value=query_vec),
            patch(
                "consolidation_memory.query_service.claim_cache.get_claim_vecs",
                return_value=np.stack([query_vec, query_vec]),
            ),
        ):
            result = service.search_claims(ClaimSearchQuery(query="debug flaky ci tests", limit=10))

        assert [claim["id"] for claim in result.claims] == ["claim-current", "claim-stale-challenged"]
        assert (
            result.claims[0]["ranking"]["components"]["drift_challenge_penalty"]
            > result.claims[1]["ranking"]["components"]["drift_challenge_penalty"]
        )

    def test_search_claims_prefers_supported_active_claims_over_unsupported_matches(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        rows = [
            {
                "id": "claim-supported",
                "claim_type": "fact",
                "canonical_text": "target flaky ci tests quickly",
                "payload": "{\"subject\":\"ci\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
            {
                "id": "claim-unsupported",
                "claim_type": "fact",
                "canonical_text": "target flaky ci tests quickly",
                "payload": "{\"subject\":\"ci\"}",
                "status": "active",
                "confidence": 0.9,
                "valid_from": "2025-01-01T00:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2026-03-12T00:00:00+00:00",
            },
        ]
        evidence = {
            "claim-supported": {
                "validation_count": 4,
                "success_count": 4,
                "source_link_count": 3,
                "source_episode_count": 1,
                "source_topic_count": 1,
                "source_record_count": 1,
                "last_observed_at": "2026-03-12T00:00:00+00:00",
            },
            "claim-unsupported": {
                "validation_count": 0,
                "success_count": 0,
                "source_link_count": 0,
                "source_episode_count": 0,
                "source_topic_count": 0,
                "source_record_count": 0,
                "last_observed_at": None,
            },
        }
        query_vec = mock_encode(["flaky ci tests"])[0]

        with (
            patch("consolidation_memory.database.get_active_claims", side_effect=[rows, []]),
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value=evidence),
            patch("consolidation_memory.query_service.backends.encode_query", return_value=query_vec),
            patch(
                "consolidation_memory.query_service.claim_cache.get_claim_vecs",
                return_value=np.stack([query_vec, query_vec]),
            ),
        ):
            result = service.search_claims(ClaimSearchQuery(query="flaky ci tests", claim_type="fact"))

        assert [claim["id"] for claim in result.claims] == ["claim-supported", "claim-unsupported"]
        assert (
            result.claims[0]["ranking"]["components"]["outcome_support"]
            > result.claims[1]["ranking"]["components"]["outcome_support"]
        )

    def test_search_claims_preserves_temporal_and_scope_filtering_before_ranking(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        out_of_scope_page = [{
            "id": f"claim-hidden-{i:02d}",
            "claim_type": "fact",
            "canonical_text": "shared build strategy",
            "payload": "{}",
            "status": "active",
            "confidence": 0.8,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2026-03-12T00:00:00+00:00",
        } for i in range(50)]
        in_scope_page = [{
            "id": "claim-visible",
            "claim_type": "fact",
            "canonical_text": "shared build strategy",
            "payload": "{}",
            "status": "active",
            "confidence": 0.8,
            "valid_from": "2025-01-01T00:00:00+00:00",
            "valid_until": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2026-03-12T00:00:00+00:00",
        }]
        query_vec = mock_encode(["shared build strategy"])[0]

        def _scope_side_effect(claims, _scope_filter):
            if claims and claims[0]["id"].startswith("claim-hidden-"):
                return []
            return claims

        with (
            patch(
                "consolidation_memory.database.get_claims_as_of",
                side_effect=[out_of_scope_page, in_scope_page, []],
            ) as mock_as_of,
            patch("consolidation_memory.database.get_active_claims") as mock_active,
            patch(
                "consolidation_memory.query_service.filter_claims_for_scope",
                side_effect=_scope_side_effect,
            ),
            patch("consolidation_memory.database.get_claim_outcome_evidence", return_value={}),
            patch("consolidation_memory.query_service.backends.encode_query", return_value=query_vec),
            patch(
                "consolidation_memory.query_service.backends.encode_documents",
                side_effect=lambda texts: np.stack([query_vec for _ in texts]),
            ),
        ):
            result = service.search_claims(
                ClaimSearchQuery(
                    query="shared build strategy",
                    claim_type="fact",
                    as_of="2026-03-01T00:00:00+00:00",
                    limit=10,
                ),
                scope_filter={"project_slug": "repo-a"},
            )

        assert [claim["id"] for claim in result.claims] == ["claim-visible"]
        assert mock_as_of.call_count >= 2
        mock_active.assert_not_called()


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


class TestCanonicalQueryServiceOutcomes:
    def test_browse_outcomes_assembles_links_and_refs(self):
        service = CanonicalQueryService(vector_store=MagicMock())
        outcome_rows = [{
            "id": "outcome-1",
            "action_key": "act_abc",
            "action_summary": "run targeted tests",
            "outcome_type": "success",
            "summary": "all good",
            "details": "{\"duration\": 12}",
            "confidence": 0.9,
            "provenance": "{\"agent\":\"codex\"}",
            "observed_at": "2026-03-01T00:00:00+00:00",
            "created_at": "2026-03-01T00:00:00+00:00",
            "updated_at": "2026-03-01T00:00:00+00:00",
        }]
        source_rows = [{
            "id": "src-1",
            "outcome_id": "outcome-1",
            "source_claim_id": "claim-1",
            "source_record_id": None,
            "source_episode_id": "ep-1",
            "created_at": "2026-03-01T00:00:00+00:00",
        }]
        ref_rows = [{
            "id": "ref-1",
            "outcome_id": "outcome-1",
            "ref_type": "issue",
            "ref_key": "id",
            "ref_value": "ISSUE-1",
            "created_at": "2026-03-01T00:00:00+00:00",
        }]

        with (
            patch("consolidation_memory.database.get_action_outcomes", return_value=outcome_rows),
            patch("consolidation_memory.database.get_action_outcome_sources_by_outcome_ids", return_value=source_rows),
            patch("consolidation_memory.database.get_action_outcome_refs_by_outcome_ids", return_value=ref_rows),
        ):
            result = service.browse_outcomes(
                OutcomeBrowseQuery(
                    outcome_type="success",
                    source_claim_id="claim-1",
                    limit=10,
                ),
                scope_filter={"project_slug": "repo-a"},
            )

        assert result.total == 1
        outcome = result.outcomes[0]
        assert outcome["id"] == "outcome-1"
        assert outcome["source_claim_ids"] == ["claim-1"]
        assert outcome["source_episode_ids"] == ["ep-1"]
        assert outcome["issue_ids"] == ["ISSUE-1"]
