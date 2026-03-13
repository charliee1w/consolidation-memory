"""Regression tests for first-class action outcome tracking."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from consolidation_memory.client import MemoryClient
from consolidation_memory.cli import cmd_import
from consolidation_memory.config import get_config
from consolidation_memory.database import (
    close_all_connections,
    ensure_schema,
    get_action_outcomes,
    get_all_action_outcome_refs,
    get_all_action_outcome_sources,
    get_all_action_outcomes,
    insert_claim_sources,
    insert_episode,
    insert_knowledge_records,
    record_action_outcome,
    upsert_claim,
    upsert_knowledge_topic,
)
from tests.helpers import make_normalized_vec as _make_normalized_vec


def _seed_claim_with_episode(
    *,
    claim_id: str,
    canonical_text: str,
    payload: dict[str, str],
    scope: dict[str, str] | None = None,
) -> str:
    episode_id = insert_episode(
        content=f"Episode for {claim_id}",
        content_type="fact",
        scope=scope,
    )
    upsert_claim(
        claim_id=claim_id,
        claim_type="solution",
        canonical_text=canonical_text,
        payload=payload,
        valid_from="2026-01-01T00:00:00+00:00",
    )
    insert_claim_sources(claim_id, [{"source_episode_id": episode_id}])
    return episode_id


def test_outcome_creation_happy_path(tmp_data_dir):
    ensure_schema()
    cfg = get_config()

    episode_id = _seed_claim_with_episode(
        claim_id="claim-outcome-happy",
        canonical_text="Run targeted pytest for changed modules",
        payload={"problem": "slow verification", "fix": "targeted pytest", "context": "ci"},
    )
    topic_id = upsert_knowledge_topic(
        filename="outcomes.md",
        title="Outcomes",
        summary="Outcome references",
        source_episodes=[episode_id],
        fact_count=1,
        confidence=0.9,
    )
    (cfg.KNOWLEDGE_DIR / "outcomes.md").write_text("# Outcomes\n", encoding="utf-8")
    record_id = insert_knowledge_records(
        topic_id,
        [{
            "record_type": "solution",
            "content": {
                "type": "solution",
                "problem": "slow verification",
                "fix": "run targeted pytest",
                "context": "outcome tracking",
            },
            "embedding_text": "run targeted pytest",
            "confidence": 0.9,
            "valid_from": "2026-01-01T00:00:00+00:00",
        }],
        source_episodes=[episode_id],
    )[0]

    client = MemoryClient(auto_consolidate=False)
    try:
        recorded = client.record_outcome(
            action_summary="Run targeted pytest on touched modules",
            outcome_type="success",
            source_claim_ids=["claim-outcome-happy"],
            source_record_ids=[record_id],
            source_episode_ids=[episode_id],
            code_anchors=[{"anchor_type": "path", "anchor_value": "tests/test_outcome_tracking.py"}],
            issue_ids=["ISSUE-17"],
            pr_ids=["PR-901"],
            summary="Targeted validation passed with no regressions.",
            details={"duration_seconds": 34},
            provenance={"agent": "codex", "surface": "python"},
            observed_at="2026-03-10T10:00:00+00:00",
        )
        assert recorded.status == "recorded"
        assert recorded.id is not None

        browse = client.query_browse_outcomes(source_claim_id="claim-outcome-happy")
        assert browse.total == 1
        outcome = browse.outcomes[0]
        assert outcome["id"] == recorded.id
        assert outcome["outcome_type"] == "success"
        assert outcome["source_claim_ids"] == ["claim-outcome-happy"]
        assert outcome["source_record_ids"] == [record_id]
        assert outcome["source_episode_ids"] == [episode_id]
        assert outcome["issue_ids"] == ["ISSUE-17"]
        assert outcome["pr_ids"] == ["PR-901"]
        assert outcome["code_anchors"] == [
            {"anchor_type": "path", "anchor_value": "tests/test_outcome_tracking.py"}
        ]
    finally:
        client.close()


def test_multiple_outcomes_linked_to_same_claim_over_time(tmp_data_dir):
    ensure_schema()
    _seed_claim_with_episode(
        claim_id="claim-outcome-timeline",
        canonical_text="Use deterministic unit tests before broad integration runs",
        payload={"problem": "slow signal", "fix": "run deterministic unit tests", "context": "ci"},
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        first = client.record_outcome(
            action_summary="Run deterministic unit tests",
            outcome_type="failure",
            source_claim_ids=["claim-outcome-timeline"],
            observed_at="2026-03-10T10:00:00+00:00",
        )
        second = client.record_outcome(
            action_summary="Run deterministic unit tests",
            outcome_type="success",
            source_claim_ids=["claim-outcome-timeline"],
            observed_at="2026-03-11T10:00:00+00:00",
        )
        assert first.id != second.id

        browse = client.query_browse_outcomes(source_claim_id="claim-outcome-timeline")
        assert browse.total == 2
        assert [row["outcome_type"] for row in browse.outcomes] == ["success", "failure"]
    finally:
        client.close()


def test_outcome_browse_normalizes_and_validates_outcome_type(tmp_data_dir):
    ensure_schema()
    _seed_claim_with_episode(
        claim_id="claim-outcome-filter",
        canonical_text="Normalize browse outcome type tokens",
        payload={"problem": "token mismatch", "fix": "normalize literals", "context": "query"},
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        client.record_outcome(
            action_summary="Use normalized outcome tokens",
            outcome_type="success",
            source_claim_ids=["claim-outcome-filter"],
        )

        filtered = client.query_browse_outcomes(
            source_claim_id="claim-outcome-filter",
            outcome_type=" SUCCESS ",
        )
        assert filtered.total == 1
        assert filtered.outcome_type == "success"

        unfiltered = client.query_browse_outcomes(
            source_claim_id="claim-outcome-filter",
            outcome_type="   ",
        )
        assert unfiltered.total == 1
        assert unfiltered.outcome_type is None

        with pytest.raises(ValueError, match="outcome_type must be one of"):
            client.query_browse_outcomes(outcome_type="unknown")
    finally:
        client.close()


def test_outcome_write_rolls_back_on_insert_failure(tmp_data_dir):
    ensure_schema()

    with patch("consolidation_memory.database.uuid.uuid4", side_effect=["outcome-rollback", "source-rollback"]):
        try:
            record_action_outcome(
                action_summary="Attempt writes with bad source linkage",
                outcome_type="failure",
                source_claim_ids=["missing-claim"],
            )
            raise AssertionError("Expected sqlite3.IntegrityError")
        except sqlite3.IntegrityError:
            pass

    assert get_action_outcomes(limit=10) == []


def test_outcome_export_import_round_trip(tmp_data_dir):
    ensure_schema()
    claim_episode_id = _seed_claim_with_episode(
        claim_id="claim-outcome-export",
        canonical_text="Use narrow regression slices before full suite",
        payload={"problem": "slow feedback", "fix": "targeted regressions", "context": "verification"},
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        client.record_outcome(
            action_summary="Run targeted regression slice",
            outcome_type="partial_success",
            source_claim_ids=["claim-outcome-export"],
            source_episode_ids=[claim_episode_id],
            issue_ids=["ISSUE-42"],
            observed_at="2026-03-12T10:00:00+00:00",
        )
        export_result = client.export()
    finally:
        client.close()

    exported = json.loads(open(export_result.path, encoding="utf-8").read())
    assert len(exported["action_outcomes"]) == 1
    assert len(exported["action_outcome_sources"]) >= 1
    assert len(exported["action_outcome_refs"]) == 1

    cfg = get_config()
    close_all_connections()
    if cfg.DB_PATH.exists():
        cfg.DB_PATH.unlink()
    for path_attr in (
        "FAISS_INDEX_PATH",
        "FAISS_IDS_PATH",
        "FAISS_TOMBSTONES_PATH",
        "FAISS_RELOAD_SIGNAL_PATH",
    ):
        path = getattr(cfg, path_attr, None)
        if path is not None and path.exists():
            path.unlink()

    with patch(
        "consolidation_memory.backends.encode_documents",
        return_value=np.vstack([_make_normalized_vec(seed=13)]),
    ):
        cmd_import(export_result.path)

    imported_outcomes = get_all_action_outcomes()
    imported_sources = get_all_action_outcome_sources()
    imported_refs = get_all_action_outcome_refs()
    assert len(imported_outcomes) == 1
    assert len(imported_sources) >= 1
    assert len(imported_refs) == 1
    assert imported_outcomes[0]["outcome_type"] == "partial_success"


def test_outcome_scope_visibility_enforced(tmp_data_dir):
    ensure_schema()
    visible_scope = {
        "namespace_slug": "default",
        "project_slug": "default",
        "app_client_name": "legacy_client",
        "app_client_type": "python_sdk",
    }
    hidden_scope = {
        "namespace_slug": "default",
        "project_slug": "default",
        "app_client_name": "other-app",
        "app_client_type": "rest",
    }
    _seed_claim_with_episode(
        claim_id="claim-outcome-visible",
        canonical_text="Visible scoped claim",
        payload={"subject": "visible", "info": "scope"},
        scope=visible_scope,
    )
    _seed_claim_with_episode(
        claim_id="claim-outcome-hidden",
        canonical_text="Hidden scoped claim",
        payload={"subject": "hidden", "info": "scope"},
        scope=hidden_scope,
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        visible_outcome = client.record_outcome(
            action_summary="Visible scoped strategy",
            outcome_type="success",
            source_claim_ids=["claim-outcome-visible"],
            scope={"project": {"slug": "default"}},
        )
        hidden_outcome = client.record_outcome(
            action_summary="Hidden scoped strategy",
            outcome_type="failure",
            source_claim_ids=["claim-outcome-hidden"],
            scope={"project": {"slug": "default"}, "app_client": {"name": "other-app", "app_type": "rest"}},
        )

        visible_results = client.query_browse_outcomes(
            scope={"project": {"slug": "default"}},
        )
        visible_ids = {str(row["id"]) for row in visible_results.outcomes}
        assert visible_outcome.id in visible_ids
        assert hidden_outcome.id not in visible_ids
    finally:
        client.close()
