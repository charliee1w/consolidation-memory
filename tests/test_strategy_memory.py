"""Regression tests for first-class strategy memory behavior."""

from __future__ import annotations

import json
from pathlib import Path

from consolidation_memory.client import MemoryClient
from consolidation_memory.database import (
    ensure_schema,
    get_claim_outcome_evidence,
    insert_claim_event,
    insert_episode_anchors,
    insert_claim_sources,
    insert_episode,
    insert_knowledge_records,
    upsert_claim,
    upsert_knowledge_topic,
)


def _seed_strategy(
    *,
    strategy_id: str,
    problem_pattern: str,
    strategy_text: str,
    scope: dict[str, str] | None = None,
) -> dict[str, str]:
    episode_id = insert_episode(
        content=f"Strategy seed: {problem_pattern}",
        content_type="fact",
        scope=scope,
    )
    topic_id = upsert_knowledge_topic(
        filename=f"{strategy_id}.md",
        title=f"{strategy_id} strategy",
        summary=strategy_text,
        source_episodes=[episode_id],
        fact_count=1,
        confidence=0.9,
        scope=scope,
    )
    record_payload = {
        "type": "strategy",
        "problem_pattern": problem_pattern,
        "strategy": strategy_text,
        "preconditions": "reproducible failure",
        "expected_signals": "same failure reproduces in isolation",
        "failure_modes": "infrastructure-only outage",
    }
    record_id = insert_knowledge_records(
        topic_id,
        [{
            "record_type": "strategy",
            "content": record_payload,
            "embedding_text": (
                f"Strategy for {problem_pattern}: {strategy_text}. "
                "Preconditions: reproducible failure."
            ),
            "confidence": 0.9,
            "valid_from": "2026-01-01T00:00:00+00:00",
        }],
        source_episodes=[episode_id],
        scope=scope,
    )[0]

    claim_id = f"claim-{strategy_id}"
    upsert_claim(
        claim_id=claim_id,
        claim_type="strategy",
        canonical_text=f"strategy | pattern={problem_pattern} | approach={strategy_text}",
        payload=record_payload,
        confidence=0.9,
        valid_from="2026-01-01T00:00:00+00:00",
    )
    insert_claim_sources(
        claim_id,
        [{
            "source_episode_id": episode_id,
            "source_topic_id": topic_id,
            "source_record_id": record_id,
        }],
    )
    return {
        "episode_id": episode_id,
        "topic_id": topic_id,
        "record_id": record_id,
        "claim_id": claim_id,
    }


def test_create_and_retrieve_strategy_claims(tmp_data_dir):
    ensure_schema()
    seeded = _seed_strategy(
        strategy_id="flaky-ci",
        problem_pattern="flaky ci tests",
        strategy_text="rerun deterministically and isolate fixture state",
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        result = client.query_search_claims(
            query="flaky ci tests",
            claim_type="strategy",
            limit=10,
        )
    finally:
        client.close()

    assert any(claim["id"] == seeded["claim_id"] for claim in result.claims)
    strategy_claim = next(claim for claim in result.claims if claim["id"] == seeded["claim_id"])
    assert strategy_claim["claim_type"] == "strategy"
    assert strategy_claim["payload"]["problem_pattern"] == "flaky ci tests"
    assert strategy_claim["strategy_evidence"]["validation_count"] == 0


def test_supporting_outcomes_increase_strategy_evidence_density(tmp_data_dir):
    ensure_schema()
    seeded = _seed_strategy(
        strategy_id="targeted-tests",
        problem_pattern="slow feedback loops",
        strategy_text="run narrow regression slice before full suite",
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        client.record_outcome(
            action_summary="run narrow regression slice",
            outcome_type="success",
            source_claim_ids=[seeded["claim_id"]],
            observed_at="2026-03-10T10:00:00+00:00",
        )
        first = client.query_browse_claims(claim_type="strategy", limit=10)
        first_claim = next(claim for claim in first.claims if claim["id"] == seeded["claim_id"])
        first_density = first_claim["strategy_evidence"]["evidence_density"]

        client.record_outcome(
            action_summary="run narrow regression slice",
            outcome_type="success",
            source_claim_ids=[seeded["claim_id"]],
            observed_at="2026-03-11T10:00:00+00:00",
        )
        second = client.query_browse_claims(claim_type="strategy", limit=10)
        second_claim = next(claim for claim in second.claims if claim["id"] == seeded["claim_id"])
        second_density = second_claim["strategy_evidence"]["evidence_density"]
    finally:
        client.close()

    assert second_density > first_density
    evidence = get_claim_outcome_evidence([seeded["claim_id"]])[seeded["claim_id"]]
    assert evidence["validation_count"] == 2
    assert evidence["success_count"] == 2


def test_strategy_evidence_exposes_reliability_inputs_for_debugging(tmp_data_dir):
    ensure_schema()
    seeded = _seed_strategy(
        strategy_id="evidence-inputs",
        problem_pattern="flaky ci tests",
        strategy_text="rerun deterministically then isolate side effects",
    )
    insert_episode_anchors(
        seeded["episode_id"],
        [{"anchor_type": "path", "anchor_value": "src/service.py"}],
    )
    insert_claim_event(
        seeded["claim_id"],
        event_type="code_drift_detected",
        details={"reason": "source file changed"},
        created_at="2026-03-12T10:00:00+00:00",
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        client.record_outcome(
            action_summary="strategy validation run",
            outcome_type="success",
            source_claim_ids=[seeded["claim_id"]],
            code_anchors=[{"anchor_type": "path", "anchor_value": "src/service.py"}],
            provenance={"agent": "codex"},
            observed_at="2026-03-12T11:00:00+00:00",
        )
        client.record_outcome(
            action_summary="strategy rollback run",
            outcome_type="reverted",
            source_claim_ids=[seeded["claim_id"]],
            observed_at="2026-03-12T12:00:00+00:00",
        )
    finally:
        client.close()

    evidence = get_claim_outcome_evidence([seeded["claim_id"]])[seeded["claim_id"]]
    assert evidence["validation_count"] == 2
    assert evidence["reverted_count"] == 1
    assert evidence["drift_event_count"] == 1
    assert evidence["source_link_count"] >= 1
    assert evidence["source_anchor_count"] >= 1
    assert evidence["outcome_anchor_count"] >= 1
    assert evidence["outcomes_with_provenance_count"] >= 1


def test_failed_or_contradicted_strategies_are_still_inspectable_but_rank_lower(tmp_data_dir):
    ensure_schema()
    validated = _seed_strategy(
        strategy_id="validated",
        problem_pattern="flaky ci tests",
        strategy_text="rerun deterministically then isolate side effects",
    )
    degraded = _seed_strategy(
        strategy_id="degraded",
        problem_pattern="flaky ci tests",
        strategy_text="rerun deterministically then isolate side effects",
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        client.record_outcome(
            action_summary="validated strategy run",
            outcome_type="success",
            source_claim_ids=[validated["claim_id"]],
        )
        client.record_outcome(
            action_summary="validated strategy run",
            outcome_type="success",
            source_claim_ids=[validated["claim_id"]],
        )
        client.record_outcome(
            action_summary="degraded strategy run",
            outcome_type="failure",
            source_claim_ids=[degraded["claim_id"]],
        )
        client.record_outcome(
            action_summary="degraded strategy run",
            outcome_type="reverted",
            source_claim_ids=[degraded["claim_id"]],
        )
        insert_claim_event(
            degraded["claim_id"],
            event_type="contradiction",
            details={"reason": "new evidence disproved assumptions"},
            created_at="2026-03-12T10:00:00+00:00",
        )

        result = client.query_search_claims(
            query="flaky ci tests",
            claim_type="strategy",
            limit=10,
        )
    finally:
        client.close()

    ranked_ids = [claim["id"] for claim in result.claims]
    assert validated["claim_id"] in ranked_ids
    assert degraded["claim_id"] in ranked_ids
    assert ranked_ids.index(validated["claim_id"]) < ranked_ids.index(degraded["claim_id"])

    validated_claim = next(claim for claim in result.claims if claim["id"] == validated["claim_id"])
    degraded_claim = next(claim for claim in result.claims if claim["id"] == degraded["claim_id"])
    assert validated_claim["strategy_evidence"]["reuse_multiplier"] > degraded_claim["strategy_evidence"]["reuse_multiplier"]


def test_strategy_scope_filtering_and_export_visibility(tmp_data_dir):
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
    visible = _seed_strategy(
        strategy_id="visible-scope",
        problem_pattern="db deadlocks",
        strategy_text="capture lock graph and replay deterministic transaction order",
        scope=visible_scope,
    )
    hidden = _seed_strategy(
        strategy_id="hidden-scope",
        problem_pattern="db deadlocks",
        strategy_text="capture lock graph and replay deterministic transaction order",
        scope=hidden_scope,
    )

    client = MemoryClient(auto_consolidate=False)
    try:
        visible_outcome = client.record_outcome(
            action_summary="visible scope strategy run",
            outcome_type="success",
            source_claim_ids=[visible["claim_id"]],
            scope={"project": {"slug": "default"}},
        )
        hidden_outcome = client.record_outcome(
            action_summary="hidden scope strategy run",
            outcome_type="failure",
            source_claim_ids=[hidden["claim_id"]],
            scope={"project": {"slug": "default"}, "app_client": {"name": "other-app", "app_type": "rest"}},
        )
        search = client.query_search_claims(
            query="deadlocks",
            claim_type="strategy",
            limit=10,
        )
        export_result = client.export()
    finally:
        client.close()

    assert {claim["id"] for claim in search.claims} == {visible["claim_id"]}

    export_data = json.loads(Path(export_result.path).read_text(encoding="utf-8"))
    assert {record["id"] for record in export_data["knowledge_records"]} == {visible["record_id"]}
    assert {claim["id"] for claim in export_data["claims"]} == {visible["claim_id"]}
    assert {outcome["id"] for outcome in export_data["action_outcomes"]} == {visible_outcome.id}
    assert hidden_outcome.id not in {outcome["id"] for outcome in export_data["action_outcomes"]}
