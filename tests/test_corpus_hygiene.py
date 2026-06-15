"""Tests for corpus hygiene scan and apply."""

from __future__ import annotations

import json
from unittest.mock import patch

from consolidation_memory.corpus_hygiene import (
    apply_corpus_hygiene,
    classify_episode_candidates,
    repair_orphaned_claims,
    scan_corpus_hygiene,
)
from consolidation_memory.database import (
    ensure_schema,
    get_connection,
    insert_claim_sources,
    soft_delete_episode,
    upsert_claim,
)
from tests.helpers import make_normalized_vec as _vec


class TestClassifyEpisodeCandidates:
    def test_classifies_temp_exchange_and_noise(self):
        episodes = [
            {
                "id": "ep-temp",
                "content_type": "fact",
                "content": "coding_agent_eval_test probe",
                "tags": "[]",
            },
            {
                "id": "ep-exchange",
                "content_type": "exchange",
                "content": "user asked something",
                "tags": "[]",
            },
            {
                "id": "ep-noise",
                "content_type": "solution",
                "content": "Completed multi-agent hardening pass for consolidation-memory.",
                "tags": "[]",
            },
            {
                "id": "ep-keep",
                "content_type": "solution",
                "content": "Problem: USB link stuck at USB2. Fix: reset hub and OVRService.",
                "tags": "[]",
            },
        ]
        buckets = classify_episode_candidates(episodes)
        assert buckets["temp_ids"] == ["ep-temp"]
        assert buckets["exchange_ids"] == ["ep-exchange"]
        assert buckets["noise_journal_ids"] == ["ep-noise"]
        assert buckets["recommended_cleanup_ids"] == ["ep-exchange", "ep-noise", "ep-temp"]


class TestCorpusHygieneScan:
    @patch("consolidation_memory.backends.encode_documents")
    def test_scan_reports_stored_noise(self, mock_embed):
        from consolidation_memory.client import MemoryClient

        mock_embed.side_effect = lambda texts: _vec(seed=hash(texts[0]) % 997).reshape(1, -1)
        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            client.store("coding_agent_eval_test persistence probe", content_type="fact")
            client.store("Completed item for roadmap.", content_type="solution")
            client.store("General user exchange", content_type="exchange")
        finally:
            client.close()

        report = scan_corpus_hygiene()
        episodes = report["episodes"]
        assert episodes["total_active"] >= 3
        assert episodes["temp"]["count"] >= 1
        assert episodes["exchange"]["count"] >= 1
        assert episodes["noise_journal"]["count"] >= 1
        assert len(episodes["recommended_cleanup_ids"]) >= 3


class TestCorpusHygieneApply:
    @patch("consolidation_memory.backends.encode_documents")
    def test_apply_forgets_and_expires_episode_only_claim(self, mock_embed):
        from consolidation_memory.client import MemoryClient

        mock_embed.return_value = _vec(seed=13).reshape(1, -1)
        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            stored = client.store("coding_agent_eval_test cleanup target", content_type="fact")
        finally:
            client.close()
        upsert_claim(
            claim_id="claim-hygiene-only-source",
            claim_type="fact",
            canonical_text="episode-only hygiene claim",
            payload={"subject": "hygiene", "info": "episode only"},
            valid_from="2026-01-01T00:00:00+00:00",
        )
        insert_claim_sources(
            "claim-hygiene-only-source",
            [{"source_episode_id": stored.id}],
        )

        dry = apply_corpus_hygiene([stored.id], dry_run=True)
        assert dry["status"] == "dry_run"
        assert dry["episode_targets"] == 1

        result = apply_corpus_hygiene([stored.id], dry_run=False)
        assert result["forgotten"] == 1

        with get_connection() as conn:
            episode_row = conn.execute(
                "SELECT deleted FROM episodes WHERE id = ?",
                (stored.id,),
            ).fetchone()
            claim_row = conn.execute(
                "SELECT status FROM claims WHERE id = ?",
                ("claim-hygiene-only-source",),
            ).fetchone()

        assert episode_row is not None
        assert episode_row["deleted"] == 1
        assert claim_row is not None
        assert claim_row["status"] == "expired"

    @patch("consolidation_memory.backends.encode_documents")
    def test_repair_orphaned_claims_after_manual_delete(self, mock_embed):
        from consolidation_memory.client import MemoryClient

        mock_embed.return_value = _vec(seed=14).reshape(1, -1)
        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            stored = client.store("orphan repair target", content_type="fact")
        finally:
            client.close()
        upsert_claim(
            claim_id="claim-historical-orphan",
            claim_type="fact",
            canonical_text="historical orphan claim",
            payload={"subject": "orphan", "info": "stale provenance"},
            valid_from="2026-01-01T00:00:00+00:00",
        )
        insert_claim_sources(
            "claim-historical-orphan",
            [{"source_episode_id": stored.id}],
        )

        with get_connection():
            soft_delete_episode(stored.id, scope=None)

        scan = scan_corpus_hygiene()
        assert scan["orphaned_claims"]["count"] >= 1
        assert "claim-historical-orphan" in scan["orphaned_claims"]["ids"]

        repair = repair_orphaned_claims(dry_run=False)
        assert repair["status"] == "ok"
        assert repair["expired_claims"] >= 1

        with get_connection() as conn:
            claim_row = conn.execute(
                "SELECT status FROM claims WHERE id = ?",
                ("claim-historical-orphan",),
            ).fetchone()
            event_row = conn.execute(
                """SELECT event_type, details
                     FROM claim_events
                    WHERE claim_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1""",
                ("claim-historical-orphan",),
            ).fetchone()

        assert claim_row is not None
        assert claim_row["status"] == "expired"
        assert event_row is not None
        assert event_row["event_type"] == "expire"
        assert json.loads(event_row["details"])["reason"] == "orphan_repair"