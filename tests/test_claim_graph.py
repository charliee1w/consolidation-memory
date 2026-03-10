"""Tests for claim graph database layer."""

import json

from consolidation_memory.database import (
    ensure_schema,
    expire_claim,
    get_claims_as_of,
    get_claims_by_anchor,
    get_connection,
    insert_claim_edge,
    insert_claim_event,
    insert_claim_sources,
    insert_episode,
    insert_episode_anchors,
    mark_claims_challenged_by_ids,
    mark_claims_challenged_by_anchors,
    upsert_claim,
)


class TestClaimGraphMigration:
    def test_migration_creates_claim_graph_tables(self, tmp_data_dir):
        ensure_schema()
        expected = {"claims", "claim_edges", "claim_sources", "claim_events", "episode_anchors"}
        with get_connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {row["name"] for row in rows}
        assert expected.issubset(names)


class TestClaimGraphMethods:
    def test_claim_upsert_idempotency(self, tmp_data_dir):
        ensure_schema()
        claim_id = "claim-upsert-1"
        upsert_claim(
            claim_id=claim_id,
            claim_type="fact",
            canonical_text="Python version",
            payload={"value": "3.12"},
            confidence=0.7,
            valid_from="2026-01-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id=claim_id,
            claim_type="fact",
            canonical_text="Python version",
            payload={"value": "3.13"},
            confidence=0.9,
            valid_from="2026-01-01T00:00:00+00:00",
        )

        with get_connection() as conn:
            row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM claims WHERE id = ?", (claim_id,),
            ).fetchone()["c"]

        assert row is not None
        assert count == 1
        assert json.loads(row["payload"]) == {"value": "3.13"}
        assert row["confidence"] == 0.9

    def test_temporal_as_of_claim_retrieval(self, tmp_data_dir):
        ensure_schema()
        upsert_claim(
            claim_id="claim-active",
            claim_type="fact",
            canonical_text="active claim",
            payload={"k": "v"},
            valid_from="2026-01-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-future",
            claim_type="fact",
            canonical_text="future claim",
            payload={"k": "future"},
            valid_from="2026-01-03T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-expired",
            claim_type="fact",
            canonical_text="expired claim",
            payload={"k": "old"},
            valid_from="2025-12-01T00:00:00+00:00",
            valid_until="2026-01-01T12:00:00+00:00",
        )

        at_jan2 = get_claims_as_of("2026-01-02T00:00:00+00:00", claim_type="fact")
        jan2_ids = {row["id"] for row in at_jan2}
        assert "claim-active" in jan2_ids
        assert "claim-future" not in jan2_ids
        assert "claim-expired" not in jan2_ids

        at_dec15 = get_claims_as_of("2025-12-15T00:00:00+00:00", claim_type="fact")
        dec15_ids = {row["id"] for row in at_dec15}
        assert "claim-expired" in dec15_ids

    def test_claims_as_of_treats_equivalent_offset_instants_equally(self, tmp_data_dir):
        ensure_schema()
        upsert_claim(
            claim_id="claim-offset",
            claim_type="fact",
            canonical_text="offset claim",
            payload={"k": "v"},
            valid_from="2026-01-01T00:00:00+00:00",
        )

        rows = get_claims_as_of("2025-12-31T19:00:00-05:00", claim_type="fact")
        assert {row["id"] for row in rows} == {"claim-offset"}

    def test_claims_as_of_restores_active_status_before_challenge(self, tmp_data_dir):
        ensure_schema()
        episode_id = insert_episode("claim history episode")
        insert_episode_anchors(
            episode_id,
            [{"anchor_type": "path", "anchor_value": "src/history.py"}],
        )
        upsert_claim(
            claim_id="claim-history",
            claim_type="fact",
            canonical_text="history claim",
            payload={"k": "v"},
            valid_from="2026-01-01T00:00:00+00:00",
        )
        insert_claim_sources("claim-history", [{"source_episode_id": episode_id}])

        challenged_ids = mark_claims_challenged_by_anchors(
            [{"anchor_type": "path", "anchor_value": "src/history.py"}],
            challenged_at="2026-02-01T00:00:00+00:00",
        )

        assert challenged_ids == ["claim-history"]

        before = {
            row["id"]: row
            for row in get_claims_as_of("2026-01-15T00:00:00+00:00", claim_type="fact")
        }
        after = {
            row["id"]: row
            for row in get_claims_as_of("2026-02-02T00:00:00+00:00", claim_type="fact")
        }

        assert before["claim-history"]["status"] == "active"
        assert after["claim-history"]["status"] == "challenged"

    def test_mark_claims_challenged_by_anchors_skips_future_claims(self, tmp_data_dir):
        ensure_schema()
        episode_id = insert_episode("future claim episode")
        insert_episode_anchors(
            episode_id,
            [{"anchor_type": "path", "anchor_value": "src/future.py"}],
        )
        upsert_claim(
            claim_id="claim-future-anchor",
            claim_type="fact",
            canonical_text="future claim",
            payload={"k": "future"},
            valid_from="2026-03-01T00:00:00+00:00",
        )
        insert_claim_sources("claim-future-anchor", [{"source_episode_id": episode_id}])

        challenged_ids = mark_claims_challenged_by_anchors(
            [{"anchor_type": "path", "anchor_value": "src/future.py"}],
            challenged_at="2026-02-01T00:00:00+00:00",
        )

        assert challenged_ids == []
        with get_connection() as conn:
            row = conn.execute(
                "SELECT status FROM claims WHERE id = ?",
                ("claim-future-anchor",),
            ).fetchone()
        assert row is not None
        assert row["status"] == "active"

    def test_mark_claims_challenged_by_ids_updates_active_only(self, tmp_data_dir):
        ensure_schema()
        upsert_claim(
            claim_id="claim-active-id",
            claim_type="fact",
            canonical_text="active claim",
            payload={"k": "active"},
            status="active",
            valid_from="2026-01-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-future-id",
            claim_type="fact",
            canonical_text="future claim",
            payload={"k": "future"},
            status="active",
            valid_from="2026-03-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-already-challenged-id",
            claim_type="fact",
            canonical_text="challenged claim",
            payload={"k": "challenged"},
            status="challenged",
            valid_from="2026-01-01T00:00:00+00:00",
        )

        challenged_ids = mark_claims_challenged_by_ids(
            ["claim-active-id", "claim-future-id", "claim-already-challenged-id"],
            challenged_at="2026-02-01T00:00:00+00:00",
        )

        assert challenged_ids == ["claim-active-id"]
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, status FROM claims WHERE id IN (?, ?, ?) ORDER BY id",
                ("claim-active-id", "claim-future-id", "claim-already-challenged-id"),
            ).fetchall()
            challenged_events = conn.execute(
                "SELECT claim_id FROM claim_events WHERE event_type = 'challenged' ORDER BY claim_id",
            ).fetchall()
        status_by_id = {row["id"]: row["status"] for row in rows}
        assert status_by_id["claim-active-id"] == "challenged"
        assert status_by_id["claim-future-id"] == "active"
        assert status_by_id["claim-already-challenged-id"] == "challenged"
        assert [row["claim_id"] for row in challenged_events] == ["claim-active-id"]

    def test_claim_edge_insert_read(self, tmp_data_dir):
        ensure_schema()
        upsert_claim("claim-a", "fact", "claim a", payload={"a": 1}, valid_from="2026-01-01T00:00:00+00:00")
        upsert_claim("claim-b", "fact", "claim b", payload={"b": 1}, valid_from="2026-01-01T00:00:00+00:00")

        edge_id = insert_claim_edge(
            from_claim_id="claim-a",
            to_claim_id="claim-b",
            edge_type="contradicts",
            confidence=0.88,
            details={"reason": "new evidence"},
        )

        with get_connection() as conn:
            row = conn.execute("SELECT * FROM claim_edges WHERE id = ?", (edge_id,)).fetchone()
        assert row is not None
        assert row["from_claim_id"] == "claim-a"
        assert row["to_claim_id"] == "claim-b"
        assert row["edge_type"] == "contradicts"
        assert row["confidence"] == 0.88
        assert json.loads(row["details"]) == {"reason": "new evidence"}

    def test_claim_source_insert_read(self, tmp_data_dir):
        ensure_schema()
        episode_id = insert_episode("source episode")
        upsert_claim(
            claim_id="claim-source",
            claim_type="solution",
            canonical_text="source claim",
            payload={"k": "v"},
            valid_from="2026-01-01T00:00:00+00:00",
        )

        source_ids = insert_claim_sources(
            "claim-source",
            [{
                "source_episode_id": episode_id,
                "source_topic_id": "topic-1",
                "source_record_id": "record-1",
            }],
        )
        assert len(source_ids) == 1

        with get_connection() as conn:
            row = conn.execute("SELECT * FROM claim_sources WHERE id = ?", (source_ids[0],)).fetchone()
        assert row is not None
        assert row["claim_id"] == "claim-source"
        assert row["source_episode_id"] == episode_id
        assert row["source_topic_id"] == "topic-1"
        assert row["source_record_id"] == "record-1"

    def test_claim_event_insert_read(self, tmp_data_dir):
        ensure_schema()
        upsert_claim(
            claim_id="claim-event",
            claim_type="fact",
            canonical_text="event claim",
            payload={"k": "v"},
            valid_from="2026-01-01T00:00:00+00:00",
        )

        event_id = insert_claim_event(
            claim_id="claim-event",
            event_type="created",
            details={"origin": "test"},
        )

        with get_connection() as conn:
            row = conn.execute("SELECT * FROM claim_events WHERE id = ?", (event_id,)).fetchone()
        assert row is not None
        assert row["claim_id"] == "claim-event"
        assert row["event_type"] == "created"
        assert json.loads(row["details"]) == {"origin": "test"}

    def test_anchor_insert_and_claim_lookup(self, tmp_data_dir):
        ensure_schema()
        episode_id = insert_episode("anchor episode")
        inserted = insert_episode_anchors(
            episode_id,
            [
                {"anchor_type": "path", "anchor_value": "src/main.py"},
                {"type": "path", "value": "src/main.py"},  # duplicate semantic anchor
                {"anchor_type": "commit", "anchor_value": "abcdef1"},
            ],
        )
        # Duplicate path should be ignored.
        assert len(inserted) == 2

        upsert_claim(
            claim_id="claim-anchor",
            claim_type="solution",
            canonical_text="anchor-linked claim",
            payload={"fix": "update parser"},
            valid_from="2026-01-01T00:00:00+00:00",
        )
        insert_claim_sources("claim-anchor", [{"source_episode_id": episode_id}])

        by_anchor = get_claims_by_anchor(anchor_type="path", anchor_value="src/main.py")
        by_anchor_ids = {row["id"] for row in by_anchor}
        assert "claim-anchor" in by_anchor_ids

        # Expired claims are excluded from default (include_expired=False) lookups.
        assert expire_claim("claim-anchor", valid_until="2026-01-02T00:00:00+00:00") is True
        by_anchor_after_expire = get_claims_by_anchor(
            anchor_type="path", anchor_value="src/main.py",
        )
        assert {row["id"] for row in by_anchor_after_expire} == set()
