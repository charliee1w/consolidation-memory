"""Tests for consolidation changelog (Phase 3.2).

The memory_consolidation_log tool provides human-readable summaries of
recent consolidation activity.

Run with: python -m pytest tests/test_consolidation_log.py -v
"""

from consolidation_memory.types import ConsolidationLogResult


class TestConsolidationLogResult:
    def test_default_values(self):
        r = ConsolidationLogResult()
        assert r.entries == []
        assert r.total == 0
        assert r.message == ""


class TestConsolidationLogClient:
    """Test the client.consolidation_log() method."""

    def test_no_runs_returns_message(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient()
        result = client.consolidation_log()
        assert result.total == 0
        assert "No consolidation runs" in result.message

    def test_returns_runs(self, tmp_data_dir):
        from consolidation_memory.database import (
            ensure_schema,
            start_consolidation_run,
            complete_consolidation_run,
        )
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        # Create a completed run
        run_id = start_consolidation_run()
        complete_consolidation_run(
            run_id,
            episodes_processed=10,
            clusters_formed=3,
            topics_created=2,
            topics_updated=1,
            episodes_pruned=5,
        )

        client = MemoryClient()
        result = client.consolidation_log()

        assert result.total == 1
        entry = result.entries[0]
        assert entry["status"] == "completed"
        assert entry["topics_created"] == 2
        assert entry["topics_updated"] == 1
        assert entry["episodes_pruned"] == 5
        assert "2 topics" in entry["summary"]
        assert "1 topic" in entry["summary"]
        assert "5 episodes" in entry["summary"]

    def test_failed_run_summary(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema, get_connection

        ensure_schema()

        # Insert a failed run directly
        import uuid
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO consolidation_runs
                   (id, started_at, completed_at, status, error_message)
                   VALUES (?, ?, ?, 'failed', 'LLM backend timeout')""",
                (str(uuid.uuid4()), now, now),
            )

        from consolidation_memory.client import MemoryClient
        client = MemoryClient()
        result = client.consolidation_log()

        assert result.total == 1
        assert "FAILED" in result.entries[0]["summary"]
        assert "timeout" in result.entries[0]["summary"].lower()

    def test_last_n_clamped(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient()

        # Should clamp to 1
        result = client.consolidation_log(last_n=-5)
        assert result.total == 0  # no runs, but no crash

        # Should clamp to 20
        result = client.consolidation_log(last_n=100)
        assert result.total == 0

    def test_run_with_no_changes(self, tmp_data_dir):
        from consolidation_memory.database import (
            ensure_schema,
            start_consolidation_run,
            complete_consolidation_run,
        )
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        run_id = start_consolidation_run()
        complete_consolidation_run(
            run_id,
            episodes_processed=5,
            clusters_formed=0,
            topics_created=0,
            topics_updated=0,
            episodes_pruned=0,
        )

        client = MemoryClient()
        result = client.consolidation_log()

        assert result.total == 1
        # Should mention processing even when no topics changed
        assert "5 episodes" in result.entries[0]["summary"].lower()


class TestConsolidationLogContradictions:
    """Test contradiction cross-referencing in consolidation log."""

    def test_contradictions_counted_within_run_window(self, tmp_data_dir):
        """Contradictions detected during a run's time window should be counted."""
        import time
        import uuid
        from consolidation_memory.database import (
            ensure_schema,
            start_consolidation_run,
            insert_contradiction,
            complete_consolidation_run,
        )
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        # Create a run with a contradiction inserted during the run
        run_id = start_consolidation_run()
        time.sleep(0.05)

        insert_contradiction(
            topic_id=str(uuid.uuid4()),
            old_record_id=None,
            new_record_id=None,
            old_content="old fact",
            new_content="new fact",
            resolution="expired_old",
        )

        time.sleep(0.05)
        complete_consolidation_run(
            run_id,
            episodes_processed=5,
            topics_created=1,
            topics_updated=0,
        )

        client = MemoryClient()
        result = client.consolidation_log()

        assert result.total == 1
        assert result.entries[0]["contradictions_detected"] == 1

    def test_contradictions_outside_window_not_counted(self, tmp_data_dir):
        """Contradictions outside a run's time window should not be counted."""
        import time
        import uuid
        from consolidation_memory.database import (
            ensure_schema,
            start_consolidation_run,
            insert_contradiction,
            complete_consolidation_run,
        )
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        # Insert contradiction BEFORE the run starts
        insert_contradiction(
            topic_id=str(uuid.uuid4()),
            old_record_id=None,
            new_record_id=None,
            old_content="old fact",
            new_content="new fact",
            resolution="expired_old",
        )

        time.sleep(0.05)

        run_id = start_consolidation_run()
        time.sleep(0.05)
        complete_consolidation_run(
            run_id,
            episodes_processed=5,
            topics_created=1,
            topics_updated=0,
        )

        client = MemoryClient()
        result = client.consolidation_log()

        assert result.total == 1
        assert result.entries[0]["contradictions_detected"] == 0

    def test_running_status_summary(self, tmp_data_dir):
        """A run with status 'running' should say 'In progress'."""
        from consolidation_memory.database import ensure_schema, start_consolidation_run
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        # start_consolidation_run inserts a row with status='running'
        start_consolidation_run()

        client = MemoryClient()
        result = client.consolidation_log()

        assert result.total == 1
        assert result.entries[0]["status"] == "running"
        assert result.entries[0]["summary"] == "In progress"

    def test_stale_running_status_is_recovered_to_failed(self, tmp_data_dir):
        """A stale running run should be auto-recovered and no longer show in-progress."""
        from datetime import datetime, timedelta, timezone
        from consolidation_memory.database import (
            ensure_schema,
            get_connection,
            start_consolidation_run,
        )
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        run_id = start_consolidation_run()
        stale_started = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "UPDATE consolidation_runs SET started_at = ? WHERE id = ?",
                (stale_started, run_id),
            )

        client = MemoryClient()
        result = client.consolidation_log()
        entry = result.entries[0]
        assert entry["status"] == "failed"
        assert "FAILED" in entry["summary"]


class TestConsolidationLogSchema:
    """Test the OpenAI schema and dispatch."""

    def test_schema_exists(self):
        from consolidation_memory.schemas import MEMORY_CONSOLIDATION_LOG_SCHEMA
        assert MEMORY_CONSOLIDATION_LOG_SCHEMA["function"]["name"] == "memory_consolidation_log"

    def test_in_tools_list(self):
        from consolidation_memory.schemas import openai_tools
        names = [t["function"]["name"] for t in openai_tools]
        assert "memory_consolidation_log" in names

    def test_dispatch(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.schemas import dispatch_tool_call

        ensure_schema()
        client = MemoryClient()
        result = dispatch_tool_call(client, "memory_consolidation_log", {"last_n": 3})
        assert "entries" in result
        assert "total" in result
