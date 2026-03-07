"""Tests for CLI commands, particularly `consolidation-memory test`."""

import json

import pytest
from unittest.mock import patch, MagicMock

from consolidation_memory.database import ensure_schema
from tests.helpers import make_normalized_vec as _make_normalized_vec


class TestCmdTest:
    """Tests for the `consolidation-memory test` CLI subcommand."""

    def test_all_checks_pass(self, capsys):
        """Happy path: all steps succeed, exit normally (no sys.exit)."""
        ensure_schema()
        vec = _make_normalized_vec()

        mock_backend = MagicMock()
        mock_backend.encode_documents.return_value = vec.reshape(1, -1)
        mock_backend.encode_query.return_value = vec.reshape(1, -1)
        mock_backend.dimension = 384

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "OK"

        with (
            patch("consolidation_memory.backends.get_embedding_backend", return_value=mock_backend),
            patch("consolidation_memory.backends.get_llm_backend", return_value=mock_llm),
        ):
            from consolidation_memory.cli import cmd_test
            cmd_test()

        captured = capsys.readouterr()
        assert "checks passed" in captured.out
        # All checks should pass — no X marks
        assert "\u2717" not in captured.out

    def test_all_checks_pass_llm_disabled(self, capsys):
        """When LLM is disabled, it should be skipped (not failed)."""
        from consolidation_memory.config import override_config

        ensure_schema()
        vec = _make_normalized_vec()

        mock_backend = MagicMock()
        mock_backend.encode_documents.return_value = vec.reshape(1, -1)
        mock_backend.encode_query.return_value = vec.reshape(1, -1)
        mock_backend.dimension = 384

        with (
            override_config(LLM_BACKEND="disabled"),
            patch("consolidation_memory.backends.get_embedding_backend", return_value=mock_backend),
        ):
            from consolidation_memory.cli import cmd_test
            cmd_test()

        captured = capsys.readouterr()
        assert "checks passed" in captured.out
        assert "disabled" in captured.out
        # LLM skipped, but no failures
        assert "\u2717" not in captured.out

    def test_embedding_failure_reports_and_cleans_up(self, capsys):
        """When embedding backend fails, recall is skipped and test episode is cleaned up."""
        ensure_schema()

        mock_backend = MagicMock()
        mock_backend.encode_documents.side_effect = ConnectionError("server unreachable")

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "OK"

        with (
            patch("consolidation_memory.backends.get_embedding_backend", return_value=mock_backend),
            patch("consolidation_memory.backends.get_llm_backend", return_value=mock_llm),
            pytest.raises(SystemExit) as exc_info,
        ):
            from consolidation_memory.cli import cmd_test
            cmd_test()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "server unreachable" in captured.out

    def test_cleanup_on_failure(self):
        """Test episode is soft-deleted even when recall fails."""
        ensure_schema()
        vec = _make_normalized_vec()

        mock_backend = MagicMock()
        mock_backend.encode_documents.return_value = vec.reshape(1, -1)
        # Make encode_query fail so recall fails
        mock_backend.encode_query.side_effect = ConnectionError("query failed")
        mock_backend.dimension = 384

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "OK"

        with (
            patch("consolidation_memory.backends.get_embedding_backend", return_value=mock_backend),
            patch("consolidation_memory.backends.get_llm_backend", return_value=mock_llm),
            pytest.raises(SystemExit),
        ):
            from consolidation_memory.cli import cmd_test
            cmd_test()

        # Verify no test episodes remain in the database
        from consolidation_memory.database import get_all_episodes
        episodes = get_all_episodes(include_deleted=False)
        test_eps = [e for e in episodes if "consolidation-memory-test" in e["content"]]
        assert test_eps == [], "Test episode was not cleaned up after failure"

    def test_no_color_when_env_set(self, capsys, monkeypatch):
        """NO_COLOR env var suppresses ANSI escape codes."""
        monkeypatch.setenv("NO_COLOR", "1")
        ensure_schema()
        vec = _make_normalized_vec()

        mock_backend = MagicMock()
        mock_backend.encode_documents.return_value = vec.reshape(1, -1)
        mock_backend.encode_query.return_value = vec.reshape(1, -1)
        mock_backend.dimension = 384

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "OK"

        with (
            patch("consolidation_memory.backends.get_embedding_backend", return_value=mock_backend),
            patch("consolidation_memory.backends.get_llm_backend", return_value=mock_llm),
        ):
            from consolidation_memory.cli import cmd_test
            cmd_test()

        captured = capsys.readouterr()
        # No ANSI escape sequences
        assert "\033[" not in captured.out


class TestMainDispatch:
    """Test that 'test' subcommand is wired up in main()."""

    def test_test_subcommand_registered(self):
        """argparse recognizes 'test' as a valid subcommand."""
        from consolidation_memory.cli import main

        # Parse just the 'test' command to verify it's registered
        with (
            patch("sys.argv", ["consolidation-memory", "test"]),
            patch("consolidation_memory.cli.cmd_test") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once()

    def test_detect_drift_subcommand_registered(self):
        """argparse recognizes 'detect-drift' and dispatches correctly."""
        from consolidation_memory.cli import main

        with (
            patch("sys.argv", ["consolidation-memory", "detect-drift", "--base-ref", "origin/main"]),
            patch("consolidation_memory.cli.cmd_detect_drift") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once_with(base_ref="origin/main", repo_path=None)

    def test_setup_memory_subcommand_registered(self):
        """argparse recognizes 'setup-memory' and dispatches correctly."""
        from consolidation_memory.cli import main

        with (
            patch("sys.argv", ["consolidation-memory", "setup-memory", "--path", "AGENTS.md"]),
            patch("consolidation_memory.cli.cmd_setup_memory") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once_with("AGENTS.md")

    def test_setup_claude_subcommand_registered(self):
        """argparse recognizes 'setup-claude' legacy alias and dispatches correctly."""
        from consolidation_memory.cli import main

        with (
            patch("sys.argv", ["consolidation-memory", "setup-claude"]),
            patch("consolidation_memory.cli.cmd_setup_claude") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once()


class TestDetectDriftCommand:
    def test_cmd_detect_drift_prints_json(self, capsys):
        from consolidation_memory.cli import cmd_detect_drift

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

        with patch("consolidation_memory.client.MemoryClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.__enter__.return_value = mock_client
            mock_client.detect_drift.return_value = expected

            cmd_detect_drift(base_ref="origin/main", repo_path="C:/repo")

            mock_client.detect_drift.assert_called_once_with(
                base_ref="origin/main",
                repo_path="C:/repo",
            )

        captured = capsys.readouterr()
        assert json.loads(captured.out) == expected

    def test_cmd_detect_drift_exits_on_runtime_error(self, capsys):
        from consolidation_memory.cli import cmd_detect_drift

        with patch("consolidation_memory.client.MemoryClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.__enter__.return_value = mock_client
            mock_client.detect_drift.side_effect = RuntimeError("git diff failed")

            with pytest.raises(SystemExit) as exc_info:
                cmd_detect_drift()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "git diff failed" in captured.err


class TestExportImportHardening:
    def test_export_skips_knowledge_path_traversal(self, tmp_data_dir):
        from consolidation_memory.cli import cmd_export
        from consolidation_memory.config import get_config
        from consolidation_memory.database import ensure_schema, upsert_knowledge_topic

        cfg = get_config()
        ensure_schema()
        outside = cfg.KNOWLEDGE_DIR.parent / "outside_secret.txt"
        outside.write_text("top-secret", encoding="utf-8")
        upsert_knowledge_topic("../outside_secret.txt", "Secret", "Secret summary", source_episodes=[])

        cmd_export()

        exports = list(cfg.BACKUP_DIR.glob("memory_export_*.json"))
        assert len(exports) == 1
        data = json.loads(exports[0].read_text(encoding="utf-8"))
        exported = next(t for t in data["knowledge_topics"] if t["filename"] == "../outside_secret.txt")
        assert exported["file_content"] == ""

    def test_import_missing_source_episodes_uses_default(self, tmp_data_dir):
        from consolidation_memory.cli import cmd_import
        from consolidation_memory.config import get_config
        from consolidation_memory.database import get_all_knowledge_topics

        cfg = get_config()
        payload = {
            "exported_at": "2026-03-05T00:00:00+00:00",
            "version": "1.1",
            "episodes": [],
            "knowledge_topics": [
                {
                    "filename": "topic.md",
                    "title": "Topic",
                    "summary": "Summary",
                    "file_content": "---\ntitle: Topic\nsummary: Summary\n---\n",
                }
            ],
            "stats": {"episode_count": 0, "knowledge_count": 1, "record_count": 0},
        }
        import_path = cfg.BACKUP_DIR / "import_missing_source_episodes.json"
        import_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd_import(str(import_path))

        topics = get_all_knowledge_topics()
        assert len(topics) == 1
        assert topics[0]["filename"] == "topic.md"
        assert json.loads(topics[0]["source_episodes"]) == []

    def test_import_round_trip_claim_graph_entities(self, tmp_data_dir):
        from consolidation_memory.cli import cmd_import
        from consolidation_memory.config import get_config
        from consolidation_memory.database import (
            ensure_schema,
            get_all_claim_edges,
            get_all_claim_events,
            get_all_claim_sources,
            get_all_claims,
            get_all_episode_anchors,
        )

        cfg = get_config()
        ensure_schema()
        payload = {
            "exported_at": "2026-03-06T00:00:00+00:00",
            "version": "1.2",
            "episodes": [
                {
                    "id": "ep-import-1",
                    "content": "Changed src/app.py to APP_MODE=legacy",
                    "content_type": "fact",
                    "tags": ["deploy"],
                    "surprise_score": 0.5,
                }
            ],
            "knowledge_topics": [],
            "knowledge_records": [],
            "claims": [
                {
                    "id": "claim-import-a",
                    "claim_type": "fact",
                    "canonical_text": "src/app.py sets APP_MODE=legacy",
                    "payload": {"path": "src/app.py", "app_mode": "legacy"},
                    "status": "active",
                    "confidence": 0.9,
                    "valid_from": "2026-01-01T00:00:00+00:00",
                    "valid_until": None,
                    "created_at": "2026-03-06T00:00:00+00:00",
                    "updated_at": "2026-03-06T00:00:00+00:00",
                },
                {
                    "id": "claim-import-b",
                    "claim_type": "fact",
                    "canonical_text": "src/app.py sets APP_MODE=modern",
                    "payload": {"path": "src/app.py", "app_mode": "modern"},
                    "status": "challenged",
                    "confidence": 0.7,
                    "valid_from": "2026-01-01T00:00:00+00:00",
                    "valid_until": None,
                    "created_at": "2026-03-06T00:00:01+00:00",
                    "updated_at": "2026-03-06T00:00:01+00:00",
                },
            ],
            "claim_edges": [
                {
                    "id": "edge-import-1",
                    "from_claim_id": "claim-import-a",
                    "to_claim_id": "claim-import-b",
                    "edge_type": "contradicts",
                    "confidence": 0.8,
                    "details": {"reason": "new evidence"},
                    "created_at": "2026-03-06T00:00:02+00:00",
                }
            ],
            "claim_sources": [
                {
                    "id": "source-import-1",
                    "claim_id": "claim-import-a",
                    "source_episode_id": "ep-import-1",
                    "source_topic_id": None,
                    "source_record_id": None,
                    "created_at": "2026-03-06T00:00:03+00:00",
                }
            ],
            "claim_events": [
                {
                    "id": "event-import-1",
                    "claim_id": "claim-import-a",
                    "event_type": "create",
                    "details": {"source": "import"},
                    "created_at": "2026-03-06T00:00:04+00:00",
                }
            ],
            "episode_anchors": [
                {
                    "id": "anchor-import-1",
                    "episode_id": "ep-import-1",
                    "anchor_type": "path",
                    "anchor_value": "src/app.py",
                    "created_at": "2026-03-06T00:00:05+00:00",
                }
            ],
            "stats": {
                "episode_count": 1,
                "knowledge_count": 0,
                "record_count": 0,
                "claim_count": 2,
                "claim_edge_count": 1,
                "claim_source_count": 1,
                "claim_event_count": 1,
                "episode_anchor_count": 1,
            },
        }
        import_path = cfg.BACKUP_DIR / "import_claim_graph_round_trip.json"
        import_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd_import(str(import_path))

        assert len(get_all_claims()) == 2
        assert len(get_all_claim_edges()) == 1
        assert len(get_all_claim_sources()) == 1
        assert len(get_all_claim_events()) == 1
        assert len(get_all_episode_anchors()) == 1

    def test_import_rolls_back_episode_rows_when_embedding_fails(self, tmp_data_dir):
        from consolidation_memory.cli import cmd_import
        from consolidation_memory.config import get_config
        from consolidation_memory.database import get_all_episodes

        cfg = get_config()
        payload = {
            "exported_at": "2026-03-06T00:00:00+00:00",
            "version": "1.2",
            "episodes": [
                {
                    "id": "ep-fail-1",
                    "content": "Episode that should roll back",
                    "content_type": "fact",
                    "tags": [],
                    "surprise_score": 0.5,
                }
            ],
            "knowledge_topics": [],
            "knowledge_records": [],
            "stats": {"episode_count": 1, "knowledge_count": 0, "record_count": 0},
        }
        import_path = cfg.BACKUP_DIR / "import_embed_failure.json"
        import_path.write_text(json.dumps(payload), encoding="utf-8")

        with patch("consolidation_memory.backends.encode_documents", side_effect=RuntimeError("embed down")):
            cmd_import(str(import_path))

        assert get_all_episodes(include_deleted=False) == []

    def test_import_remaps_topic_ids_for_knowledge_records(self, tmp_data_dir):
        from consolidation_memory.cli import cmd_import
        from consolidation_memory.config import get_config
        from consolidation_memory.database import get_all_active_records, get_all_knowledge_topics

        cfg = get_config()
        payload = {
            "exported_at": "2026-03-06T00:00:00+00:00",
            "version": "1.2",
            "episodes": [],
            "knowledge_topics": [
                {
                    "id": "topic-export-1",
                    "filename": "topic.md",
                    "title": "Topic",
                    "summary": "Summary",
                    "source_episodes": [],
                    "file_content": "# Topic\n",
                }
            ],
            "knowledge_records": [
                {
                    "id": "record-export-1",
                    "topic_id": "topic-export-1",
                    "record_type": "fact",
                    "content": {"subject": "A", "info": "B"},
                    "embedding_text": "A:B",
                    "source_episodes": [],
                    "confidence": 0.8,
                }
            ],
            "stats": {"episode_count": 0, "knowledge_count": 1, "record_count": 1},
        }
        import_path = cfg.BACKUP_DIR / "import_topic_id_remap.json"
        import_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd_import(str(import_path))

        topics = get_all_knowledge_topics()
        records = get_all_active_records()
        assert len(topics) == 1
        assert len(records) == 1
        assert topics[0]["id"] != "topic-export-1"
        assert records[0]["topic_id"] == topics[0]["id"]
