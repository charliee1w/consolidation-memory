"""Tests for corpus hygiene tools across dispatch surfaces."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from consolidation_memory.corpus_hygiene import apply_corpus_hygiene, scan_corpus_hygiene
from consolidation_memory.schemas import dispatch_tool_call, openai_tools
from tests.surface_contract_helpers import invoke_surfaces_with_execute_tool_call

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TestHygieneToolDispatch:
    def test_hygiene_tools_present_in_openai_schemas(self):
        names = {tool["function"]["name"] for tool in openai_tools}
        assert "memory_hygiene_scan" in names
        assert "memory_hygiene_apply" in names

    def test_dispatch_hygiene_scan_without_client(self, tmp_data_dir):
        result = dispatch_tool_call(MagicMock(), "memory_hygiene_scan", {})
        assert result["status"] == "ok"
        assert "episodes" in result
        assert "orphaned_claims" in result

    def test_dispatch_hygiene_apply_dry_run_without_client(self, tmp_data_dir):
        result = dispatch_tool_call(
            MagicMock(),
            "memory_hygiene_apply",
            {"use_recommended": True, "dry_run": True},
        )
        assert result["status"] == "dry_run"
        assert "episode_targets" in result

    def test_tool_requires_client_false_for_hygiene(self):
        from consolidation_memory.tool_dispatch import tool_requires_client

        assert tool_requires_client("memory_hygiene_scan") is False
        assert tool_requires_client("memory_hygiene_apply") is False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestHygieneSurfaceContract:
    def test_hygiene_scan_matches_across_surfaces(self):
        from consolidation_memory.server import memory_hygiene_scan

        expected = scan_corpus_hygiene()
        dispatch_out, mcp_out, rest_out, mock_execute = invoke_surfaces_with_execute_tool_call(
            tool_name="memory_hygiene_scan",
            tool_args={},
            expected_result=expected,
            mcp_coro_factory=memory_hygiene_scan,
            rest_path="/memory/hygiene/scan",
            rest_method="GET",
        )

        assert dispatch_out["status"] == "ok"
        assert mcp_out["status"] == "ok"
        assert rest_out["status"] == "ok"
        assert mock_execute.call_count >= 1

    def test_hygiene_apply_rest_endpoint(self):
        from consolidation_memory.rest import create_app

        expected = {
            "status": "dry_run",
            "episode_targets": 0,
            "episode_ids": [],
            "expire_orphans": False,
            "orphan_repair": None,
        }
        with patch(
            "consolidation_memory.rest.execute_tool_call",
            return_value=expected,
        ):
            client = TestClient(create_app())
            resp = client.post(
                "/memory/hygiene/apply",
                json={"use_recommended": True, "dry_run": True},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "dry_run"


class TestHygieneCliIntegration:
    def test_apply_rejects_empty_targets_message(self, tmp_data_dir, capsys):
        from consolidation_memory.cli import cmd_hygiene_apply

        with pytest.raises(SystemExit) as exc:
            cmd_hygiene_apply(
                episode_ids=None,
                use_recommended=False,
                expire_orphans=False,
                dry_run=True,
            )
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "recommended" in captured.err.lower() or "episode-id" in captured.err.lower()

    def test_scan_json_output(self, tmp_data_dir, capsys):
        from consolidation_memory.cli import cmd_hygiene_scan

        cmd_hygiene_scan(as_json=True)
        captured = capsys.readouterr()
        assert '"status": "ok"' in captured.out or '"status": "ok"' in captured.out.replace(" ", "")

    def test_apply_dry_run_round_trip(self, tmp_data_dir):
        result = apply_corpus_hygiene(use_recommended=True, dry_run=True)
        assert result["status"] == "dry_run"