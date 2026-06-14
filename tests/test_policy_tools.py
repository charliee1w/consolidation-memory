"""Tests for policy administration tools across dispatch surfaces."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from consolidation_memory.policy_admin import grant_policy_binding, list_policy_bindings
from consolidation_memory.schemas import dispatch_tool_call, openai_tools
from tests.surface_contract_helpers import invoke_surfaces_with_execute_tool_call

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TestPolicyAdminCore:
    def test_list_policy_bindings_empty(self, tmp_data_dir):
        result = list_policy_bindings()
        assert result["status"] == "ok"
        assert result["count"] == 0
        assert result["policies"] == []

    def test_grant_and_list_policy_binding(self, tmp_data_dir):
        granted = grant_policy_binding(
            namespace="team-a",
            project="repo-a",
            principal_type="app_client",
            principal_key="python_sdk:legacy_client",
            write_mode="deny",
            read_visibility="namespace",
        )
        assert granted["status"] == "granted"
        assert granted["write_mode"] == "deny"

        listed = list_policy_bindings()
        assert listed["count"] == 1
        row = listed["policies"][0]
        assert row["namespace_slug"] == "team-a"
        assert row["project_slug"] == "repo-a"
        assert row["principal"] == {
            "type": "app_client",
            "key": "python_sdk:legacy_client",
        }

    def test_grant_requires_mode_or_visibility(self, tmp_data_dir):
        with pytest.raises(ValueError, match="write_mode or read_visibility"):
            grant_policy_binding(
                principal_type="app_client",
                principal_key="python_sdk:legacy_client",
            )


class TestPolicyToolDispatch:
    def test_policy_tools_present_in_openai_schemas(self):
        names = {tool["function"]["name"] for tool in openai_tools}
        assert "memory_policy_list" in names
        assert "memory_policy_grant" in names

    def test_dispatch_policy_list_without_client(self, tmp_data_dir):
        result = dispatch_tool_call(MagicMock(), "memory_policy_list", {})
        assert result["status"] == "ok"
        assert result["count"] == 0

    def test_dispatch_policy_grant_without_client(self, tmp_data_dir):
        result = dispatch_tool_call(
            MagicMock(),
            "memory_policy_grant",
            {
                "principal_type": "app_client",
                "principal_key": "mcp:cursor",
                "write_mode": "allow",
                "read_visibility": "project",
            },
        )
        assert result["status"] == "granted"
        assert result["write_mode"] == "allow"


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestPolicySurfaceContract:
    def test_policy_list_matches_across_surfaces(self):
        from consolidation_memory.server import memory_policy_list

        expected = {"status": "ok", "count": 0, "policies": []}
        dispatch_out, mcp_out, rest_out, mock_execute = invoke_surfaces_with_execute_tool_call(
            tool_name="memory_policy_list",
            tool_args={},
            expected_result=expected,
            mcp_coro_factory=memory_policy_list,
            rest_path="/memory/policy",
            rest_method="GET",
        )

        assert mock_execute.call_count >= 1
        assert dispatch_out == mcp_out == rest_out == expected

    def test_policy_grant_matches_across_surfaces(self):
        from consolidation_memory.server import memory_policy_grant

        expected = {
            "status": "granted",
            "policy_id": "policy-1",
            "principal_id": "principal-1",
            "acl_entry_id": "acl-1",
            "namespace": "team-a",
            "project": None,
            "principal_type": "app_client",
            "principal_key": "mcp:cursor",
            "write_mode": "deny",
            "read_visibility": None,
        }
        tool_args = {
            "namespace": "team-a",
            "principal_type": "app_client",
            "principal_key": "mcp:cursor",
            "write_mode": "deny",
        }

        dispatch_out, mcp_out, rest_out, mock_execute = invoke_surfaces_with_execute_tool_call(
            tool_name="memory_policy_grant",
            tool_args=tool_args,
            expected_result=expected,
            mcp_coro_factory=lambda: memory_policy_grant(
                namespace="team-a",
                principal_type="app_client",
                principal_key="mcp:cursor",
                write_mode="deny",
            ),
            rest_path="/memory/policy/grant",
            rest_json=tool_args,
        )

        transport_args = mock_execute.call_args.args[1]
        assert transport_args["principal_type"] == "app_client"
        assert transport_args["write_mode"] == "deny"
        assert dispatch_out == mcp_out == rest_out == expected

    def test_policy_grant_integration_via_rest(self, tmp_data_dir):
        from consolidation_memory.rest import create_app

        app = create_app()
        with TestClient(app) as client:
            grant_resp = client.post(
                "/memory/policy/grant",
                json={
                    "principal_type": "app_client",
                    "principal_key": "rest:client",
                    "write_mode": "allow",
                    "read_visibility": "private",
                },
            )
            assert grant_resp.status_code == 200
            assert grant_resp.json()["status"] == "granted"

            list_resp = client.get("/memory/policy")
            assert list_resp.status_code == 200
            payload = list_resp.json()
            assert payload["count"] == 1
            assert payload["policies"][0]["principal"]["key"] == "rest:client"

    def test_policy_list_mcp_does_not_require_client_init(self):
        from consolidation_memory.server import memory_policy_list

        with patch(
            "consolidation_memory.server._get_client_with_timeout",
            side_effect=RuntimeError("client init should not run"),
        ):
            output = json.loads(asyncio.run(memory_policy_list()))
        assert output["status"] == "ok"