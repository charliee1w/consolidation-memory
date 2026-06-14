"""Shared helpers for cross-surface contract tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from consolidation_memory.schemas import dispatch_tool_call

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None  # type: ignore[misc, assignment]


def assert_recall_deadline_injected(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    deadline = normalized.pop("_recall_deadline_monotonic", None)
    assert isinstance(deadline, float)
    assert deadline > 0
    return normalized


def invoke_surfaces_with_execute_tool_call(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    expected_result: dict[str, Any],
    mcp_coro_factory: Callable[[], Any],
    rest_path: str,
    rest_json: dict[str, Any] | None = None,
    rest_method: str = "POST",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], MagicMock]:
    """Run OpenAI dispatch, MCP wrapper, and REST with one execute_tool_call mock."""
    from consolidation_memory.rest import create_app

    if TestClient is None:
        raise RuntimeError("fastapi not installed")

    mock_client = MagicMock()
    mock_execute = MagicMock(return_value=expected_result)
    recorded_tool_calls: list[tuple[str, dict[str, Any]]] = []
    rest_body = rest_json if rest_json is not None else tool_args

    def _chain_alias_tools(
        name: str,
        arguments: dict[str, Any],
        *,
        client: Any = None,
    ) -> dict[str, Any]:
        recorded_tool_calls.append((name, dict(arguments)))
        if name == "memory_remember":
            from consolidation_memory.simple_api import build_remember_store_arguments

            return _chain_alias_tools(
                "memory_store",
                build_remember_store_arguments(arguments),
                client=client,
            )
        if name == "memory_ask":
            from consolidation_memory.simple_api import (
                build_ask_recall_arguments,
                simplify_recall_result,
            )

            recall_args = build_ask_recall_arguments(arguments)
            raw = _chain_alias_tools("memory_recall", recall_args, client=client)
            simplified = simplify_recall_result(raw)
            simplified["query"] = recall_args["query"]
            return simplified
        return expected_result

    mock_execute.side_effect = _chain_alias_tools
    mock_execute.recorded_tool_calls = recorded_tool_calls

    with (
        patch("consolidation_memory.tool_dispatch.execute_tool_call", mock_execute),
        patch("consolidation_memory.server.execute_tool_call", mock_execute),
        patch("consolidation_memory.rest.execute_tool_call", mock_execute),
    ):
        dispatch_out = dispatch_tool_call(mock_client, tool_name, tool_args)

        with (
            patch(
                "consolidation_memory.server._get_client_with_timeout",
                return_value=mock_client,
            ),
            patch(
                "consolidation_memory.server._await_warmup_ready",
                return_value=True,
            ),
        ):
            mcp_out = json.loads(asyncio.run(mcp_coro_factory()))

        app = create_app()
        with patch(
            "consolidation_memory.rest.MemoryRuntime.get_client_with_timeout",
            return_value=mock_client,
        ):
            with TestClient(app) as client:
                if rest_method.upper() == "GET":
                    rest_resp = client.get(rest_path)
                else:
                    rest_resp = client.post(rest_path, json=rest_body)
        rest_out = rest_resp.json()

    assert rest_resp.status_code == 200
    return dispatch_out, mcp_out, rest_out, mock_execute