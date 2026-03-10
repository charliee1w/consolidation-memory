"""Cross-surface scope policy contract tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from consolidation_memory.database import ensure_schema
from consolidation_memory.schemas import dispatch_tool_call

try:
    from fastapi.testclient import TestClient
    from consolidation_memory.rest import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


_DENY_SCOPE = {
    "namespace": {"slug": "team-a"},
    "project": {"slug": "repo-a"},
    "policy": {"write_mode": "deny"},
}


class TestScopePolicyCrossSurfaceParity:
    def test_python_and_openai_dispatch_respect_write_deny_policy(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        with MemoryClient(auto_consolidate=False) as client:
            python_result = client.store_with_scope(
                content="blocked write",
                scope=_DENY_SCOPE,
            )
            assert python_result.status == "write_denied"

            openai_store = dispatch_tool_call(
                client,
                "memory_store",
                {"content": "blocked write", "scope": _DENY_SCOPE},
            )
            assert openai_store["status"] == "write_denied"

            openai_batch = dispatch_tool_call(
                client,
                "memory_store_batch",
                {"episodes": [{"content": "blocked write"}], "scope": _DENY_SCOPE},
            )
            assert openai_batch["status"] == "write_denied"

    def test_mcp_tools_respect_write_deny_policy(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.server import memory_store, memory_store_batch

        ensure_schema()
        with MemoryClient(auto_consolidate=False) as client:
            with pytest.MonkeyPatch.context() as mp:
                async def _get_client():
                    return client

                mp.setattr("consolidation_memory.server._get_client_with_timeout", _get_client)
                store_output = json.loads(
                    asyncio.run(memory_store(content="blocked write", scope=_DENY_SCOPE))
                )
                batch_output = json.loads(
                    asyncio.run(
                        memory_store_batch(
                            episodes=[{"content": "blocked write"}],
                            scope=_DENY_SCOPE,
                        )
                    )
                )

        assert store_output["status"] == "write_denied"
        assert batch_output["status"] == "write_denied"

    @pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
    def test_rest_endpoints_respect_write_deny_policy(self, tmp_data_dir):
        ensure_schema()
        app = create_app()
        with TestClient(app) as api_client:
            store_resp = api_client.post(
                "/memory/store",
                json={"content": "blocked write", "scope": _DENY_SCOPE},
            )
            batch_resp = api_client.post(
                "/memory/store/batch",
                json={
                    "episodes": [{"content": "blocked write"}],
                    "scope": _DENY_SCOPE,
                },
            )

        assert store_resp.status_code == 200
        assert store_resp.json()["status"] == "write_denied"
        assert batch_resp.status_code == 200
        assert batch_resp.json()["status"] == "write_denied"
