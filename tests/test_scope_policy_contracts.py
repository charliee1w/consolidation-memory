"""Cross-surface scope/policy contract tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from consolidation_memory.database import (
    ensure_schema,
    insert_episode,
    upsert_access_policy,
    upsert_policy_acl_entry,
    upsert_policy_principal,
)
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

_NAMESPACE_SCOPE = {
    "namespace": {"slug": "default"},
    "project": {"slug": "default"},
    "app_client": {"name": "legacy_client", "app_type": "python_sdk"},
    "policy": {"read_visibility": "namespace"},
}


def _seed_persisted_policy(
    *,
    write_mode: str | None = None,
    read_visibility: str | None = None,
) -> None:
    principal_id = upsert_policy_principal("app_client", "python_sdk:legacy_client")
    policy_id = upsert_access_policy(namespace_slug="default", project_slug="default")
    upsert_policy_acl_entry(
        policy_id=policy_id,
        principal_id=principal_id,
        write_mode=write_mode,
        read_visibility=read_visibility,
    )


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

    def test_persisted_acl_write_deny_is_enforced_across_surfaces(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.server import memory_store, memory_store_batch

        ensure_schema()
        _seed_persisted_policy(write_mode="deny")

        with MemoryClient(auto_consolidate=False) as client:
            python_result = client.store(content="blocked by persisted acl")
            assert python_result.status == "write_denied"

            openai_store = dispatch_tool_call(
                client,
                "memory_store",
                {"content": "blocked by persisted acl"},
            )
            assert openai_store["status"] == "write_denied"

            openai_batch = dispatch_tool_call(
                client,
                "memory_store_batch",
                {"episodes": [{"content": "blocked by persisted acl"}]},
            )
            assert openai_batch["status"] == "write_denied"

            with pytest.MonkeyPatch.context() as mp:
                async def _get_client():
                    return client

                mp.setattr("consolidation_memory.server._get_client_with_timeout", _get_client)
                mcp_store = json.loads(
                    asyncio.run(memory_store(content="blocked by persisted acl"))
                )
                mcp_batch = json.loads(
                    asyncio.run(
                        memory_store_batch(
                            episodes=[{"content": "blocked by persisted acl"}],
                        )
                    )
                )

        assert mcp_store["status"] == "write_denied"
        assert mcp_batch["status"] == "write_denied"

        if HAS_FASTAPI:
            app = create_app()
            with TestClient(app) as api_client:
                rest_store = api_client.post(
                    "/memory/store",
                    json={"content": "blocked by persisted acl"},
                )
                rest_batch = api_client.post(
                    "/memory/store/batch",
                    json={"episodes": [{"content": "blocked by persisted acl"}]},
                )

            assert rest_store.status_code == 200
            assert rest_store.json()["status"] == "write_denied"
            assert rest_batch.status_code == 200
            assert rest_batch.json()["status"] == "write_denied"

    def test_persisted_acl_read_visibility_is_enforced_across_surfaces(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.server import memory_search

        ensure_schema()
        legacy_episode_id = insert_episode(
            content="cross-surface persisted read token",
            scope={
                "namespace_slug": "default",
                "project_slug": "default",
                "app_client_name": "legacy_client",
                "app_client_type": "python_sdk",
            },
        )
        insert_episode(
            content="cross-surface persisted read token",
            scope={
                "namespace_slug": "default",
                "project_slug": "default",
                "app_client_name": "other-app",
                "app_client_type": "rest",
            },
        )
        _seed_persisted_policy(read_visibility="private")

        with MemoryClient(auto_consolidate=False) as client:
            python_result = client.query_search(
                query="cross-surface persisted read token",
                scope=_NAMESPACE_SCOPE,
            )
            python_ids = {ep.get("id") for ep in python_result.episodes}
            assert python_ids == {legacy_episode_id}

            openai_result = dispatch_tool_call(
                client,
                "memory_search",
                {"query": "cross-surface persisted read token", "scope": _NAMESPACE_SCOPE},
            )
            openai_ids = {ep.get("id") for ep in openai_result["episodes"]}
            assert openai_ids == {legacy_episode_id}

            with pytest.MonkeyPatch.context() as mp:
                async def _get_client():
                    return client

                mp.setattr("consolidation_memory.server._get_client_with_timeout", _get_client)
                mcp_result = json.loads(
                    asyncio.run(
                        memory_search(
                            query="cross-surface persisted read token",
                            scope=_NAMESPACE_SCOPE,
                        )
                    )
                )
            mcp_ids = {ep.get("id") for ep in mcp_result["episodes"]}
            assert mcp_ids == {legacy_episode_id}

        if HAS_FASTAPI:
            app = create_app()
            with TestClient(app) as api_client:
                rest_result = api_client.post(
                    "/memory/search",
                    json={
                        "query": "cross-surface persisted read token",
                        "scope": _NAMESPACE_SCOPE,
                    },
                )
            assert rest_result.status_code == 200
            rest_ids = {
                ep.get("id")
                for ep in rest_result.json()["episodes"]
            }
            assert rest_ids == {legacy_episode_id}

    def test_forget_protect_and_correct_respect_write_deny_policy_across_surfaces(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import get_config
        from consolidation_memory.database import ensure_schema, insert_episode, upsert_knowledge_topic
        from consolidation_memory.server import memory_correct, memory_forget, memory_protect

        ensure_schema()
        cfg = get_config()
        episode_id = insert_episode(content="protected content")
        filename = "policy-topic.md"
        (cfg.KNOWLEDGE_DIR / filename).write_text(
            "---\ntitle: Policy Topic\nsummary: Topic\n---\n\n## Facts\n- **Policy**: topic\n",
            encoding="utf-8",
        )
        upsert_knowledge_topic(
            filename=filename,
            title="Policy Topic",
            summary="Topic",
            source_episodes=[episode_id],
            fact_count=1,
            confidence=0.8,
        )

        with MemoryClient(auto_consolidate=False) as client:
            python_forget = client.forget(episode_id, scope=_DENY_SCOPE)
            python_protect = client.protect(episode_id=episode_id, scope=_DENY_SCOPE)
            python_correct = client.correct(filename, "blocked", scope=_DENY_SCOPE)
            assert python_forget.status == "write_denied"
            assert python_protect.status == "write_denied"
            assert python_correct.status == "write_denied"

            openai_forget = dispatch_tool_call(
                client,
                "memory_forget",
                {"episode_id": episode_id, "scope": _DENY_SCOPE},
            )
            openai_protect = dispatch_tool_call(
                client,
                "memory_protect",
                {"episode_id": episode_id, "scope": _DENY_SCOPE},
            )
            openai_correct = dispatch_tool_call(
                client,
                "memory_correct",
                {"topic_filename": filename, "correction": "blocked", "scope": _DENY_SCOPE},
            )
            assert openai_forget["status"] == "write_denied"
            assert openai_protect["status"] == "write_denied"
            assert openai_correct["status"] == "write_denied"

            with pytest.MonkeyPatch.context() as mp:
                async def _get_client():
                    return client

                mp.setattr("consolidation_memory.server._get_client_with_timeout", _get_client)
                mcp_forget = json.loads(
                    asyncio.run(memory_forget(episode_id=episode_id, scope=_DENY_SCOPE))
                )
                mcp_protect = json.loads(
                    asyncio.run(memory_protect(episode_id=episode_id, scope=_DENY_SCOPE))
                )
                mcp_correct = json.loads(
                    asyncio.run(
                        memory_correct(
                            topic_filename=filename,
                            correction="blocked",
                            scope=_DENY_SCOPE,
                        )
                    )
                )

            assert mcp_forget["status"] == "write_denied"
            assert mcp_protect["status"] == "write_denied"
            assert mcp_correct["status"] == "write_denied"

        if HAS_FASTAPI:
            app = create_app()
            with TestClient(app) as api_client:
                rest_forget = api_client.post(
                    "/memory/forget",
                    json={"episode_id": episode_id, "scope": _DENY_SCOPE},
                )
                rest_protect = api_client.post(
                    "/memory/protect",
                    json={"episode_id": episode_id, "scope": _DENY_SCOPE},
                )
                rest_correct = api_client.post(
                    "/memory/correct",
                    json={
                        "topic_filename": filename,
                        "correction": "blocked",
                        "scope": _DENY_SCOPE,
                    },
                )

            assert rest_forget.status_code == 200
            assert rest_forget.json()["status"] == "write_denied"
            assert rest_protect.status_code == 200
            assert rest_protect.json()["status"] == "write_denied"
            assert rest_correct.status_code == 200
            assert rest_correct.json()["status"] == "write_denied"
