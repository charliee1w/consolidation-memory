"""Tests for persisted policy/ACL evaluation and compatibility behavior."""

from __future__ import annotations

from consolidation_memory.policy_engine import resolve_effective_policy
from consolidation_memory.types import PolicyScope


class TestPolicyEngine:
    def test_default_policy_is_used_when_no_acl_rows_exist(self):
        base = PolicyScope(read_visibility="namespace", write_mode="allow")
        resolved = resolve_effective_policy(base, [])
        assert resolved.source == "scope_policy"
        assert resolved.policy == base
        assert resolved.matched_entries == 0

    def test_write_conflict_uses_deny_overrides_allow(self):
        base = PolicyScope(read_visibility="project", write_mode="allow")
        resolved = resolve_effective_policy(
            base,
            [
                {"write_mode": "allow"},
                {"write_mode": "deny"},
            ],
        )
        assert resolved.source == "persisted_acl"
        assert resolved.policy.write_mode == "deny"
        assert "write_mode_conflict_deny_overrides_allow" in resolved.conflicts

    def test_read_conflict_uses_most_restrictive_visibility(self):
        base = PolicyScope(read_visibility="namespace", write_mode="allow")
        resolved = resolve_effective_policy(
            base,
            [
                {"read_visibility": "namespace"},
                {"read_visibility": "project"},
                {"read_visibility": "private"},
            ],
        )
        assert resolved.source == "persisted_acl"
        assert resolved.policy.read_visibility == "private"
        assert "read_visibility_conflict_most_restrictive_wins" in resolved.conflicts


class TestPersistedPolicyIntegration:
    def test_persisted_acl_is_authoritative_when_present(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import (
            ensure_schema,
            upsert_access_policy,
            upsert_policy_acl_entry,
            upsert_policy_principal,
        )

        ensure_schema()
        principal_id = upsert_policy_principal("app_client", "python_sdk:legacy_client")
        policy_id = upsert_access_policy(namespace_slug="default", project_slug="default")
        upsert_policy_acl_entry(
            policy_id=policy_id,
            principal_id=principal_id,
            write_mode="deny",
            read_visibility="private",
        )

        with MemoryClient(auto_consolidate=False) as client:
            resolved = client.resolve_scope(
                {
                    "namespace": {"slug": "default"},
                    "project": {"slug": "default"},
                    "policy": {"write_mode": "allow", "read_visibility": "namespace"},
                }
            )

        assert resolved.policy_source == "persisted_acl"
        assert resolved.policy_acl_matches >= 1
        assert resolved.policy.write_mode == "deny"
        assert resolved.policy.read_visibility == "private"

    def test_legacy_scope_policy_still_works_without_acl_rows(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        app_a_scope = {
            "namespace_slug": "default",
            "project_slug": "default",
            "app_client_name": "app-a",
            "app_client_type": "rest",
        }
        app_b_scope = {
            "namespace_slug": "default",
            "project_slug": "default",
            "app_client_name": "app-b",
            "app_client_type": "rest",
        }
        ep_a = insert_episode("legacy namespace visibility token", scope=app_a_scope)
        ep_b = insert_episode("legacy namespace visibility token", scope=app_b_scope)

        with MemoryClient(auto_consolidate=False) as client:
            result = client.query_search(
                query="legacy namespace visibility token",
                scope={
                    "namespace": {"slug": "default", "sharing_mode": "private"},
                    "project": {"slug": "default"},
                    "app_client": {"name": "app-a", "app_type": "rest"},
                    "policy": {"read_visibility": "namespace"},
                },
            )

        ids = {episode["id"] for episode in result.episodes}
        assert {ep_a, ep_b}.issubset(ids)

