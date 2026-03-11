"""Tests for canonical domain-model scope skeleton types."""

import pytest

from consolidation_memory.types import (
    AgentScope,
    AppClientScope,
    NamespaceScope,
    PolicyScope,
    ProjectRepoScope,
    ScopeEnvelope,
    SessionScope,
    coerce_scope_envelope,
)


class TestScopeEnvelopeCoercion:
    def test_none_scope_returns_none(self):
        assert coerce_scope_envelope(None) is None

    def test_typed_scope_round_trips(self):
        scope = ScopeEnvelope(
            namespace=NamespaceScope(slug="team-a"),
            app_client=AppClientScope(app_type="rest", name="gateway"),
            agent=AgentScope(name="triage"),
            session=SessionScope(external_key="thread-1", session_kind="thread"),
            project=ProjectRepoScope(slug="repo-a"),
            policy=PolicyScope(read_visibility="project", write_mode="allow"),
        )
        assert coerce_scope_envelope(scope) is scope

    def test_mapping_scope_coerces_aliases(self):
        scope = coerce_scope_envelope(
            {
                "namespace": {"slug": "team-b", "sharing_mode": "shared"},
                "app": {"app_type": "mcp", "name": "desktop"},
                "agent": {"name": "assistant"},
                "session": {"external_key": "sess-9", "session_kind": "workflow"},
                "project_repo": {"slug": "repo-b", "default_branch": "main"},
                "policy": {"read_visibility": "namespace", "write_mode": "deny"},
            }
        )
        assert scope is not None
        assert scope.namespace.slug == "team-b"
        assert scope.namespace.sharing_mode == "shared"
        assert scope.app_client.app_type == "mcp"
        assert scope.app_client.name == "desktop"
        assert scope.agent is not None and scope.agent.name == "assistant"
        assert scope.session is not None and scope.session.session_kind == "workflow"
        assert scope.project is not None and scope.project.slug == "repo-b"
        assert scope.policy is not None and scope.policy.read_visibility == "namespace"
        assert scope.policy is not None and scope.policy.write_mode == "deny"

    def test_mapping_scope_invalid_policy_values_raise(self):
        with pytest.raises(ValueError, match=r"scope\.policy\.read_visibility must be one of"):
            coerce_scope_envelope(
                {
                    "namespace": {"slug": "team-c"},
                    "policy": {"read_visibility": "invalid", "write_mode": "allow"},
                }
            )

        with pytest.raises(ValueError, match=r"scope\.policy\.write_mode must be one of"):
            coerce_scope_envelope(
                {
                    "namespace": {"slug": "team-c"},
                    "policy": {"read_visibility": "private", "write_mode": "invalid"},
                }
            )

    def test_invalid_scope_type_raises(self):
        with pytest.raises(TypeError, match="scope must be a ScopeEnvelope, mapping, or None"):
            coerce_scope_envelope(123)  # type: ignore[arg-type]

    def test_invalid_nested_project_type_raises(self):
        with pytest.raises(TypeError, match=r"scope\.project must be an object"):
            coerce_scope_envelope({"project": 5})  # type: ignore[arg-type]

    def test_invalid_nested_policy_type_raises(self):
        with pytest.raises(TypeError, match=r"scope\.policy must be an object"):
            coerce_scope_envelope({"policy": 5})  # type: ignore[arg-type]

    def test_invalid_nested_string_field_type_raises(self):
        with pytest.raises(TypeError, match=r"scope\.project\.slug must be a string"):
            coerce_scope_envelope({"project": {"slug": 5}})  # type: ignore[arg-type]
