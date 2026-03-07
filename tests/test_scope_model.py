"""Tests for canonical domain-model scope skeleton types."""

import pytest

from consolidation_memory.types import (
    AgentScope,
    AppClientScope,
    NamespaceScope,
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

    def test_invalid_scope_type_raises(self):
        with pytest.raises(TypeError, match="scope must be a ScopeEnvelope, mapping, or None"):
            coerce_scope_envelope(123)  # type: ignore[arg-type]
