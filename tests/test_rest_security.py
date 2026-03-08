"""Security-focused tests for REST host binding and bearer auth."""

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


def test_validate_rest_bind_rejects_public_host_without_auth(monkeypatch):
    from consolidation_memory.rest import validate_rest_bind

    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with pytest.raises(RuntimeError, match="Refusing to bind REST API"):
        validate_rest_bind("0.0.0.0")


def test_validate_rest_bind_allows_public_host_with_auth_token(monkeypatch):
    from consolidation_memory.rest import validate_rest_bind

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    validate_rest_bind("0.0.0.0")


def test_validate_rest_bind_allows_public_host_with_explicit_override(monkeypatch):
    from consolidation_memory.rest import validate_rest_bind

    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", "true")

    validate_rest_bind("0.0.0.0")


def test_rest_auth_rejects_missing_bearer_header(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with TestClient(create_app()) as client:
        response = client.get("/memory/status")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_rest_auth_rejects_wrong_bearer_token(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with TestClient(create_app()) as client:
        response = client.get(
            "/memory/status",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_rest_auth_accepts_valid_bearer_token(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with TestClient(create_app()) as client:
        response = client.get(
            "/memory/status",
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200


def test_health_remains_open_when_auth_enabled(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")

    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
