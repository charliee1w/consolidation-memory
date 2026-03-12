"""Security-focused tests for REST host binding and bearer auth."""

import importlib

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


def test_create_app_rejects_explicit_public_bind_without_auth(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with pytest.raises(RuntimeError, match="Refusing to bind REST API"):
        create_app(bind_host="0.0.0.0")


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


def test_rest_auth_accepts_case_insensitive_bearer_scheme(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with TestClient(create_app()) as client:
        response = client.get(
            "/memory/status",
            headers={"Authorization": "bearer test-token"},
        )
    assert response.status_code == 200


def test_rest_auth_rejects_malformed_bearer_header(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with TestClient(create_app()) as client:
        response = client.get(
            "/memory/status",
            headers={"Authorization": "Bearer test-token extra"},
        )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_rest_module_uses_defaults_when_numeric_env_values_are_invalid(monkeypatch):
    monkeypatch.setenv("CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS", "nan")
    monkeypatch.setenv("CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS", "not-a-float")
    monkeypatch.setenv("CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS", "inf")
    monkeypatch.setenv("CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS", "")

    import consolidation_memory.rest as rest

    rest = importlib.reload(rest)

    assert rest._MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS == 180.0
    assert rest._MEMORY_RECALL_TIMEOUT_SECONDS == 45.0
    assert rest._MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS == 10.0
    assert rest._CLIENT_INIT_TIMEOUT_SECONDS == 45.0


def test_health_remains_open_when_auth_enabled(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.setenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", "test-token")

    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_programmatic_app_blocks_public_requests_without_auth(monkeypatch):
    from consolidation_memory.rest import create_app

    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND", raising=False)

    with TestClient(create_app(), base_url="http://0.0.0.0:8123") as client:
        response = client.get("/memory/status")
        health = client.get("/health")

    assert response.status_code == 503
    assert "Refusing unauthenticated REST requests on a non-loopback bind" in response.json()["detail"]
    assert health.status_code == 200
