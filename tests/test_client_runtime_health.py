"""Health probe helpers for embedding backends."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from consolidation_memory.client_runtime import (
    _embedding_probe_request,
    _ollama_model_available,
    check_embedding_backend,
    probe_backend,
)


class _ProbeClient:
    _probe_cache = None
    _probe_cache_ttl = 30.0


def test_embedding_probe_request_uses_ollama_tags_endpoint():
    cfg = SimpleNamespace(
        EMBEDDING_BACKEND="ollama",
        EMBEDDING_API_BASE="http://localhost:11434/v1",
    )
    url, body = _embedding_probe_request(cfg)
    assert url == "http://localhost:11434/api/tags"
    assert body is None


def test_embedding_probe_request_uses_openai_models_endpoint():
    cfg = SimpleNamespace(
        EMBEDDING_BACKEND="openai",
        EMBEDDING_API_BASE="https://api.openai.com/v1",
    )
    url, body = _embedding_probe_request(cfg)
    assert url == "https://api.openai.com/v1/models"
    assert body is None


def test_ollama_model_available_matches_tagged_model_names():
    names = ["nomic-embed-text:latest", "dolphin3:8b"]
    assert _ollama_model_available(names, "nomic-embed-text")
    assert not _ollama_model_available(names, "missing-model")


def test_probe_backend_uses_ollama_tags_url(monkeypatch):
    captured: dict[str, str] = {}

    class _FakeResponse:
        def read(self):
            return b'{"models":[]}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=3):
        captured["url"] = req.full_url
        return _FakeResponse()

    cfg = SimpleNamespace(
        EMBEDDING_BACKEND="ollama",
        EMBEDDING_API_BASE="http://localhost:11434",
        EMBEDDING_MODEL_NAME="nomic-embed-text",
    )
    monkeypatch.setattr("consolidation_memory.config.get_config", lambda: cfg)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert probe_backend(_ProbeClient()) is True
    assert captured["url"] == "http://localhost:11434/api/tags"


def test_check_embedding_backend_logs_success_for_ollama_model(monkeypatch):
    class _FakeResponse:
        def read(self):
            return (
                b'{"models":[{"name":"nomic-embed-text:latest"},'
                b'{"name":"dolphin3:8b"}]}'
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    cfg = SimpleNamespace(
        EMBEDDING_BACKEND="ollama",
        EMBEDDING_API_BASE="http://localhost:11434",
        EMBEDDING_MODEL_NAME="nomic-embed-text",
    )
    monkeypatch.setattr("consolidation_memory.config.get_config", lambda: cfg)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=5: _FakeResponse())

    with patch("consolidation_memory.client_runtime.logger") as mock_logger:
        check_embedding_backend(_ProbeClient())
        mock_logger.info.assert_called_once()