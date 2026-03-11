"""Positive-path checks for advertised optional extras."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest


def test_rest_extra_exposes_create_app_when_fastapi_is_installed():
    pytest.importorskip("fastapi")

    from consolidation_memory.rest import create_app

    app = create_app(bind_host="127.0.0.1")
    assert app.title == "Consolidation Memory"


def test_dashboard_extra_exposes_textual_dashboard_when_installed():
    pytest.importorskip("textual")

    import consolidation_memory.dashboard as dashboard

    assert dashboard._TEXTUAL_AVAILABLE is True
    app = dashboard.DashboardApp()
    assert app.TITLE == "consolidation-memory dashboard"


def test_openai_extra_allows_backend_construction_with_installed_sdk(monkeypatch):
    openai = pytest.importorskip("openai")

    from consolidation_memory.backends.openai_backend import (
        OpenAIEmbeddingBackend,
        OpenAILLMBackend,
    )

    fake_openai_cls = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", fake_openai_cls)

    OpenAIEmbeddingBackend(model_name="text-embedding-3-small", dimension=3, api_key="test-key")
    OpenAILLMBackend(model="gpt-4o-mini", api_key="test-key")

    assert fake_openai_cls.call_args_list == [
        call(api_key="test-key"),
        call(api_key="test-key"),
    ]
