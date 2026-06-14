"""Tests for the browser UI and simplified REST helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from consolidation_memory.simple_api import map_simple_kind, simplify_recall_result
from consolidation_memory.web_ui import load_index_html

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


class TestSimplificationHelpers:
    def test_map_simple_kind(self):
        assert map_simple_kind("note") == "exchange"
        assert map_simple_kind("fix") == "solution"
        assert map_simple_kind("FACT") == "fact"

    def test_map_simple_kind_rejects_unknown(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            map_simple_kind("strategy")

    def test_simplify_recall_result(self):
        payload = {
            "episodes": [
                {
                    "id": "ep-1",
                    "content": "x" * 300,
                    "content_type": "solution",
                    "tags": ["a"],
                    "score": 0.9,
                    "created_at": "2026-06-14T00:00:00+00:00",
                }
            ],
            "claims": [
                {
                    "id": "clm-1",
                    "claim_type": "solution",
                    "canonical_text": "fix auth",
                    "status": "active",
                    "reliability": {"band": "high"},
                    "relevance": 0.8,
                }
            ],
            "total_episodes": 1,
        }
        simplified = simplify_recall_result(payload)
        assert len(simplified["episodes"]) == 1
        assert simplified["episodes"][0]["preview"].endswith("...")
        assert simplified["claims"][0]["trust"] == "high"


class TestWebUiRoutes:
    @pytest.fixture
    def ui_client(self):
        from consolidation_memory.rest import create_app

        app = create_app()
        with TestClient(app) as client:
            yield client

    def test_root_redirects_to_ui(self, ui_client):
        resp = ui_client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/ui/"

    def test_ui_page_served(self, ui_client):
        resp = ui_client.get("/ui/")
        assert resp.status_code == 200
        assert "consolidation-memory" in resp.text
        assert "Search memory" in resp.text

    def test_load_index_html_matches_package_asset(self):
        html = load_index_html()
        assert "<!DOCTYPE html>" in html
        assert 'id="ask-query"' in html

    def test_overview_endpoint(self, ui_client):
        resp = ui_client.get("/ui/api/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "health" in data
        assert "version" in data

    @patch("consolidation_memory.backends.encode_documents")
    def test_remember_and_ask_flow(self, mock_embed, ui_client):
        from tests.helpers import make_normalized_vec as _vec

        mock_embed.return_value = _vec(seed=7).reshape(1, -1)

        store = ui_client.post(
            "/ui/api/remember",
            json={"content": "UI remember test", "kind": "fact", "tags": ["ui-test"]},
        )
        assert store.json()["content_type"] == "fact"
        assert store.status_code == 200
        assert store.json()["status"] == "stored"

        ask = ui_client.post(
            "/ui/api/ask",
            json={"query": "UI remember test", "n_results": 5},
        )
        assert ask.status_code == 200
        body = ask.json()
        assert body["query"] == "UI remember test"
        assert isinstance(body["episodes"], list)

    def test_remember_rejects_invalid_kind(self, ui_client):
        resp = ui_client.post(
            "/ui/api/remember",
            json={"content": "bad kind", "kind": "note"},
        )
        assert resp.status_code == 200

        resp = ui_client.post(
            "/ui/api/remember",
            json={"content": "bad", "kind": "invalid"},
        )
        assert resp.status_code == 422