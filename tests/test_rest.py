"""Tests for the REST API.

Run with: python -m pytest tests/test_rest.py -v
Requires: pip install consolidation-memory[rest,dev]
"""

from unittest.mock import patch

import numpy as np
import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


def _make_normalized_vec(dim=384, seed=None):
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def api_client():
    """Create a FastAPI TestClient with MemoryClient."""
    from consolidation_memory.rest import create_app
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestStoreEndpoint:
    @patch("consolidation_memory.backends.encode_documents")
    def test_store_episode(self, mock_embed, api_client):
        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        resp = api_client.post("/memory/store", json={
            "content": "REST store test",
            "content_type": "fact",
            "tags": ["rest"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["id"] is not None
        assert data["content_type"] == "fact"


class TestRecallEndpoint:
    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_recall(self, mock_embed_docs, mock_embed_query, api_client):
        vec = _make_normalized_vec(seed=42)
        mock_embed_docs.return_value = vec.reshape(1, -1)
        mock_embed_query.return_value = vec.reshape(1, -1)

        # Store first
        api_client.post("/memory/store", json={"content": "REST recall test"})

        # Recall
        resp = api_client.post("/memory/recall", json={"query": "REST recall"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_episodes"] == 1
        assert len(data["episodes"]) >= 1


class TestStatusEndpoint:
    def test_status(self, api_client):
        resp = api_client.get("/memory/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "episodic_buffer" in data
        assert "version" in data


class TestForgetEndpoint:
    @patch("consolidation_memory.backends.encode_documents")
    def test_forget_existing(self, mock_embed, api_client):
        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        store_resp = api_client.post("/memory/store", json={"content": "forget me"})
        ep_id = store_resp.json()["id"]

        resp = api_client.delete(f"/memory/episodes/{ep_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "forgotten"

    def test_forget_nonexistent(self, api_client):
        resp = api_client.delete("/memory/episodes/nonexistent-uuid")
        assert resp.status_code == 404


class TestExportEndpoint:
    def test_export(self, api_client):
        resp = api_client.post("/memory/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "exported"
        assert "path" in data
