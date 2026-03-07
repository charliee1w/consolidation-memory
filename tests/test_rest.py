"""Tests for the REST API.

Run with: python -m pytest tests/test_rest.py -v
Requires: pip install consolidation-memory[rest,dev]
"""

import time
from unittest.mock import patch

import pytest

from tests.helpers import make_normalized_vec as _make_normalized_vec

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


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

    def test_store_with_scope_uses_scoped_client_method(self, api_client):
        from consolidation_memory.types import StoreResult

        with patch(
            "consolidation_memory.client.MemoryClient.store_with_scope",
            return_value=StoreResult(status="stored", id="scoped-id", content_type="fact"),
        ) as mock_store_with_scope:
            resp = api_client.post(
                "/memory/store",
                json={
                    "content": "scoped store",
                    "content_type": "fact",
                    "scope": {"namespace": {"slug": "team-a"}},
                },
            )

        assert resp.status_code == 200
        assert resp.json()["id"] == "scoped-id"
        assert mock_store_with_scope.call_count == 1


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

    def test_recall_with_scope_uses_scoped_client_method(self, api_client):
        from consolidation_memory.types import RecallResult

        with patch(
            "consolidation_memory.client.MemoryClient.query_recall",
            return_value=RecallResult(episodes=[], knowledge=[], total_episodes=0, total_knowledge_topics=0),
        ) as mock_query_recall:
            resp = api_client.post(
                "/memory/recall",
                json={
                    "query": "scope",
                    "scope": {"namespace": {"slug": "team-a", "sharing_mode": "shared"}},
                },
            )

        assert resp.status_code == 200
        assert resp.json()["total_episodes"] == 0
        assert mock_query_recall.call_count == 1


class TestStatusEndpoint:
    def test_status(self, api_client):
        resp = api_client.get("/memory/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "episodic_buffer" in data
        assert "version" in data


class TestSearchEndpoint:
    def test_search_with_scope_uses_scoped_client_method(self, api_client):
        from consolidation_memory.types import SearchResult

        with patch(
            "consolidation_memory.client.MemoryClient.query_search",
            return_value=SearchResult(episodes=[], total_matches=0, query="scope"),
        ) as mock_query_search:
            resp = api_client.post(
                "/memory/search",
                json={
                    "query": "scope",
                    "scope": {"project": {"slug": "repo-a"}},
                },
            )

        assert resp.status_code == 200
        assert resp.json()["total_matches"] == 0
        assert mock_query_search.call_count == 1


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


class TestBatchStoreEndpoint:
    @patch("consolidation_memory.backends.encode_documents")
    def test_batch_store(self, mock_embed, api_client):
        from tests.helpers import make_normalized_batch
        mock_embed.return_value = make_normalized_batch(2, seed=42)

        resp = api_client.post("/memory/store/batch", json={
            "episodes": [
                {"content": "Episode 1", "content_type": "fact"},
                {"content": "Episode 2", "tags": ["test"]},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["stored"] == 2

    def test_batch_store_with_scope_uses_scoped_client_method(self, api_client):
        from consolidation_memory.types import BatchStoreResult

        with patch(
            "consolidation_memory.client.MemoryClient.store_batch_with_scope",
            return_value=BatchStoreResult(status="stored", stored=1, duplicates=0),
        ) as mock_store_batch_with_scope:
            resp = api_client.post(
                "/memory/store/batch",
                json={
                    "episodes": [{"content": "Episode 1"}],
                    "scope": {"project": {"slug": "repo-a"}},
                },
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "stored"
        assert mock_store_batch_with_scope.call_count == 1

    def test_batch_store_malformed_episode(self, api_client):
        """Missing required 'content' field should return 422, not 500."""
        resp = api_client.post("/memory/store/batch", json={
            "episodes": [
                {"content_type": "fact", "tags": ["no-content"]},
            ]
        })
        assert resp.status_code == 422


class TestExportEndpoint:
    def test_export(self, api_client):
        resp = api_client.post("/memory/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "exported"
        assert "path" in data


class TestClaimEndpoints:
    def test_claim_browse(self, api_client):
        from consolidation_memory.database import upsert_claim

        upsert_claim(
            claim_id="claim-rest-browse-1",
            claim_type="fact",
            canonical_text="python runtime is 3.12",
            payload={"subject": "python", "info": "3.12"},
            valid_from="2025-01-01T00:00:00+00:00",
        )

        resp = api_client.post("/memory/claims/browse", json={"claim_type": "fact"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any(c["id"] == "claim-rest-browse-1" for c in data["claims"])

    def test_claim_search(self, api_client):
        from consolidation_memory.database import upsert_claim

        upsert_claim(
            claim_id="claim-rest-search-1",
            claim_type="procedure",
            canonical_text="start API with uvicorn main:app",
            payload={"trigger": "run server", "steps": "uvicorn main:app --reload"},
            valid_from="2025-01-01T00:00:00+00:00",
        )

        resp = api_client.post("/memory/claims/search", json={"query": "uvicorn"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_matches"] >= 1
        assert any(c["id"] == "claim-rest-search-1" for c in data["claims"])

    def test_claim_browse_as_of(self, api_client):
        from consolidation_memory.database import upsert_claim

        upsert_claim(
            claim_id="claim-rest-old",
            claim_type="fact",
            canonical_text="uses python 3.11",
            payload={"subject": "python", "info": "3.11"},
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-rest-new",
            claim_type="fact",
            canonical_text="uses python 3.12",
            payload={"subject": "python", "info": "3.12"},
            valid_from="2025-07-01T00:00:00+00:00",
        )

        resp = api_client.post(
            "/memory/claims/browse",
            json={"as_of": "2025-03-01T00:00:00+00:00", "claim_type": "fact"},
        )
        assert resp.status_code == 200
        data = resp.json()
        ids = {claim["id"] for claim in data["claims"]}
        assert "claim-rest-old" in ids
        assert "claim-rest-new" not in ids

    def test_claim_browse_with_scope_uses_canonical_client_method(self, api_client):
        from consolidation_memory.types import ClaimBrowseResult

        with patch(
            "consolidation_memory.client.MemoryClient.query_browse_claims",
            return_value=ClaimBrowseResult(claims=[], total=0),
        ) as mock_query_browse:
            resp = api_client.post(
                "/memory/claims/browse",
                json={
                    "claim_type": "fact",
                    "scope": {"project": {"slug": "repo-a"}},
                },
            )

        assert resp.status_code == 200
        mock_query_browse.assert_called_once_with(
            claim_type="fact",
            as_of=None,
            limit=50,
            scope={"project": {"slug": "repo-a"}},
        )

    def test_claim_search_with_scope_uses_canonical_client_method(self, api_client):
        from consolidation_memory.types import ClaimSearchResult

        with patch(
            "consolidation_memory.client.MemoryClient.query_search_claims",
            return_value=ClaimSearchResult(claims=[], total_matches=0, query="python"),
        ) as mock_query_search:
            resp = api_client.post(
                "/memory/claims/search",
                json={
                    "query": "python",
                    "scope": {"namespace": {"slug": "team-a"}},
                },
            )

        assert resp.status_code == 200
        mock_query_search.assert_called_once_with(
            query="python",
            claim_type=None,
            as_of=None,
            limit=50,
            scope={"namespace": {"slug": "team-a"}},
        )


class TestDriftEndpoint:
    def test_detect_drift(self, api_client):
        expected = {
            "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            "impacted_claim_ids": ["claim-1"],
            "challenged_claim_ids": ["claim-1"],
            "impacts": [{
                "claim_id": "claim-1",
                "previous_status": "active",
                "new_status": "challenged",
                "matched_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            }],
        }

        with patch("consolidation_memory.client.MemoryClient.query_detect_drift", return_value=expected) as mock_detect:
            resp = api_client.post("/memory/detect-drift", json={"base_ref": "origin/main"})

        assert resp.status_code == 200
        assert resp.json() == expected
        mock_detect.assert_called_once()
        assert mock_detect.call_args.kwargs == {"base_ref": "origin/main", "repo_path": None}

    def test_detect_drift_runtime_error_returns_400(self, api_client):
        with patch(
            "consolidation_memory.client.MemoryClient.query_detect_drift",
            side_effect=RuntimeError("git diff failed"),
        ):
            resp = api_client.post("/memory/detect-drift", json={})

        assert resp.status_code == 400
        assert "git diff failed" in resp.json()["detail"]

    def test_detect_drift_timeout_returns_408(self, api_client):
        def _slow_detect(*args, **kwargs):
            del args, kwargs
            time.sleep(0.05)
            return {}

        with (
            patch("consolidation_memory.client.MemoryClient.query_detect_drift", side_effect=_slow_detect),
            patch("consolidation_memory.rest._MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS", 0.01),
        ):
            resp = api_client.post("/memory/detect-drift", json={})

        assert resp.status_code == 408
        assert "timed out after" in resp.json()["detail"]
