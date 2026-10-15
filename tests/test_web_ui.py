"""Tests for the browser UI and simplified REST helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from consolidation_memory.simple_api import map_simple_kind, simplify_recall_result

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


requires_fastapi = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


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


@requires_fastapi
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
        from consolidation_memory.web_ui import load_index_html

        html = load_index_html()
        assert "<!DOCTYPE html>" in html
        assert 'id="ask-query"' in html
        assert 'data-tab="health"' in html
        assert 'data-tab="hygiene"' in html
        assert 'data-tab="metrics"' in html
        assert 'id="wizard-backdrop"' in html

    def test_overview_endpoint(self, ui_client):
        resp = ui_client.get("/ui/api/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "health" in data
        assert "version" in data
        assert "warnings" in data
        assert "fix_actions" in data
        assert isinstance(data["warnings"], list)

    def test_setup_status_endpoint(self, ui_client):
        resp = ui_client.get("/ui/api/setup/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "needs_setup" in data
        assert "project" in data
        assert "version" in data

    def test_metrics_endpoint(self, ui_client):
        resp = ui_client.get("/ui/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "benchmark" in data
        assert "sections" in data
        assert isinstance(data["sections"], list)

    @patch("consolidation_memory.web_ui.run_quick_setup")
    def test_setup_quick_endpoint(self, mock_setup, ui_client):
        mock_setup.return_value = {
            "status": "configured",
            "config_path": "/tmp/config.toml",
            "project": "test",
            "mcp": {"full": {}, "simple": {}},
            "message": "done",
        }
        resp = ui_client.post("/ui/api/setup/quick")
        assert resp.status_code == 200
        assert resp.json()["status"] == "configured"

    @patch("consolidation_memory.web_ui.run_quick_setup")
    def test_setup_quick_missing_dependency(self, mock_setup, ui_client):
        mock_setup.return_value = {
            "status": "missing_dependency",
            "message": "Install fastembed first",
        }
        resp = ui_client.post("/ui/api/setup/quick")
        assert resp.status_code == 400

    @patch("consolidation_memory.web_ui.warmup_recall_caches")
    def test_warmup_endpoint(self, mock_warmup, ui_client):
        mock_warmup.return_value = {"status": "ok", "elapsed_seconds": 0.1}
        resp = ui_client.post("/ui/api/warmup")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @patch("consolidation_memory.web_ui.reindex_all_episodes")
    def test_reindex_endpoint(self, mock_reindex, ui_client):
        mock_reindex.return_value = {"status": "ok", "episodes_reindexed": 0}
        resp = ui_client.post("/ui/api/reindex")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @patch("consolidation_memory.web_ui.scan_corpus_hygiene")
    def test_hygiene_scan_endpoint(self, mock_scan, ui_client):
        mock_scan.return_value = {
            "status": "ok",
            "episodes": {
                "total_active": 10,
                "temp": {"count": 1, "ids": ["a"], "samples": []},
                "exchange": {"count": 2, "ids": ["b"], "samples": []},
                "noise_journal": {"count": 0, "ids": [], "samples": []},
                "recommended_cleanup_ids": ["a", "b"],
                "would_remain": 8,
            },
            "orphaned_claims": {"count": 0, "ids": [], "samples": []},
            "stale_episode_sources": {"count": 0, "episode_ids": []},
        }
        resp = ui_client.get("/ui/api/hygiene/scan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episodes"]["total_active"] == 10
        assert len(data["episodes"]["recommended_cleanup_ids"]) == 2

    @patch("consolidation_memory.web_ui.apply_corpus_hygiene")
    def test_hygiene_apply_endpoint(self, mock_apply, ui_client):
        mock_apply.return_value = {
            "status": "ok",
            "forgotten": 2,
            "not_found": 0,
            "episode_targets": 2,
            "expire_orphans": True,
            "orphan_repair": {"expired_claims": 1},
        }
        resp = ui_client.post(
            "/ui/api/hygiene/apply",
            json={"use_recommended": True, "expire_orphans": True, "dry_run": False},
        )
        assert resp.status_code == 200
        assert resp.json()["forgotten"] == 2
        mock_apply.assert_called_once()

    def test_hygiene_apply_requires_targets(self, ui_client):
        resp = ui_client.post(
            "/ui/api/hygiene/apply",
            json={"dry_run": False},
        )
        assert resp.status_code == 400

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

    @patch("consolidation_memory.backends.encode_documents")
    def test_ask_still_recalls_knowledge_end_to_end(self, mock_embed, ui_client):
        """Dropping the pre-translation must not drop the knowledge half.

        ``include_knowledge`` is set by ``build_ask_recall_arguments`` from
        inside ``memory_ask``'s dispatch, so the UI never had to send it; this
        asserts the value that reaches the client instead of the key that no
        longer crosses the hop.
        """
        from unittest.mock import patch as _patch

        from consolidation_memory.types import RecallResult
        from tests.helpers import make_normalized_vec as _vec

        mock_embed.return_value = _vec(seed=13).reshape(1, -1)

        with _patch(
            "consolidation_memory.client.MemoryClient.query_recall",
            autospec=True,
            return_value=RecallResult(),
        ) as mock_recall:
            response = ui_client.post("/ui/api/ask", json={"query": "ui knowledge probe"})

        assert response.status_code == 200, response.text
        assert mock_recall.call_args.kwargs["include_knowledge"] is True
        assert mock_recall.call_args.kwargs["query"] == "ui knowledge probe"

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


@requires_fastapi
class TestAskPathSendsPublishedArguments:
    """``/ui/api/ask`` must not pre-translate its arguments.

    The UI used to pass ``build_ask_recall_arguments(...)`` output —
    ``query``/``n_results``/**``include_knowledge``** — into ``memory_ask``,
    whose published schema declares only ``query``/``n_results``/``scope``.
    ``include_knowledge`` was a constant the translator always set, so the
    double translation was a no-op that would now be a contract violation;
    these tests pin both the argument names and the value that mattered.
    """

    @pytest.fixture
    def recorded_ui_app(self):
        """Web UI on a recording ``execute``, so the hop is observed directly."""
        from fastapi import FastAPI

        from consolidation_memory.web_ui import register_web_ui_routes

        calls: list[tuple[str, dict[str, object]]] = []

        async def _record(name, arguments, **_kwargs):
            calls.append((name, dict(arguments)))
            return {"query": arguments.get("query"), "episodes": []}

        app = FastAPI()
        register_web_ui_routes(app, execute=_record)
        with TestClient(app) as client:
            yield client, calls

    def test_ask_passes_only_published_ask_arguments(self, recorded_ui_app):
        from consolidation_memory.tool_dispatch import accepted_argument_names

        client, calls = recorded_ui_app

        response = client.post("/ui/api/ask", json={"query": "ui ask", "n_results": 4})

        assert response.status_code == 200, response.text
        assert [name for name, _ in calls] == ["memory_ask"]
        arguments = calls[0][1]
        assert arguments == {"query": "ui ask", "n_results": 4}
        assert set(arguments) <= accepted_argument_names("memory_ask")
        assert "include_knowledge" not in arguments



@requires_fastapi
class TestOpsRoutes:
    @pytest.fixture
    def ops_client(self):
        from consolidation_memory.rest import create_app

        app = create_app()
        with TestClient(app) as client:
            yield client

    def test_ops_overview_matches_ui_shape(self, ops_client):
        resp = ops_client.get("/ops/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "health" in data
        assert "maintenance_daemon" in data
        assert "warnings" in data
        assert "fix_actions" in data

    def test_ops_metrics_endpoint(self, ops_client):
        resp = ops_client.get("/ops/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark"] == "real_world_eval"
        assert isinstance(data["sections"], list)

    @patch("consolidation_memory.daemon_service.daemon_status")
    def test_ops_daemon_status(self, mock_status, ops_client):
        mock_status.return_value = {"installed": True, "running": False, "message": "stopped"}
        resp = ops_client.get("/ops/daemon/status")
        assert resp.status_code == 200
        assert resp.json()["installed"] is True

    @patch("consolidation_memory.daemon_service.install_daemon")
    def test_ops_daemon_install(self, mock_install, ops_client):
        mock_install.return_value = {"status": "ok", "message": "installed"}
        resp = ops_client.post("/ops/daemon/install")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @patch("consolidation_memory.ops_routes.warmup_recall_caches")
    def test_ops_warmup(self, mock_warmup, ops_client):
        mock_warmup.return_value = {"status": "ok"}
        resp = ops_client.post("/ops/warmup")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"