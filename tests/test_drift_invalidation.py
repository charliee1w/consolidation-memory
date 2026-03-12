"""Tests for code drift detection and claim invalidation."""

from __future__ import annotations

import json
import subprocess
import threading
import time

import pytest


class TestChangedFiles:
    def test_get_changed_files_combines_sources(self, monkeypatch, tmp_path):
        from consolidation_memory import drift

        calls: list[tuple[str, ...]] = []
        responses = {
            ("diff", "--name-only"): ["src/c.py", "src/a.py"],
            ("diff", "--name-only", "--cached"): ["src/b.py"],
            ("ls-files", "--others", "--exclude-standard"): ["src/d.py"],
            ("diff", "--name-only", "origin/main...HEAD"): ["src/a.py", "src/e.py"],
        }

        def fake_run_git_lines(repo_dir, git_args):
            del repo_dir
            key = tuple(git_args)
            calls.append(key)
            return responses[key]

        monkeypatch.setattr(drift, "_run_git_lines", fake_run_git_lines)

        changed = drift.get_changed_files(base_ref="origin/main", repo_path=tmp_path)

        assert changed == ["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"]
        assert ("diff", "--name-only", "origin/main...HEAD") in calls

    def test_run_git_lines_timeout_raises_runtime_error(self, monkeypatch, tmp_path):
        from consolidation_memory import drift

        def fake_run(*args, **kwargs):
            del args, kwargs
            raise subprocess.TimeoutExpired(cmd="git diff --name-only", timeout=1.0)

        monkeypatch.setattr(drift.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="timed out"):
            drift._run_git_lines(tmp_path, ["diff", "--name-only"])

    def test_get_changed_files_truncates_large_sets(self, monkeypatch, tmp_path):
        from consolidation_memory import drift

        monkeypatch.setattr(drift, "_MAX_CHANGED_FILES", 3)

        def fake_run_git_lines(repo_dir, git_args):
            del repo_dir
            key = tuple(git_args)
            if key == ("diff", "--name-only"):
                return ["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"]
            return []

        monkeypatch.setattr(drift, "_run_git_lines", fake_run_git_lines)

        changed = drift.get_changed_files(repo_path=tmp_path)

        assert changed == ["src/a.py", "src/b.py", "src/c.py"]


class TestDriftDetection:
    def test_detect_code_drift_challenges_claims_and_logs_events(self, monkeypatch, tmp_data_dir):
        from consolidation_memory.database import (
            ensure_schema,
            get_connection,
            insert_claim_sources,
            insert_episode,
            insert_episode_anchors,
            upsert_claim,
        )
        from consolidation_memory.drift import detect_code_drift

        ensure_schema()
        episode_id = insert_episode("drift anchor episode")
        insert_episode_anchors(
            episode_id,
            [{"anchor_type": "path", "anchor_value": "src/app.py"}],
        )

        upsert_claim(
            claim_id="claim-active",
            claim_type="fact",
            canonical_text="active claim",
            payload={"subject": "app", "info": "active"},
            status="active",
            valid_from="2025-01-01T00:00:00+00:00",
        )
        insert_claim_sources("claim-active", [{"source_episode_id": episode_id}])

        upsert_claim(
            claim_id="claim-challenged",
            claim_type="fact",
            canonical_text="already challenged claim",
            payload={"subject": "app", "info": "challenged"},
            status="challenged",
            valid_from="2025-01-01T00:00:00+00:00",
        )
        insert_claim_sources("claim-challenged", [{"source_episode_id": episode_id}])

        monkeypatch.setattr(
            "consolidation_memory.drift.get_changed_files",
            lambda base_ref=None, repo_path=None: ["src/app.py"],
        )

        result = detect_code_drift(base_ref="origin/main", repo_path=tmp_data_dir)

        assert result["checked_anchors"] == [{"anchor_type": "path", "anchor_value": "src/app.py"}]
        assert result["impacted_claim_ids"] == ["claim-active", "claim-challenged"]
        assert result["challenged_claim_ids"] == ["claim-active"]

        impacts = {impact["claim_id"]: impact for impact in result["impacts"]}
        assert impacts["claim-active"]["previous_status"] == "active"
        assert impacts["claim-active"]["new_status"] == "challenged"
        assert impacts["claim-active"]["matched_anchors"] == [
            {"anchor_type": "path", "anchor_value": "src/app.py"}
        ]
        assert impacts["claim-challenged"]["previous_status"] == "challenged"
        assert impacts["claim-challenged"]["new_status"] == "challenged"

        with get_connection() as conn:
            claim_rows = conn.execute(
                "SELECT id, status FROM claims WHERE id IN (?, ?) ORDER BY id",
                ("claim-active", "claim-challenged"),
            ).fetchall()
            event_rows = conn.execute(
                """SELECT claim_id, event_type, details
                   FROM claim_events
                   WHERE event_type = 'code_drift_detected'
                   ORDER BY claim_id""",
            ).fetchall()

        assert [row["status"] for row in claim_rows] == ["challenged", "challenged"]
        assert [row["claim_id"] for row in event_rows] == ["claim-active", "claim-challenged"]
        assert all(row["event_type"] == "code_drift_detected" for row in event_rows)

        first_details = json.loads(event_rows[0]["details"])
        assert first_details["base_ref"] == "origin/main"
        assert first_details["changed_files"] == ["src/app.py"]

    def test_detect_code_drift_challenges_by_impacted_ids(self, monkeypatch, tmp_data_dir):
        from consolidation_memory import drift

        checked_anchors = [
            {"anchor_type": "path", "anchor_value": "src/app.py"},
            {"anchor_type": "path", "anchor_value": "src/lib.py"},
        ]
        claim_rows = {
            "claim-a": {"id": "claim-a", "status": "active"},
            "claim-b": {"id": "claim-b", "status": "challenged"},
        }
        matched = {
            "claim-a": {("path", "src/app.py")},
            "claim-b": {("path", "src/lib.py")},
        }

        monkeypatch.setattr(
            drift,
            "get_changed_files",
            lambda base_ref=None, repo_path=None: ["src/app.py", "src/lib.py"],
        )
        monkeypatch.setattr(
            drift,
            "map_changed_files_to_claims",
            lambda changed_files, repo_path=None, scope=None: (
                checked_anchors,
                claim_rows,
                matched,
            ),
        )
        monkeypatch.setattr(
            drift,
            "_build_path_anchor_candidates",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected candidate expansion")),
        )

        called: dict[str, list[str]] = {}

        def fake_mark_claims_challenged_by_ids(claim_ids, challenged_at=None):
            del challenged_at
            called["claim_ids"] = list(claim_ids)
            return ["claim-a"]

        monkeypatch.setattr(
            "consolidation_memory.database.mark_claims_challenged_by_ids",
            fake_mark_claims_challenged_by_ids,
        )
        captured_events: dict[str, list[dict[str, object]]] = {}
        monkeypatch.setattr(
            "consolidation_memory.database.insert_claim_events",
            lambda events: captured_events.setdefault(
                "events", [dict(event) for event in events]
            ),
        )

        result = drift.detect_code_drift(base_ref="origin/main", repo_path=tmp_data_dir)

        assert called["claim_ids"] == ["claim-a", "claim-b"]
        assert result["challenged_claim_ids"] == ["claim-a"]
        assert [event["claim_id"] for event in captured_events["events"]] == [
            "claim-a",
            "claim-b",
        ]

    def test_detect_code_drift_singleflights_concurrent_calls(self, monkeypatch, tmp_data_dir):
        from consolidation_memory import drift

        # Keep only one expensive drift run active for identical scope/base_ref.
        call_count = 0
        call_count_lock = threading.Lock()
        first_call_entered = threading.Event()
        unblock_first_call = threading.Event()

        def fake_get_changed_files(base_ref=None, repo_path=None):
            del base_ref, repo_path
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                call_index = call_count
            if call_index == 1:
                first_call_entered.set()
                assert unblock_first_call.wait(timeout=2.0)
            return ["src/app.py"]

        monkeypatch.setattr(drift, "get_changed_files", fake_get_changed_files)
        monkeypatch.setattr(
            drift,
            "map_changed_files_to_claims",
            lambda changed_files, repo_path=None, scope=None: (
                [{"anchor_type": "path", "anchor_value": "src/app.py"}],
                {},
                {},
            ),
        )

        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def _run_detect() -> None:
            try:
                results.append(
                    drift.detect_code_drift(base_ref="origin/main", repo_path=tmp_data_dir)
                )
            except BaseException as exc:  # pragma: no cover - defensive capture
                errors.append(exc)

        first = threading.Thread(target=_run_detect)
        second = threading.Thread(target=_run_detect)
        first.start()
        assert first_call_entered.wait(timeout=2.0)
        second.start()
        time.sleep(0.05)
        unblock_first_call.set()

        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert not errors
        assert not first.is_alive()
        assert not second.is_alive()
        assert call_count == 1
        assert len(results) == 2
        assert results[0] == results[1]

    def test_detect_code_drift_does_not_share_singleflight_across_scopes(
        self,
        monkeypatch,
        tmp_data_dir,
    ):
        from consolidation_memory import drift

        call_count = 0
        mapped_scopes: list[dict[str, str] | None] = []
        call_lock = threading.Lock()
        first_call_entered = threading.Event()
        release_calls = threading.Event()

        def fake_get_changed_files(base_ref=None, repo_path=None):
            del base_ref, repo_path
            nonlocal call_count
            with call_lock:
                call_count += 1
                call_index = call_count
            if call_index == 1:
                first_call_entered.set()
                assert release_calls.wait(timeout=2.0)
            return ["src/app.py"]

        def fake_map_changed_files_to_claims(changed_files, repo_path=None, scope=None):
            del changed_files, repo_path
            with call_lock:
                mapped_scopes.append(dict(scope) if scope is not None else None)
            return ([{"anchor_type": "path", "anchor_value": "src/app.py"}], {}, {})

        monkeypatch.setattr(drift, "get_changed_files", fake_get_changed_files)
        monkeypatch.setattr(drift, "map_changed_files_to_claims", fake_map_changed_files_to_claims)

        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def _run_detect(scope: dict[str, str]) -> None:
            try:
                results.append(
                    drift.detect_code_drift(
                        base_ref="origin/main",
                        repo_path=tmp_data_dir,
                        scope=scope,
                    )
                )
            except BaseException as exc:  # pragma: no cover - defensive capture
                errors.append(exc)

        first = threading.Thread(
            target=_run_detect,
            args=({"namespace_slug": "default", "project_slug": "repo-a"},),
        )
        second = threading.Thread(
            target=_run_detect,
            args=({"namespace_slug": "default", "project_slug": "repo-b"},),
        )
        first.start()
        assert first_call_entered.wait(timeout=2.0)
        second.start()
        time.sleep(0.05)
        release_calls.set()

        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert not errors
        assert not first.is_alive()
        assert not second.is_alive()
        assert call_count == 2
        assert len(results) == 2
        assert sorted(mapped_scopes, key=lambda scope: scope["project_slug"]) == [
            {"namespace_slug": "default", "project_slug": "repo-a"},
            {"namespace_slug": "default", "project_slug": "repo-b"},
        ]
