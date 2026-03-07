"""Tests for code drift detection and claim invalidation."""

from __future__ import annotations

import json
import subprocess

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
