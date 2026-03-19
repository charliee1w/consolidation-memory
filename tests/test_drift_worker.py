from __future__ import annotations

import json

from unittest.mock import patch


def test_run_detect_drift_worker_sets_project_and_scope():
    from consolidation_memory.drift_worker import run_detect_drift_worker

    expected = {
        "checked_anchors": [],
        "impacted_claim_ids": ["claim-1"],
        "challenged_claim_ids": ["claim-1"],
        "impacts": [],
    }

    with (
        patch("consolidation_memory.config.set_active_project", return_value="universal") as mock_set_project,
        patch("consolidation_memory.database.ensure_schema") as mock_ensure_schema,
        patch("consolidation_memory.drift.detect_code_drift", return_value=expected) as mock_detect,
    ):
        result = run_detect_drift_worker(
            base_ref="origin/main",
            repo_path="C:/repo",
            project="universal",
        )

    assert result == expected
    mock_set_project.assert_called_once_with("universal")
    mock_ensure_schema.assert_called_once_with()
    mock_detect.assert_called_once_with(
        base_ref="origin/main",
        repo_path="C:/repo",
        scope={
            "namespace_slug": "default",
            "project_slug": "universal",
        },
    )


def test_drift_worker_main_prints_json(capsys):
    from consolidation_memory.drift_worker import main

    payload = {
        "checked_anchors": [],
        "impacted_claim_ids": [],
        "challenged_claim_ids": [],
        "impacts": [],
    }

    with patch(
        "consolidation_memory.drift_worker.run_detect_drift_worker",
        return_value=payload,
    ) as mock_run:
        exit_code = main(["--project", "universal", "--base-ref", "origin/main", "--repo-path", "C:/repo"])

    assert exit_code == 0
    mock_run.assert_called_once_with(
        base_ref="origin/main",
        repo_path="C:/repo",
        project="universal",
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_drift_worker_main_reports_runtime_error(capsys):
    from consolidation_memory.drift_worker import main

    with patch(
        "consolidation_memory.drift_worker.run_detect_drift_worker",
        side_effect=RuntimeError("git failed"),
    ):
        exit_code = main(["--project", "universal"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "git failed" in captured.err
