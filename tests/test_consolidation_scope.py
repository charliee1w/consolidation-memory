"""Regression tests for consolidation scope normalization and cluster scope derivation."""

from consolidation_memory.consolidation.engine import _cluster_scope_row, _episode_scope_row


def test_episode_scope_row_normalizes_windows_path_values():
    scope = _episode_scope_row(
        {
            "project_root_uri": r"C:\trackerscope\\",
            "project_repo_remote": "https://example.com/repo.git/",
        }
    )
    assert scope["project_root_uri"] == "c:/trackerscope"
    assert scope["project_repo_remote"] == "https://example.com/repo.git"


def test_cluster_scope_prefers_shared_non_null_project_root():
    scope = _cluster_scope_row(
        [
            {"id": "ep-1", "project_root_uri": None},
            {"id": "ep-2", "project_root_uri": r"C:\trackerscope"},
        ]
    )
    assert scope["project_root_uri"] == "c:/trackerscope"


def test_cluster_scope_drops_conflicting_project_roots():
    scope = _cluster_scope_row(
        [
            {"id": "ep-1", "project_root_uri": r"C:\trackerscope"},
            {"id": "ep-2", "project_root_uri": r"D:\other-repo"},
        ]
    )
    assert scope["project_root_uri"] is None
