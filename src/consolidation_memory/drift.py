"""Code-drift detection and claim-challenge helpers."""

from __future__ import annotations

import logging
import posixpath
import subprocess
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Sequence

from consolidation_memory.types import DriftAnchor, DriftClaimImpact, DriftOutput

logger = logging.getLogger(__name__)


def _normalize_changed_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return ""
    normalized = posixpath.normpath(normalized)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized == ".":
        return ""
    return normalized


def _resolve_repo_dir(repo_path: str | PathLike[str] | None) -> Path:
    repo_dir = Path(repo_path).expanduser().resolve() if repo_path else Path.cwd().resolve()
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise RuntimeError(f"Repository path does not exist or is not a directory: {repo_dir}")
    return repo_dir


def _run_git_lines(repo_dir: Path, git_args: Sequence[str]) -> list[str]:
    cmd = ["git", "-c", f"safe.directory={repo_dir.as_posix()}", *git_args]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        if not details:
            details = f"exit code {proc.returncode}"
        raise RuntimeError(f"git {' '.join(git_args)} failed: {details}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _build_path_anchor_candidates(path_value: str, repo_dir: Path) -> list[str]:
    normalized = _normalize_changed_path(path_value)
    if not normalized:
        return []

    windows_rel = normalized.replace("/", "\\")
    abs_path = (repo_dir / normalized).resolve()
    abs_posix = abs_path.as_posix()
    abs_windows = str(abs_path)

    seen: set[str] = set()
    candidates: list[str] = []
    for candidate in (
        normalized,
        f"./{normalized}",
        windows_rel,
        f".\\{windows_rel}",
        abs_posix,
        abs_windows,
    ):
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def get_changed_files(
    base_ref: str | None = None,
    repo_path: str | PathLike[str] | None = None,
) -> list[str]:
    """Return normalized changed paths from git working tree and optional base ref."""
    repo_dir = _resolve_repo_dir(repo_path)

    changed: set[str] = set()
    raw_paths: list[str] = []

    raw_paths.extend(_run_git_lines(repo_dir, ["diff", "--name-only"]))
    raw_paths.extend(_run_git_lines(repo_dir, ["diff", "--name-only", "--cached"]))
    raw_paths.extend(_run_git_lines(repo_dir, ["ls-files", "--others", "--exclude-standard"]))

    if base_ref:
        raw_paths.extend(_run_git_lines(repo_dir, ["diff", "--name-only", f"{base_ref}...HEAD"]))

    for path in raw_paths:
        normalized = _normalize_changed_path(path)
        if normalized:
            changed.add(normalized)

    return sorted(changed)


def _to_path_anchors(paths: Iterable[str]) -> list[DriftAnchor]:
    anchors: list[DriftAnchor] = []
    for path in sorted(set(paths)):
        anchors.append({"anchor_type": "path", "anchor_value": path})
    return anchors


def map_changed_files_to_claims(
    changed_files: Sequence[str],
    repo_path: str | PathLike[str] | None = None,
) -> tuple[list[DriftAnchor], dict[str, dict[str, Any]], dict[str, set[tuple[str, str]]]]:
    """Map changed file paths to claims linked through path anchors."""
    from consolidation_memory.database import get_claims_by_anchor

    repo_dir = _resolve_repo_dir(repo_path)
    normalized_files = sorted(
        {
            normalized
            for normalized in (_normalize_changed_path(path) for path in changed_files)
            if normalized
        }
    )
    checked_anchors = _to_path_anchors(normalized_files)

    claim_rows: dict[str, dict[str, Any]] = {}
    matched_anchors: dict[str, set[tuple[str, str]]] = {}

    for path_value in normalized_files:
        candidates = _build_path_anchor_candidates(path_value, repo_dir)
        for candidate in candidates:
            rows = get_claims_by_anchor(
                anchor_type="path",
                anchor_value=candidate,
                include_expired=False,
                limit=1000,
            )
            for row in rows:
                claim_id = str(row.get("id") or "")
                if not claim_id:
                    continue
                if claim_id not in claim_rows:
                    claim_rows[claim_id] = row
                matched_anchors.setdefault(claim_id, set()).add(("path", path_value))

    return checked_anchors, claim_rows, matched_anchors


def detect_code_drift(
    base_ref: str | None = None,
    repo_path: str | PathLike[str] | None = None,
) -> DriftOutput:
    """Detect code drift and challenge impacted claims."""
    from consolidation_memory.database import insert_claim_event, mark_claims_challenged_by_anchors

    repo_dir = _resolve_repo_dir(repo_path)
    changed_files = get_changed_files(base_ref=base_ref, repo_path=repo_dir)

    checked_anchors, claim_rows, matched_anchor_pairs = map_changed_files_to_claims(
        changed_files=changed_files,
        repo_path=repo_dir,
    )

    impacted_claim_ids = sorted(claim_rows.keys())
    if not impacted_claim_ids:
        return {
            "checked_anchors": checked_anchors,
            "impacted_claim_ids": [],
            "challenged_claim_ids": [],
            "impacts": [],
        }

    lookup_anchors: list[dict[str, str]] = []
    for anchor in checked_anchors:
        path_value = anchor["anchor_value"]
        for candidate in _build_path_anchor_candidates(path_value, repo_dir):
            lookup_anchors.append({"anchor_type": "path", "anchor_value": candidate})

    challenged_claim_ids = sorted(mark_claims_challenged_by_anchors(lookup_anchors))
    challenged_ids_set = set(challenged_claim_ids)

    impacts: list[DriftClaimImpact] = []
    for claim_id in impacted_claim_ids:
        previous_status = str(claim_rows[claim_id].get("status") or "")
        new_status = "challenged" if claim_id in challenged_ids_set else previous_status
        matched_anchors: list[DriftAnchor] = [
            {"anchor_type": anchor_type, "anchor_value": anchor_value}
            for anchor_type, anchor_value in sorted(
                matched_anchor_pairs.get(claim_id, set()),
                key=lambda item: (item[0], item[1]),
            )
        ]
        impacts.append(
            {
                "claim_id": claim_id,
                "previous_status": previous_status,
                "new_status": new_status,
                "matched_anchors": matched_anchors,
            }
        )

        insert_claim_event(
            claim_id=claim_id,
            event_type="code_drift_detected",
            details={
                "base_ref": base_ref,
                "changed_files": [anchor["anchor_value"] for anchor in checked_anchors],
                "matched_anchors": matched_anchors,
                "new_status": new_status,
            },
        )

    logger.info(
        "code drift detected base_ref=%r changed_paths=%d impacted_claims=%d challenged_claims=%d",
        base_ref,
        len(checked_anchors),
        len(impacted_claim_ids),
        len(challenged_claim_ids),
    )

    return {
        "checked_anchors": checked_anchors,
        "impacted_claim_ids": impacted_claim_ids,
        "challenged_claim_ids": challenged_claim_ids,
        "impacts": impacts,
    }


__all__ = [
    "get_changed_files",
    "map_changed_files_to_claims",
    "detect_code_drift",
]
