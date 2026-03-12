"""Code-drift detection and claim-challenge helpers."""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import posixpath
import subprocess  # nosec B404
import threading
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Sequence

from consolidation_memory.types import DriftAnchor, DriftClaimImpact, DriftOutput

logger = logging.getLogger(__name__)

_GIT_COMMAND_TIMEOUT_SECONDS = 15.0
_MAX_CHANGED_FILES = 2000
_DriftRunKey = tuple[str, str]
_drift_singleflight_lock = threading.Lock()
_drift_singleflight: dict[_DriftRunKey, concurrent.futures.Future[DriftOutput]] = {}


def _chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(values), size):
        yield list(values[idx: idx + size])


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
    try:
        proc = subprocess.run(  # nosec B603
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(git_args)} timed out after {_GIT_COMMAND_TIMEOUT_SECONDS:.0f}s"
        ) from exc
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

    changed_list = sorted(changed)
    if len(changed_list) > _MAX_CHANGED_FILES:
        logger.warning(
            "drift scan found %d changed files, truncating to %d for bounded runtime",
            len(changed_list),
            _MAX_CHANGED_FILES,
        )
        return changed_list[:_MAX_CHANGED_FILES]
    return changed_list


def _to_path_anchors(paths: Iterable[str]) -> list[DriftAnchor]:
    anchors: list[DriftAnchor] = []
    for path in sorted(set(paths)):
        anchors.append({"anchor_type": "path", "anchor_value": path})
    return anchors


def _drift_run_key(repo_dir: Path, base_ref: str | None) -> _DriftRunKey:
    return (repo_dir.as_posix(), (base_ref or "").strip())


def _acquire_drift_future(
    *,
    repo_dir: Path,
    base_ref: str | None,
) -> tuple[concurrent.futures.Future[DriftOutput], bool]:
    key = _drift_run_key(repo_dir, base_ref)
    with _drift_singleflight_lock:
        existing = _drift_singleflight.get(key)
        if existing is not None:
            return existing, False
        future: concurrent.futures.Future[DriftOutput] = concurrent.futures.Future()
        _drift_singleflight[key] = future
        return future, True


def _release_drift_future(
    *,
    repo_dir: Path,
    base_ref: str | None,
    future: concurrent.futures.Future[DriftOutput],
) -> None:
    key = _drift_run_key(repo_dir, base_ref)
    with _drift_singleflight_lock:
        current = _drift_singleflight.get(key)
        if current is future:
            _drift_singleflight.pop(key, None)


def _copy_drift_output(result: DriftOutput) -> DriftOutput:
    return copy.deepcopy(result)


def map_changed_files_to_claims(
    changed_files: Sequence[str],
    repo_path: str | PathLike[str] | None = None,
) -> tuple[list[DriftAnchor], dict[str, dict[str, Any]], dict[str, set[tuple[str, str]]]]:
    """Map changed file paths to claims linked through path anchors."""
    from consolidation_memory.database import get_claims_by_anchor_values

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
    candidate_to_paths: dict[str, set[str]] = {}

    for path_value in normalized_files:
        candidates = _build_path_anchor_candidates(path_value, repo_dir)
        for candidate in candidates:
            candidate_to_paths.setdefault(candidate, set()).add(path_value)

    candidate_values = sorted(candidate_to_paths.keys())
    if not candidate_values:
        return checked_anchors, claim_rows, matched_anchors

    for chunk in _chunked(candidate_values, 250):
        rows = get_claims_by_anchor_values(
            anchor_type="path",
            anchor_values=chunk,
            include_expired=False,
        )
        for row in rows:
            claim_id = str(row.get("id") or "")
            if not claim_id:
                continue
            if claim_id not in claim_rows:
                claim_rows[claim_id] = row

            anchor_value = str(row.get("anchor_value") or "")
            for source_path in candidate_to_paths.get(anchor_value, set()):
                matched_anchors.setdefault(claim_id, set()).add(("path", source_path))

    return checked_anchors, claim_rows, matched_anchors


def _detect_code_drift_once(
    *,
    base_ref: str | None,
    repo_dir: Path,
) -> DriftOutput:
    from consolidation_memory.database import insert_claim_events, mark_claims_challenged_by_ids

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

    challenged_claim_ids = mark_claims_challenged_by_ids(impacted_claim_ids)
    challenged_ids_set = set(challenged_claim_ids)

    changed_anchor_values = [anchor["anchor_value"] for anchor in checked_anchors]
    drift_events: list[dict[str, Any]] = []
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
        drift_events.append(
            {
                "claim_id": claim_id,
                "event_type": "code_drift_detected",
                "details": {
                    "base_ref": base_ref,
                    "changed_files": changed_anchor_values,
                    "matched_anchors": matched_anchors,
                    "new_status": new_status,
                },
            }
        )

    insert_claim_events(drift_events)

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


def detect_code_drift(
    base_ref: str | None = None,
    repo_path: str | PathLike[str] | None = None,
) -> DriftOutput:
    """Detect code drift and challenge impacted claims.

    Calls for the same `(repo_path, base_ref)` share a single in-flight run so
    concurrent agents do not duplicate expensive git and DB work.
    """
    repo_dir = _resolve_repo_dir(repo_path)
    run_future, is_owner = _acquire_drift_future(repo_dir=repo_dir, base_ref=base_ref)

    if not is_owner:
        logger.info(
            "drift detection already in progress for base_ref=%r repo_path=%s; waiting for shared result",
            base_ref,
            repo_dir,
        )
        return _copy_drift_output(run_future.result())

    try:
        result = _detect_code_drift_once(base_ref=base_ref, repo_dir=repo_dir)
    except Exception as exc:
        run_future.set_exception(exc)
        raise
    else:
        run_future.set_result(result)
        return _copy_drift_output(result)
    finally:
        _release_drift_future(
            repo_dir=repo_dir,
            base_ref=base_ref,
            future=run_future,
        )


__all__ = [
    "get_changed_files",
    "map_changed_files_to_claims",
    "detect_code_drift",
]
