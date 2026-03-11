"""Helpers for scope-safe knowledge topic file storage."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_SUFFIX_RE = re.compile(r"[^A-Za-z0-9.]+")


def _normalize_segment(value: object, default: str) -> str:
    token = str(value or "").strip()
    if not token:
        return default
    normalized = _SAFE_SEGMENT_RE.sub("-", token).strip("._-")
    return normalized[:48] or default


def _safe_logical_filename(filename: object) -> str:
    raw = str(filename or "").strip()
    name = PurePosixPath(raw).name or "topic.md"
    stem, suffix = Path(name).stem, Path(name).suffix
    safe_stem = _SAFE_SEGMENT_RE.sub("-", stem).strip("._-") or "topic"
    safe_suffix = _SAFE_SUFFIX_RE.sub("", suffix)[:10] or ".md"
    return f"{safe_stem[:64]}{safe_suffix}"


def _legacy_default_scope(scope: Mapping[str, Any] | None) -> bool:
    if not scope:
        return True
    defaults = {
        "namespace_slug": "default",
        "project_slug": "default",
        "app_client_name": "legacy_client",
        "app_client_type": "python_sdk",
    }
    for key, expected in defaults.items():
        if str(scope.get(key) or "").strip() != expected:
            return False
    optional_keys = (
        "app_client_provider",
        "app_client_external_key",
        "agent_name",
        "agent_external_key",
        "session_external_key",
        "session_kind",
    )
    return all(not str(scope.get(key) or "").strip() for key in optional_keys)


def build_topic_storage_path(
    filename: str,
    scope: Mapping[str, Any] | None = None,
) -> str:
    """Return the durable storage path for a knowledge topic."""
    logical_filename = str(filename or "").strip() or "topic.md"
    if _legacy_default_scope(scope):
        return logical_filename

    scope_row = scope or {}
    safe_name = _safe_logical_filename(logical_filename)
    digest_input = "|".join(
        [
            logical_filename,
            str(scope_row.get("namespace_slug") or ""),
            str(scope_row.get("project_slug") or ""),
            str(scope_row.get("app_client_name") or ""),
            str(scope_row.get("app_client_type") or ""),
            str(scope_row.get("app_client_provider") or ""),
            str(scope_row.get("app_client_external_key") or ""),
            str(scope_row.get("agent_name") or ""),
            str(scope_row.get("agent_external_key") or ""),
            str(scope_row.get("session_external_key") or ""),
            str(scope_row.get("session_kind") or ""),
        ]
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    relpath = PurePosixPath(
        "_scoped",
        _normalize_segment(scope_row.get("namespace_slug"), "default"),
        _normalize_segment(scope_row.get("project_slug"), "default"),
        _normalize_segment(scope_row.get("app_client_type"), "python_sdk"),
        _normalize_segment(scope_row.get("app_client_name"), "legacy_client"),
        f"{digest}-{safe_name}",
    )
    return str(relpath)


def topic_storage_path(topic: Mapping[str, Any]) -> str:
    """Return the canonical relative storage path for a topic row."""
    storage_path = str(topic.get("storage_path") or "").strip()
    if storage_path:
        return storage_path
    storage_filename = str(topic.get("storage_filename") or "").strip()
    if storage_filename:
        return storage_filename
    return str(topic.get("filename") or "").strip()


def topic_storage_candidates(topic: Mapping[str, Any]) -> list[str]:
    """Return candidate relative paths for reading an existing topic file."""
    candidates: list[str] = []
    for candidate in (topic_storage_path(topic), str(topic.get("filename") or "").strip()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_topic_path(
    knowledge_dir: Path,
    topic: Mapping[str, Any],
    *,
    prefer_existing: bool = False,
) -> Path:
    """Resolve the best on-disk path for a topic row.

    When ``prefer_existing`` is true, legacy root-level files are still honored
    for backward compatibility with older databases and manually created files.
    """
    root = knowledge_dir.resolve()
    first_valid: Path | None = None
    for candidate in topic_storage_candidates(topic):
        path = (knowledge_dir / candidate).resolve()
        if not path.is_relative_to(root):
            continue
        if prefer_existing and path.exists():
            return path
        if first_valid is None:
            first_valid = path
    if first_valid is None:
        raise ValueError("Topic path resolves outside KNOWLEDGE_DIR")
    return first_valid
