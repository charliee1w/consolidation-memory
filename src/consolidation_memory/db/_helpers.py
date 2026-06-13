"""Shared database helpers and small utilities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Sequence

from consolidation_memory.utils import parse_datetime

OUTCOME_TYPES: tuple[str, ...] = (
    "success",
    "failure",
    "partial_success",
    "reverted",
    "superseded",
)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_utc_timestamp(value: str | datetime) -> str:
    """Normalize a timestamp to a UTC ISO 8601 string."""
    dt = parse_datetime(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _normalize_outcome_type(value: object) -> str:
    token = str(value or "").strip().lower()
    if token not in OUTCOME_TYPES:
        raise ValueError(
            "outcome_type must be one of: " + ", ".join(OUTCOME_TYPES)
        )
    return token


def _derive_action_key(action_summary: str) -> str:
    normalized = " ".join(action_summary.split()).strip().casefold()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"act_{digest}"


def _normalize_id_tokens(values: Sequence[str] | None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped
