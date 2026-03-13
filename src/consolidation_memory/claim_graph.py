"""Deterministic claim canonicalization helpers.

This module is pure transformation logic and does not perform any I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict


_WS_RE = re.compile(r"\s+")

_RECORD_TYPE_ALIASES: dict[str, str] = {
    "fact": "fact",
    "solution": "solution",
    "preference": "preference",
    "procedure": "procedure",
    "strategy": "strategy",
}

_RECORD_FIELD_SPECS: dict[str, list[tuple[str, bool]]] = {
    # (field_name, lowercase_for_identity)
    "fact": [("subject", True), ("info", False)],
    "solution": [("problem", True), ("fix", False), ("context", False)],
    "preference": [("key", True), ("value", False), ("context", False)],
    "procedure": [("trigger", True), ("steps", False), ("context", False)],
    "strategy": [
        ("problem_pattern", True),
        ("strategy", False),
        ("preconditions", False),
        ("expected_signals", False),
        ("failure_modes", False),
        ("context", False),
    ],
}


class ClaimObject(TypedDict):
    """Normalized claim representation derived from a knowledge record."""

    id: str
    claim_type: str
    canonical_text: str
    payload: dict[str, str]


def _normalize_text(value: Any, *, lowercase: bool = False) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    text = text.rstrip(".;")
    if lowercase:
        text = text.casefold()
    return text


def _normalize_value(value: Any, *, lowercase: bool = False) -> str:
    if isinstance(value, Mapping):
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _normalize_text(serialized, lowercase=lowercase)
    if isinstance(value, (list, tuple, set)):
        pieces: list[str] = []
        for item in value:
            normalized_item = _normalize_text(item, lowercase=lowercase)
            if normalized_item:
                pieces.append(normalized_item)
        return " | ".join(pieces)
    return _normalize_text(value, lowercase=lowercase)


def _parse_content(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return {}


def normalize_record_type(record_type: str) -> str:
    """Normalize and validate a record type string."""
    normalized = _normalize_text(record_type, lowercase=True)
    if normalized not in _RECORD_TYPE_ALIASES:
        raise ValueError(f"Unsupported record_type: {record_type!r}")
    return _RECORD_TYPE_ALIASES[normalized]


def normalize_claim_payload(record_type: str, payload: Mapping[str, Any]) -> dict[str, str]:
    """Normalize a record payload into a deterministic claim payload."""
    claim_type = normalize_record_type(record_type)
    specs = _RECORD_FIELD_SPECS[claim_type]

    normalized: dict[str, str] = {"type": claim_type}
    for field_name, lowercase in specs:
        value = _normalize_value(payload.get(field_name), lowercase=lowercase)
        if value:
            normalized[field_name] = value
    return normalized


def canonical_claim_id(record_type: str, payload: Mapping[str, Any], prefix: str = "clm") -> str:
    """Generate a stable claim ID from normalized type + payload."""
    normalized_payload = normalize_claim_payload(record_type, payload)
    canonical_blob = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical_blob.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def claim_text(record_type: str, normalized_payload: Mapping[str, Any]) -> str:
    """Create deterministic claim text from normalized payload fields."""
    claim_type = normalize_record_type(record_type)
    fields = [name for name, _ in _RECORD_FIELD_SPECS[claim_type] if normalized_payload.get(name)]
    parts = [f"type={claim_type}", *[f"{field}={normalized_payload[field]}" for field in fields]]
    return " | ".join(parts)


def claim_from_record(record: Mapping[str, Any]) -> ClaimObject:
    """Map a knowledge record into a deterministic claim object.

    Supported record inputs:
    - extraction-like: {"type": "...", ...fields...}
    - DB-like: {"record_type": "...", "content": "{...json...}"}
    - DB-like with dict content: {"record_type": "...", "content": {...}}
    """
    content_obj = _parse_content(record.get("content"))
    if not content_obj:
        content_obj = dict(record)

    record_type = (
        record.get("record_type")
        or content_obj.get("type")
        or record.get("type")
        or "fact"
    )
    claim_type = normalize_record_type(str(record_type))
    normalized_payload = normalize_claim_payload(claim_type, content_obj)
    claim_id = canonical_claim_id(claim_type, normalized_payload)
    canonical_text = claim_text(claim_type, normalized_payload)

    return {
        "id": claim_id,
        "claim_type": claim_type,
        "canonical_text": canonical_text,
        "payload": normalized_payload,
    }


def claims_from_records(records: Iterable[Mapping[str, Any]]) -> list[ClaimObject]:
    """Convert records to unique deterministic claims in stable input order."""
    claims: list[ClaimObject] = []
    seen_ids: set[str] = set()
    for record in records:
        claim = claim_from_record(record)
        if claim["id"] in seen_ids:
            continue
        seen_ids.add(claim["id"])
        claims.append(claim)
    return claims


__all__ = [
    "ClaimObject",
    "canonical_claim_id",
    "claim_from_record",
    "claim_text",
    "claims_from_records",
    "normalize_claim_payload",
    "normalize_record_type",
]
