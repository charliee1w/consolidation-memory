"""Entity-centric recall helpers — thin layer over anchors and record subjects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, cast

from consolidation_memory.anchors import AnchorResult, extract_anchors
from consolidation_memory.database import (
    get_claim_ids_by_subject_token,
    get_claims_by_anchor_values,
    get_episode_ids_by_entity_anchors,
    get_record_ids_by_subject_token,
)

_PATH_LIKE_RE = re.compile(
    r"(?:[/\\]|\.(?:py|md|toml|yml|yaml|json|ps1|ts|tsx|js|rs|go)\b)",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,127}$")


@dataclass(frozen=True)
class EntityResolution:
    """Resolved entity linkage used to boost entity-centric recall."""

    entity: str
    kind: str
    anchors: tuple[AnchorResult, ...] = ()
    episode_ids: frozenset[str] = field(default_factory=frozenset)
    claim_ids: frozenset[str] = field(default_factory=frozenset)
    record_ids: frozenset[str] = field(default_factory=frozenset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "kind": self.kind,
            "anchors": list(self.anchors),
            "linked_episode_count": len(self.episode_ids),
            "linked_claim_count": len(self.claim_ids),
            "linked_record_count": len(self.record_ids),
        }


def _normalize_entity(entity: str) -> str:
    return str(entity or "").strip()


def _path_basename(value: str) -> str:
    return value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def resolve_entity_anchors(entity: str) -> list[AnchorResult]:
    """Build anchor candidates from an entity string."""
    token = _normalize_entity(entity)
    if not token:
        return []

    anchors = list(extract_anchors(token))
    seen = {(a["anchor_type"], a["anchor_value"]) for a in anchors}

    if _PATH_LIKE_RE.search(token):
        if not any(a["anchor_type"] == "path" for a in anchors):
            anchors.append({"anchor_type": "path", "anchor_value": token})
            seen.add(("path", token))
        basename = _path_basename(token)
        if basename and basename != token and ("path", basename) not in seen:
            anchors.append({"anchor_type": "path", "anchor_value": basename})
    elif _IDENTIFIER_RE.match(token):
        # Subject-style entity (module name, config key, etc.)
        pass

    return anchors


def _entity_kind(entity: str, anchors: list[AnchorResult]) -> str:
    anchor_types = {anchor["anchor_type"] for anchor in anchors}
    if "path" in anchor_types:
        return "path"
    if len(anchor_types) == 1:
        return next(iter(anchor_types))
    if anchor_types:
        return "mixed"
    if _IDENTIFIER_RE.match(entity):
        return "subject"
    return "unknown"


def resolve_entity_context(
    entity: str | None,
    *,
    scope: Mapping[str, str | None] | None = None,
) -> EntityResolution | None:
    """Resolve episodes, claims, and records linked to an entity."""
    token = _normalize_entity(entity or "")
    if not token:
        return None

    anchors = resolve_entity_anchors(token)
    kind = _entity_kind(token, anchors)

    episode_ids = set(
        get_episode_ids_by_entity_anchors(
            cast(Sequence[Mapping[str, str]], anchors),
            scope=scope,
            limit=200,
        )
    )

    claim_ids: set[str] = set()
    record_ids: set[str] = set()

    path_values = [
        anchor["anchor_value"]
        for anchor in anchors
        if anchor["anchor_type"] == "path"
    ]
    if path_values:
        for row in get_claims_by_anchor_values(
            "path",
            path_values,
            scope=scope,
            limit=100,
        ):
            claim_id = str(row.get("id") or "").strip()
            if claim_id:
                claim_ids.add(claim_id)

    for anchor in anchors:
        if anchor["anchor_type"] == "tool":
            claim_ids.update(
                get_claim_ids_by_subject_token(anchor["anchor_value"], scope=scope, limit=50)
            )
            record_ids.update(
                get_record_ids_by_subject_token(anchor["anchor_value"], scope=scope, limit=50)
            )

    if kind == "subject" or (not anchors and _IDENTIFIER_RE.match(token)):
        claim_ids.update(get_claim_ids_by_subject_token(token, scope=scope, limit=100))
        record_ids.update(get_record_ids_by_subject_token(token, scope=scope, limit=100))

    return EntityResolution(
        entity=token,
        kind=kind,
        anchors=tuple(anchors),
        episode_ids=frozenset(episode_ids),
        claim_ids=frozenset(claim_ids),
        record_ids=frozenset(record_ids),
    )


def entity_content_match_multiplier(entity: str, text: str) -> float:
    """Lightweight text overlap multiplier when DB linkage is incomplete."""
    token = _normalize_entity(entity).lower()
    if not token or not text:
        return 1.0

    lowered = str(text).lower()
    if token in lowered:
        return 1.12

    basename = _path_basename(token).lower()
    if basename and basename != token and basename in lowered:
        return 1.08

    return 1.0


__all__ = [
    "EntityResolution",
    "entity_content_match_multiplier",
    "resolve_entity_anchors",
    "resolve_entity_context",
]