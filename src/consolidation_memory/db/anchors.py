"""Entity-centric anchor lookups for recall."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters

_PATH_SUFFIX_LIMIT = 8


def get_episode_ids_by_entity_anchors(
    anchors: Sequence[Mapping[str, str]],
    *,
    scope: Mapping[str, Any] | None = None,
    limit: int = 200,
) -> list[str]:
    """Return episode IDs linked to any resolved entity anchor."""
    if not anchors:
        return []

    path_values: list[str] = []
    exact_pairs: list[tuple[str, str]] = []
    path_suffixes: list[str] = []
    seen_path: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    seen_suffix: set[str] = set()

    for anchor in anchors:
        anchor_type = str(anchor.get("anchor_type") or anchor.get("type") or "").strip()
        anchor_value = str(anchor.get("anchor_value") or anchor.get("value") or "").strip()
        if not anchor_type or not anchor_value:
            continue

        pair = (anchor_type, anchor_value)
        if pair not in seen_pair:
            seen_pair.add(pair)
            exact_pairs.append(pair)

        if anchor_type == "path":
            if anchor_value not in seen_path:
                seen_path.add(anchor_value)
                path_values.append(anchor_value)
            basename = anchor_value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if basename and basename not in seen_suffix and len(path_suffixes) < _PATH_SUFFIX_LIMIT:
                seen_suffix.add(basename)
                path_suffixes.append(basename)

    if not exact_pairs and not path_suffixes:
        return []

    max_results = max(1, int(limit))
    conditions: list[str] = []
    params: list[Any] = []
    _apply_scope_filters(conditions, params, scope, table_alias="e")

    anchor_clauses: list[str] = []
    anchor_params: list[Any] = []

    if exact_pairs:
        pair_placeholders = ",".join("(?, ?)" for _ in exact_pairs)
        anchor_clauses.append(f"(ea.anchor_type, ea.anchor_value) IN ({pair_placeholders})")
        for anchor_type, anchor_value in exact_pairs:
            anchor_params.extend([anchor_type, anchor_value])

    for suffix in path_suffixes:
        anchor_clauses.append("(ea.anchor_type = 'path' AND ea.anchor_value LIKE '%' || ?)")
        anchor_params.append(suffix)

    if not anchor_clauses:
        return []

    where_parts = [*conditions, f"({' OR '.join(anchor_clauses)})"]
    where = " AND ".join(where_parts)
    query_params = [*params, *anchor_params, max_results]
    sql = f"""SELECT DISTINCT ea.episode_id
              FROM episode_anchors ea
              JOIN episodes e ON e.id = ea.episode_id
              WHERE {where}
              ORDER BY ea.created_at DESC, ea.episode_id ASC
              LIMIT ?"""

    with get_connection() as conn:
        rows = conn.execute(sql, query_params).fetchall()
    return [str(row["episode_id"]) for row in rows]


def get_record_ids_by_subject_token(
    subject_token: str,
    *,
    scope: Mapping[str, Any] | None = None,
    limit: int = 100,
) -> list[str]:
    """Return knowledge record IDs whose structured subject/key matches the token."""
    token = str(subject_token or "").strip().lower()
    if not token:
        return []

    conditions = [
        "kr.deleted = 0",
        "("
        "LOWER(json_extract(kr.content, '$.subject')) = ? OR "
        "LOWER(json_extract(kr.content, '$.key')) = ? OR "
        "LOWER(kr.embedding_text) LIKE ?"
        ")",
    ]
    params: list[Any] = [token, token, f"%{token}%"]
    _apply_scope_filters(conditions, params, scope, table_alias="kr")

    where = " AND ".join(conditions)
    max_results = max(1, int(limit))
    sql = f"""SELECT kr.id
              FROM knowledge_records kr
              WHERE {where}
              ORDER BY kr.updated_at DESC, kr.id ASC
              LIMIT ?"""

    with get_connection() as conn:
        rows = conn.execute(sql, [*params, max_results]).fetchall()
    return [str(row["id"]) for row in rows]


def get_claim_ids_by_subject_token(
    subject_token: str,
    *,
    scope: Mapping[str, Any] | None = None,
    limit: int = 100,
) -> list[str]:
    """Return claim IDs whose payload subject/key matches the token."""
    token = str(subject_token or "").strip().lower()
    if not token:
        return []

    conditions = [
        "("
        "LOWER(json_extract(c.payload, '$.subject')) = ? OR "
        "LOWER(json_extract(c.payload, '$.key')) = ? OR "
        "LOWER(c.canonical_text) LIKE ?"
        ")",
    ]
    params: list[Any] = [token, token, f"%{token}%"]
    _apply_scope_filters(conditions, params, scope, table_alias="e")

    where = " AND ".join(conditions)
    max_results = max(1, int(limit))
    sql = f"""SELECT DISTINCT c.id
              FROM claims c
              LEFT JOIN claim_sources cs ON cs.claim_id = c.id
              LEFT JOIN episodes e ON e.id = cs.source_episode_id
              WHERE {where}
              ORDER BY c.updated_at DESC, c.id ASC
              LIMIT ?"""

    with get_connection() as conn:
        rows = conn.execute(sql, [*params, max_results]).fetchall()
    return [str(row["id"]) for row in rows]