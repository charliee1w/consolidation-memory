"""SQLite database for episode metadata and knowledge tracking.

Uses WAL mode for concurrent read (MCP server) / write (consolidation script).
Thread-local connection caching avoids per-operation open/close overhead.
Includes schema versioning with automatic migration.
"""

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.types import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RunStatus,
    StatsDict,
)
from consolidation_memory.utils import parse_datetime, parse_json_list

logger = logging.getLogger(__name__)

_local = threading.local()
_all_connections: list[sqlite3.Connection] = []  # Track all thread-local connections for cleanup
_conn_list_lock = threading.Lock()

# Cached FTS5 availability flag (None = not yet checked)
_fts5_available: bool | None = None
_fts5_lock = threading.Lock()

# ── Schema versioning ────────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 16

_DEFAULT_NAMESPACE_SLUG = "default"
_DEFAULT_NAMESPACE_SHARING_MODE = "private"
_DEFAULT_APP_CLIENT_NAME = "legacy_client"
_DEFAULT_APP_CLIENT_TYPE = "python_sdk"
_TOPIC_STORAGE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Future migrations go here: version -> list of SQL statements
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE episodes ADD COLUMN consolidation_attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN last_consolidation_attempt TEXT",
    ],
    3: [
        """CREATE TABLE IF NOT EXISTS consolidation_metrics (
            id                  TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            clusters_succeeded  INTEGER NOT NULL DEFAULT 0,
            clusters_failed     INTEGER NOT NULL DEFAULT 0,
            avg_confidence      REAL NOT NULL DEFAULT 0.0,
            episodes_processed  INTEGER NOT NULL DEFAULT 0,
            duration_seconds    REAL NOT NULL DEFAULT 0.0,
            api_calls           INTEGER NOT NULL DEFAULT 0,
            topics_created      INTEGER NOT NULL DEFAULT 0,
            topics_updated      INTEGER NOT NULL DEFAULT 0,
            episodes_pruned     INTEGER NOT NULL DEFAULT 0
        )""",
        "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON consolidation_metrics(timestamp)",
    ],
    4: [
        "CREATE INDEX IF NOT EXISTS idx_episodes_consolidation_attempts ON episodes(consolidation_attempts)",
    ],
    5: [
        """CREATE TABLE IF NOT EXISTS knowledge_records (
            id              TEXT PRIMARY KEY,
            topic_id        TEXT NOT NULL,
            record_type     TEXT NOT NULL,
            content         TEXT NOT NULL,
            embedding_text  TEXT NOT NULL,
            source_episodes TEXT NOT NULL DEFAULT '[]',
            confidence      REAL NOT NULL DEFAULT 0.8,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            access_count    INTEGER NOT NULL DEFAULT 0,
            deleted         INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (topic_id) REFERENCES knowledge_topics(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_records_topic ON knowledge_records(topic_id)",
        "CREATE INDEX IF NOT EXISTS idx_records_type ON knowledge_records(record_type)",
        "CREATE INDEX IF NOT EXISTS idx_records_deleted ON knowledge_records(deleted)",
    ],
    6: [
        "ALTER TABLE knowledge_records ADD COLUMN valid_from TEXT",
        "ALTER TABLE knowledge_records ADD COLUMN valid_until TEXT",
        "CREATE INDEX IF NOT EXISTS idx_records_valid_until ON knowledge_records(valid_until)",
    ],
    7: [
        "ALTER TABLE episodes ADD COLUMN protected INTEGER NOT NULL DEFAULT 0",
    ],
    8: [
        """CREATE TABLE IF NOT EXISTS contradiction_log (
            id              TEXT PRIMARY KEY,
            topic_id        TEXT,
            old_record_id   TEXT,
            new_record_id   TEXT,
            old_content     TEXT NOT NULL,
            new_content     TEXT NOT NULL,
            resolution      TEXT NOT NULL DEFAULT 'expired_old',
            reason          TEXT,
            detected_at     TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_contradiction_topic ON contradiction_log(topic_id)",
        "CREATE INDEX IF NOT EXISTS idx_contradiction_detected ON contradiction_log(detected_at)",
    ],
    9: [
        """CREATE TABLE IF NOT EXISTS tag_cooccurrence (
            tag_a       TEXT NOT NULL,
            tag_b       TEXT NOT NULL,
            count       INTEGER NOT NULL DEFAULT 1,
            last_seen   TEXT NOT NULL,
            UNIQUE(tag_a, tag_b)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cooccurrence_tag_a ON tag_cooccurrence(tag_a)",
        "CREATE INDEX IF NOT EXISTS idx_cooccurrence_tag_b ON tag_cooccurrence(tag_b)",
    ],
    # Migration 10 is applied specially in _apply_migration() because FTS5
    # may not be available in all SQLite builds.  The SQL list here is empty;
    # the real work happens in _apply_fts5_migration().
    10: [],
    11: [
        """CREATE TABLE IF NOT EXISTS claims (
            id              TEXT PRIMARY KEY,
            claim_type      TEXT NOT NULL,
            canonical_text  TEXT NOT NULL,
            payload         TEXT NOT NULL DEFAULT '{}',
            status          TEXT NOT NULL DEFAULT 'active',
            confidence      REAL NOT NULL DEFAULT 0.8,
            valid_from      TEXT NOT NULL,
            valid_until     TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_claims_temporal ON claims(valid_from, valid_until)",
        "CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status)",
        """CREATE TABLE IF NOT EXISTS claim_edges (
            id              TEXT PRIMARY KEY,
            from_claim_id   TEXT NOT NULL,
            to_claim_id     TEXT NOT NULL,
            edge_type       TEXT NOT NULL,
            confidence      REAL NOT NULL DEFAULT 1.0,
            details         TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (from_claim_id) REFERENCES claims(id),
            FOREIGN KEY (to_claim_id) REFERENCES claims(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_claim_edges_from_claim ON claim_edges(from_claim_id)",
        "CREATE INDEX IF NOT EXISTS idx_claim_edges_to_claim ON claim_edges(to_claim_id)",
        """CREATE TABLE IF NOT EXISTS claim_sources (
            id                  TEXT PRIMARY KEY,
            claim_id            TEXT NOT NULL,
            source_episode_id   TEXT,
            source_topic_id     TEXT,
            source_record_id    TEXT,
            created_at          TEXT NOT NULL,
            FOREIGN KEY (claim_id) REFERENCES claims(id),
            FOREIGN KEY (source_episode_id) REFERENCES episodes(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_claim_sources_claim_id ON claim_sources(claim_id)",
        "CREATE INDEX IF NOT EXISTS idx_claim_sources_episode ON claim_sources(source_episode_id)",
        """CREATE TABLE IF NOT EXISTS claim_events (
            id              TEXT PRIMARY KEY,
            claim_id        TEXT NOT NULL,
            event_type      TEXT NOT NULL,
            details         TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_claim_events_claim_created ON claim_events(claim_id, created_at)",
        """CREATE TABLE IF NOT EXISTS episode_anchors (
            id              TEXT PRIMARY KEY,
            episode_id      TEXT NOT NULL,
            anchor_type     TEXT NOT NULL,
            anchor_value    TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (episode_id) REFERENCES episodes(id),
            UNIQUE(episode_id, anchor_type, anchor_value)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_episode_anchors_lookup ON episode_anchors(anchor_type, anchor_value)",
        "CREATE INDEX IF NOT EXISTS idx_episode_anchors_episode ON episode_anchors(episode_id)",
    ],
    12: [
        """CREATE TABLE IF NOT EXISTS consolidation_scheduler (
            id                      TEXT PRIMARY KEY,
            last_run_started_at     TEXT,
            last_run_completed_at   TEXT,
            last_status             TEXT NOT NULL DEFAULT 'idle',
            last_error              TEXT,
            last_trigger            TEXT,
            last_utility_score      REAL,
            next_due_at             TEXT,
            lease_owner             TEXT,
            lease_expires_at        TEXT,
            updated_at              TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_scheduler_next_due ON consolidation_scheduler(next_due_at)",
        "CREATE INDEX IF NOT EXISTS idx_scheduler_lease_expires ON consolidation_scheduler(lease_expires_at)",
    ],
    # Migration 13 is applied specially in _apply_migration() so we can add
    # scope columns idempotently (only when a column does not already exist).
    13: [],
    # Migration 14 is applied specially in _apply_migration() for first-class
    # persisted policy/ACL entities.
    14: [],
    # Migration 15 is applied specially in _apply_migration() to decouple the
    # user-facing topic filename from the unique on-disk storage filename.
    15: [],
    # Migration 16 is applied specially in _apply_migration() to hide episodes
    # until their vectors are durably persisted.
    16: [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_utc_timestamp(value: str | datetime) -> str:
    """Normalize a timestamp to a UTC ISO 8601 string."""
    dt = parse_datetime(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _normalize_scope_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _default_project_slug() -> str:
    project = _normalize_scope_token(getattr(_get_config(), "active_project", None))
    return project or "default"


def _coerce_scope_row(scope: Mapping[str, Any] | None = None) -> dict[str, str | None]:
    raw = scope or {}
    project_slug = _normalize_scope_token(raw.get("project_slug")) or _default_project_slug()
    project_display_name = _normalize_scope_token(raw.get("project_display_name")) or project_slug
    return {
        "namespace_slug": _normalize_scope_token(raw.get("namespace_slug")) or _DEFAULT_NAMESPACE_SLUG,
        "namespace_sharing_mode": (
            _normalize_scope_token(raw.get("namespace_sharing_mode"))
            or _DEFAULT_NAMESPACE_SHARING_MODE
        ),
        "app_client_name": _normalize_scope_token(raw.get("app_client_name")) or _DEFAULT_APP_CLIENT_NAME,
        "app_client_type": _normalize_scope_token(raw.get("app_client_type")) or _DEFAULT_APP_CLIENT_TYPE,
        "app_client_provider": _normalize_scope_token(raw.get("app_client_provider")),
        "app_client_external_key": _normalize_scope_token(raw.get("app_client_external_key")),
        "agent_name": _normalize_scope_token(raw.get("agent_name")),
        "agent_external_key": _normalize_scope_token(raw.get("agent_external_key")),
        "session_external_key": _normalize_scope_token(raw.get("session_external_key")),
        "session_kind": _normalize_scope_token(raw.get("session_kind")),
        "project_slug": project_slug,
        "project_display_name": project_display_name,
        "project_root_uri": _normalize_scope_token(raw.get("project_root_uri")),
        "project_repo_remote": _normalize_scope_token(raw.get("project_repo_remote")),
        "project_default_branch": _normalize_scope_token(raw.get("project_default_branch")),
    }


def _topic_storage_filename(
    logical_filename: str,
    scope: Mapping[str, Any] | None = None,
) -> str:
    """Build a deterministic storage filename for a topic within one exact scope."""
    scope_row = _coerce_scope_row(scope)
    pure_name = PurePosixPath(str(logical_filename or "")).name
    suffix = "".join(PurePosixPath(pure_name).suffixes)
    stem = pure_name[: -len(suffix)] if suffix else pure_name
    cleaned_stem = _TOPIC_STORAGE_SAFE_RE.sub("_", stem).strip("._") or "topic"
    normalized_suffix = suffix or ".md"

    identity_tokens = [
        f"{key}={scope_row.get(key) or ''}"
        for key in _EXACT_SCOPE_MATCH_KEYS
    ]
    identity_payload = "|".join([*identity_tokens, f"filename={logical_filename}"])
    digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned_stem}__{digest}{normalized_suffix}"


def topic_storage_filename(topic: Mapping[str, Any]) -> str:
    """Return the unique on-disk filename for a knowledge topic row."""
    storage = topic.get("storage_filename")
    if isinstance(storage, str) and storage.strip():
        return storage.strip()
    filename = topic.get("filename")
    if isinstance(filename, str):
        return filename
    raise ValueError("Knowledge topic row is missing filename metadata")


def _apply_exact_scope_filters(
    conditions: list[str],
    params: list[Any],
    scope: Mapping[str, Any] | None = None,
    *,
    table_alias: str = "",
) -> None:
    """Apply exact-scope matching, including explicit NULL checks."""
    if scope is None:
        return

    prefix = f"{table_alias}." if table_alias else ""
    for key in _EXACT_SCOPE_MATCH_KEYS:
        value = _normalize_scope_token(scope.get(key))
        if value is None:
            conditions.append(f"{prefix}{key} IS NULL")
        else:
            conditions.append(f"{prefix}{key} = ?")
            params.append(value)


def _apply_scope_filters(
    conditions: list[str],
    params: list[Any],
    scope: Mapping[str, Any] | None = None,
    *,
    table_alias: str = "",
) -> None:
    if not scope:
        return

    prefix = f"{table_alias}." if table_alias else ""
    filter_keys = (
        "namespace_slug",
        "project_slug",
        "app_client_name",
        "app_client_type",
        "app_client_provider",
        "app_client_external_key",
        "agent_name",
        "agent_external_key",
        "session_external_key",
        "session_kind",
    )
    for key in filter_keys:
        value = _normalize_scope_token(scope.get(key))
        if value is None:
            continue
        conditions.append(f"{prefix}{key} = ?")
        params.append(value)


_POLICY_SELECTOR_KEYS: tuple[str, ...] = (
    "namespace_slug",
    "project_slug",
    "app_client_name",
    "app_client_type",
    "app_client_provider",
    "app_client_external_key",
    "agent_name",
    "agent_external_key",
    "session_external_key",
    "session_kind",
)

_EXACT_SCOPE_MATCH_KEYS: tuple[str, ...] = (
    "namespace_slug",
    "namespace_sharing_mode",
    "project_slug",
    "app_client_name",
    "app_client_type",
    "app_client_provider",
    "app_client_external_key",
    "agent_name",
    "agent_external_key",
    "session_external_key",
    "session_kind",
)


def _normalize_principal_token(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("principal value must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("principal value must be non-empty")
    return cleaned


def upsert_policy_principal(
    principal_type: str,
    principal_key: str,
    *,
    principal_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Create or reuse a persisted policy principal and return its ID."""
    normalized_type = _normalize_principal_token(principal_type)
    normalized_key = _normalize_principal_token(principal_key)

    def _upsert(active_conn: sqlite3.Connection) -> str:
        existing = active_conn.execute(
            """SELECT id
               FROM policy_principals
               WHERE principal_type = ? AND principal_key = ?""",
            (normalized_type, normalized_key),
        ).fetchone()
        if existing is not None:
            return str(existing["id"])

        resolved_id = principal_id or str(uuid.uuid4())
        active_conn.execute(
            """INSERT INTO policy_principals
               (id, principal_type, principal_key, created_at)
               VALUES (?, ?, ?, ?)""",
            (resolved_id, normalized_type, normalized_key, _now()),
        )
        return resolved_id

    if conn is not None:
        return _upsert(conn)
    with get_connection() as managed_conn:
        return _upsert(managed_conn)


def upsert_access_policy(
    *,
    namespace_slug: str | None = None,
    project_slug: str | None = None,
    app_client_name: str | None = None,
    app_client_type: str | None = None,
    app_client_provider: str | None = None,
    app_client_external_key: str | None = None,
    agent_name: str | None = None,
    agent_external_key: str | None = None,
    session_external_key: str | None = None,
    session_kind: str | None = None,
    enabled: bool = True,
    policy_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Insert/update an access policy row and return its ID."""
    now = _now()
    resolved_id = policy_id or str(uuid.uuid4())
    values = {
        "namespace_slug": _normalize_scope_token(namespace_slug),
        "project_slug": _normalize_scope_token(project_slug),
        "app_client_name": _normalize_scope_token(app_client_name),
        "app_client_type": _normalize_scope_token(app_client_type),
        "app_client_provider": _normalize_scope_token(app_client_provider),
        "app_client_external_key": _normalize_scope_token(app_client_external_key),
        "agent_name": _normalize_scope_token(agent_name),
        "agent_external_key": _normalize_scope_token(agent_external_key),
        "session_external_key": _normalize_scope_token(session_external_key),
        "session_kind": _normalize_scope_token(session_kind),
        "enabled": 1 if enabled else 0,
    }

    def _upsert(active_conn: sqlite3.Connection) -> str:
        active_conn.execute(
            """INSERT INTO access_policies
               (id, namespace_slug, project_slug, app_client_name, app_client_type,
                app_client_provider, app_client_external_key, agent_name, agent_external_key,
                session_external_key, session_kind, enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   namespace_slug = excluded.namespace_slug,
                   project_slug = excluded.project_slug,
                   app_client_name = excluded.app_client_name,
                   app_client_type = excluded.app_client_type,
                   app_client_provider = excluded.app_client_provider,
                   app_client_external_key = excluded.app_client_external_key,
                   agent_name = excluded.agent_name,
                   agent_external_key = excluded.agent_external_key,
                   session_external_key = excluded.session_external_key,
                   session_kind = excluded.session_kind,
                   enabled = excluded.enabled,
                   updated_at = excluded.updated_at""",
            (
                resolved_id,
                values["namespace_slug"],
                values["project_slug"],
                values["app_client_name"],
                values["app_client_type"],
                values["app_client_provider"],
                values["app_client_external_key"],
                values["agent_name"],
                values["agent_external_key"],
                values["session_external_key"],
                values["session_kind"],
                values["enabled"],
                now,
                now,
            ),
        )
        return resolved_id

    if conn is not None:
        return _upsert(conn)
    with get_connection() as managed_conn:
        return _upsert(managed_conn)


def upsert_policy_acl_entry(
    *,
    policy_id: str,
    principal_id: str,
    write_mode: str | None = None,
    read_visibility: str | None = None,
    acl_entry_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Insert/update a policy ACL entry and return its ID."""
    normalized_write_mode = _normalize_scope_token(write_mode)
    normalized_read_visibility = _normalize_scope_token(read_visibility)
    if normalized_write_mode not in {None, "allow", "deny"}:
        raise ValueError("write_mode must be one of: allow, deny")
    if normalized_read_visibility not in {None, "private", "project", "namespace"}:
        raise ValueError("read_visibility must be one of: private, project, namespace")
    if normalized_write_mode is None and normalized_read_visibility is None:
        raise ValueError("ACL entry requires write_mode and/or read_visibility")

    now = _now()
    resolved_id = acl_entry_id or str(uuid.uuid4())

    def _upsert(active_conn: sqlite3.Connection) -> str:
        existing = active_conn.execute(
            """SELECT id
               FROM policy_acl_entries
               WHERE policy_id = ?
                 AND principal_id = ?
                 AND ((write_mode IS NULL AND ? IS NULL) OR write_mode = ?)
                 AND ((read_visibility IS NULL AND ? IS NULL) OR read_visibility = ?)""",
            (
                policy_id,
                principal_id,
                normalized_write_mode,
                normalized_write_mode,
                normalized_read_visibility,
                normalized_read_visibility,
            ),
        ).fetchone()
        if existing is not None:
            active_conn.execute(
                "UPDATE policy_acl_entries SET updated_at = ? WHERE id = ?",
                (now, str(existing["id"])),
            )
            return str(existing["id"])

        active_conn.execute(
            """INSERT INTO policy_acl_entries
               (id, policy_id, principal_id, write_mode, read_visibility, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                resolved_id,
                policy_id,
                principal_id,
                normalized_write_mode,
                normalized_read_visibility,
                now,
                now,
            ),
        )
        return resolved_id

    if conn is not None:
        return _upsert(conn)
    with get_connection() as managed_conn:
        return _upsert(managed_conn)


def get_matching_policy_acl_entries(
    scope: Mapping[str, Any] | None,
    principals: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return ACL entries matching a scope envelope and principal tokens."""
    if not principals:
        return []

    scope_row = _coerce_scope_row(scope)
    scope_conditions = [
        f"(p.{key} IS NULL OR p.{key} = ?)"
        for key in _POLICY_SELECTOR_KEYS
    ]
    scope_params: list[Any] = [scope_row.get(key) for key in _POLICY_SELECTOR_KEYS]

    principal_conditions = []
    principal_params: list[Any] = []
    for principal_type, principal_key in principals:
        principal_conditions.append("(pp.principal_type = ? AND pp.principal_key = ?)")
        principal_params.append(_normalize_principal_token(principal_type))
        principal_params.append(_normalize_principal_token(principal_key))

    where = " AND ".join([
        "p.enabled = 1",
        *scope_conditions,
        f"({' OR '.join(principal_conditions)})",
    ])

    query = f"""SELECT
            pae.id AS acl_entry_id,
            pae.policy_id,
            pae.write_mode,
            pae.read_visibility,
            pp.id AS principal_id,
            pp.principal_type,
            pp.principal_key
        FROM policy_acl_entries pae
        JOIN access_policies p ON p.id = pae.policy_id
        JOIN policy_principals pp ON pp.id = pae.principal_id
        WHERE {where}
        ORDER BY p.updated_at DESC, pae.updated_at DESC, pae.id ASC"""  # nosec B608

    with get_connection() as conn:
        rows = conn.execute(
            query,
            [*scope_params, *principal_params],
        ).fetchall()
    return [dict(row) for row in rows]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _close_and_untrack_connection(conn: sqlite3.Connection) -> None:
    """Close a cached connection and remove it from global tracking."""
    try:
        conn.close()
    except Exception:  # nosec B110
        pass
    with _conn_list_lock:
        try:
            _all_connections.remove(conn)
        except ValueError:
            pass


def _get_cached_connection() -> sqlite3.Connection:
    """Return a thread-local cached connection. Creates one if needed."""
    current_db_path = str(_get_config().DB_PATH)
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    cached_db_path: str | None = getattr(_local, "db_path", None)
    if conn is not None:
        # Project switching updates config paths at runtime; if the DB path changed,
        # discard the stale thread-local connection and open a new one.
        if cached_db_path and cached_db_path != current_db_path:
            _close_and_untrack_connection(conn)
            _local.conn = None
            _local.db_path = None
            _local.conn_depth = 0
            conn = None
        else:
            try:
                conn.execute("SELECT 1")
                return conn
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                _close_and_untrack_connection(conn)
                _local.conn = None
                _local.db_path = None
                _local.conn_depth = 0
                conn = None

    _ensure_parent(_get_config().DB_PATH)
    # Connections are still thread-local for normal use, but we allow
    # cross-thread close during global teardown in tests/shutdown.
    conn = sqlite3.connect(current_db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    _local.db_path = current_db_path
    with _conn_list_lock:
        _all_connections.append(conn)
    return conn


def close_thread_local_connection() -> None:
    """Close only the current thread's cached connection, if present."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        _close_and_untrack_connection(conn)
    _local.conn = None
    _local.db_path = None
    _local.conn_depth = 0


def close_all_connections() -> None:
    """Close all thread-local connections. Call during shutdown or test teardown."""
    with _conn_list_lock:
        for conn in _all_connections:
            try:
                conn.close()
            except Exception:  # nosec B110
                pass
        _all_connections.clear()
    # Also clear this thread's cached reference
    _local.conn = None
    _local.db_path = None
    _local.conn_depth = 0
    # Reset FTS5 availability cache (new DB may or may not have FTS5)
    _reset_fts5_cache()


@contextmanager
def get_connection():
    """Yield a thread-local SQLite connection with commit/rollback.

    Re-entrant: nested calls share the same connection and only the
    outermost context manager commits or rolls back.
    """
    conn = _get_cached_connection()
    depth = int(getattr(_local, "conn_depth", 0))
    _local.conn_depth = depth + 1
    try:
        yield conn
        if depth == 0:
            conn.commit()
    except Exception:
        if depth == 0:
            try:
                conn.rollback()
            except Exception:  # nosec B110
                pass
        raise
    finally:
        _local.conn_depth = depth


def ensure_schema() -> None:
    with get_connection() as conn:
        # Individual execute() calls instead of executescript() — the latter
        # implicitly COMMITs before running, breaking transaction atomicity.
        conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
                id              TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                content         TEXT NOT NULL,
                content_type    TEXT NOT NULL DEFAULT 'exchange',
                tags            TEXT NOT NULL DEFAULT '[]',
                surprise_score  REAL NOT NULL DEFAULT 0.5,
                indexed         INTEGER NOT NULL DEFAULT 1,
                access_count    INTEGER NOT NULL DEFAULT 0,
                source_session  TEXT,
                consolidated    INTEGER NOT NULL DEFAULT 0,
                consolidated_at TEXT,
                consolidated_to TEXT,
                deleted         INTEGER NOT NULL DEFAULT 0
            )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(content_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_deleted ON episodes(deleted)"
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_topics (
                id              TEXT PRIMARY KEY,
                filename        TEXT NOT NULL,
                storage_filename TEXT NOT NULL UNIQUE,
                title           TEXT NOT NULL,
                summary         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                source_episodes TEXT NOT NULL DEFAULT '[]',
                fact_count      INTEGER NOT NULL DEFAULT 0,
                access_count    INTEGER NOT NULL DEFAULT 0,
                confidence      REAL NOT NULL DEFAULT 0.8
            )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_filename ON knowledge_topics(filename)"
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_runs (
                id                  TEXT PRIMARY KEY,
                started_at          TEXT NOT NULL,
                completed_at        TEXT,
                episodes_processed  INTEGER NOT NULL DEFAULT 0,
                clusters_formed     INTEGER NOT NULL DEFAULT 0,
                topics_created      INTEGER NOT NULL DEFAULT 0,
                topics_updated      INTEGER NOT NULL DEFAULT 0,
                episodes_pruned     INTEGER NOT NULL DEFAULT 0,
                status              TEXT NOT NULL DEFAULT 'running',
                error_message       TEXT
            )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER NOT NULL,
                applied_at  TEXT NOT NULL
            )""")

        # Check and apply migrations
        _check_and_migrate(conn)
        _repair_schema_invariants(conn)


def _check_and_migrate(conn: sqlite3.Connection) -> None:
    """Check current schema version and apply pending migrations.

    Each version's migration runs inside a SAVEPOINT so a crash mid-migration
    leaves the database in the last fully-applied version rather than a
    half-migrated state.
    """
    row = conn.execute(
        "SELECT MAX(version) as v FROM schema_version"
    ).fetchone()
    current = row["v"] if row and row["v"] is not None else 0

    if current == 0:
        # First time: apply all migrations then record current version.
        # The base CREATE TABLE statements only define the v1 schema;
        # columns and tables added in later migrations must still be executed.
        for version in range(2, CURRENT_SCHEMA_VERSION + 1):
            _apply_migration(conn, version)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (CURRENT_SCHEMA_VERSION, _now()),
        )
        return

    if current >= CURRENT_SCHEMA_VERSION:
        return

    # Apply pending migrations, each in its own savepoint
    for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
        _apply_migration(conn, version)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, _now()),
        )


def _apply_migration(conn: sqlite3.Connection, version: int) -> None:
    """Apply a single migration version inside a SAVEPOINT for atomicity."""
    if version not in MIGRATIONS:
        return
    if not isinstance(version, int) or version < 0:
        raise ValueError(f"Invalid migration version: {version}")
    if version == 15:
        _apply_topic_storage_migration(conn)
        return
    if version == 16:
        _apply_episode_index_visibility_migration(conn)
        return
    savepoint = f"migration_v{version}"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for sql in MIGRATIONS[version]:
            conn.execute(sql)
        # FTS5 migration — may fail if FTS5 extension is not compiled in
        if version == 10:
            _apply_fts5_migration(conn)
        if version == 13:
            _apply_scope_migration(conn)
        if version == 14:
            _apply_policy_acl_migration(conn)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        if version == 10:
            # FTS5 unavailable — log and continue, recall degrades gracefully
            logger.warning("FTS5 not available, hybrid search will be disabled")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            return
        raise


def _apply_fts5_migration(conn: sqlite3.Connection) -> None:
    """Create FTS5 virtual table and backfill from existing episodes."""
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            episode_id UNINDEXED,
            content
        )"""
    )
    # Guard against duplicate inserts if migration re-runs
    conn.execute(
        """INSERT INTO episodes_fts(episode_id, content)
           SELECT id, content FROM episodes
           WHERE deleted = 0
             AND id NOT IN (SELECT episode_id FROM episodes_fts)"""
    )


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _add_column_if_missing(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    existing = _get_table_columns(conn, table_name)
    if column_name in existing:
        return
    try:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
    except sqlite3.OperationalError:
        # Another process may have added the column after our PRAGMA check.
        if column_name in _get_table_columns(conn, table_name):
            return
        raise


def _repair_schema_invariants(conn: sqlite3.Connection) -> None:
    """Repair idempotent runtime-required schema invariants.

    ``schema_version`` reflects the expected migration history, but existing
    user databases can drift when a prior release recorded a version while
    leaving an additive column behind. Re-apply safe additive repairs here so
    startup is resilient even when the version marker is ahead of the actual
    table shape.
    """
    _apply_topic_storage_migration(conn)
    _apply_episode_index_visibility_migration(conn)


def _apply_scope_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v13: persistent scope columns for shared memory isolation."""
    shared_columns = (
        ("namespace_slug", "namespace_slug TEXT NOT NULL DEFAULT 'default'"),
        ("namespace_sharing_mode", "namespace_sharing_mode TEXT NOT NULL DEFAULT 'private'"),
        ("app_client_name", "app_client_name TEXT NOT NULL DEFAULT 'legacy_client'"),
        ("app_client_type", "app_client_type TEXT NOT NULL DEFAULT 'python_sdk'"),
        ("app_client_provider", "app_client_provider TEXT"),
        ("app_client_external_key", "app_client_external_key TEXT"),
        ("agent_name", "agent_name TEXT"),
        ("agent_external_key", "agent_external_key TEXT"),
        ("session_external_key", "session_external_key TEXT"),
        ("session_kind", "session_kind TEXT"),
        ("project_slug", "project_slug TEXT NOT NULL DEFAULT 'default'"),
        ("project_display_name", "project_display_name TEXT"),
        ("project_root_uri", "project_root_uri TEXT"),
        ("project_repo_remote", "project_repo_remote TEXT"),
        ("project_default_branch", "project_default_branch TEXT"),
    )

    for table_name in ("episodes", "knowledge_records", "knowledge_topics"):
        for column_name, column_sql in shared_columns:
            _add_column_if_missing(
                conn,
                table_name=table_name,
                column_name=column_name,
                column_sql=column_sql,
            )

    project_slug = _default_project_slug()
    for table_name in ("episodes", "knowledge_records", "knowledge_topics"):
        conn.execute(
            f"""UPDATE {table_name}
                SET namespace_slug = COALESCE(NULLIF(namespace_slug, ''), ?),
                    namespace_sharing_mode = COALESCE(NULLIF(namespace_sharing_mode, ''), ?),
                    app_client_name = COALESCE(NULLIF(app_client_name, ''), ?),
                    app_client_type = COALESCE(NULLIF(app_client_type, ''), ?),
                    project_slug = CASE
                        WHEN project_slug IS NULL OR project_slug = '' OR project_slug = 'default'
                        THEN ?
                        ELSE project_slug
                    END,
                    project_display_name = COALESCE(
                        NULLIF(project_display_name, ''),
                        CASE
                            WHEN project_slug IS NULL OR project_slug = '' OR project_slug = 'default'
                            THEN ?
                            ELSE project_slug
                        END
                    )""",  # nosec B608
            (
                _DEFAULT_NAMESPACE_SLUG,
                _DEFAULT_NAMESPACE_SHARING_MODE,
                _DEFAULT_APP_CLIENT_NAME,
                _DEFAULT_APP_CLIENT_TYPE,
                project_slug,
                project_slug,
            ),
        )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_scope_ns_project ON episodes(namespace_slug, project_slug)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_scope_app ON episodes(namespace_slug, project_slug, app_client_name, app_client_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_scope_agent_session ON episodes(namespace_slug, project_slug, agent_external_key, session_external_key)"
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_scope_ns_project ON knowledge_records(namespace_slug, project_slug)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_scope_app ON knowledge_records(namespace_slug, project_slug, app_client_name, app_client_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_scope_agent_session ON knowledge_records(namespace_slug, project_slug, agent_external_key, session_external_key)"
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_topics_scope_ns_project ON knowledge_topics(namespace_slug, project_slug)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_topics_scope_app ON knowledge_topics(namespace_slug, project_slug, app_client_name, app_client_type)"
    )


def _apply_policy_acl_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v14: persisted policy/ACL entities."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS policy_principals (
            id              TEXT PRIMARY KEY,
            principal_type  TEXT NOT NULL,
            principal_key   TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            UNIQUE(principal_type, principal_key)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS access_policies (
            id                      TEXT PRIMARY KEY,
            namespace_slug          TEXT,
            project_slug            TEXT,
            app_client_name         TEXT,
            app_client_type         TEXT,
            app_client_provider     TEXT,
            app_client_external_key TEXT,
            agent_name              TEXT,
            agent_external_key      TEXT,
            session_external_key    TEXT,
            session_kind            TEXT,
            enabled                 INTEGER NOT NULL DEFAULT 1,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS policy_acl_entries (
            id              TEXT PRIMARY KEY,
            policy_id       TEXT NOT NULL,
            principal_id    TEXT NOT NULL,
            write_mode      TEXT CHECK(write_mode IN ('allow', 'deny')),
            read_visibility TEXT CHECK(read_visibility IN ('private', 'project', 'namespace')),
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            FOREIGN KEY (policy_id) REFERENCES access_policies(id) ON DELETE CASCADE,
            FOREIGN KEY (principal_id) REFERENCES policy_principals(id) ON DELETE CASCADE,
            CHECK(write_mode IS NOT NULL OR read_visibility IS NOT NULL),
            UNIQUE(policy_id, principal_id, write_mode, read_visibility)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_policy_principals_type_key ON policy_principals(principal_type, principal_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_access_policies_scope ON access_policies(namespace_slug, project_slug, app_client_name, app_client_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_policy_acl_entries_policy ON policy_acl_entries(policy_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_policy_acl_entries_principal ON policy_acl_entries(principal_id)"
    )


def _apply_topic_storage_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v15: topic display filenames decoupled from storage filenames."""
    columns = _get_table_columns(conn, "knowledge_topics")
    if "storage_filename" in columns:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_storage_filename "
            "ON knowledge_topics(storage_filename)"
        )
        return

    foreign_keys_enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if foreign_keys_enabled:
        conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute(
            """CREATE TABLE knowledge_topics_v15 (
                id               TEXT PRIMARY KEY,
                filename         TEXT NOT NULL,
                storage_filename TEXT NOT NULL UNIQUE,
                title            TEXT NOT NULL,
                summary          TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                source_episodes  TEXT NOT NULL DEFAULT '[]',
                fact_count       INTEGER NOT NULL DEFAULT 0,
                access_count     INTEGER NOT NULL DEFAULT 0,
                confidence       REAL NOT NULL DEFAULT 0.8,
                namespace_slug          TEXT NOT NULL DEFAULT 'default',
                namespace_sharing_mode  TEXT NOT NULL DEFAULT 'private',
                app_client_name         TEXT NOT NULL DEFAULT 'legacy_client',
                app_client_type         TEXT NOT NULL DEFAULT 'python_sdk',
                app_client_provider     TEXT,
                app_client_external_key TEXT,
                agent_name              TEXT,
                agent_external_key      TEXT,
                session_external_key    TEXT,
                session_kind            TEXT,
                project_slug            TEXT NOT NULL DEFAULT 'default',
                project_display_name    TEXT,
                project_root_uri        TEXT,
                project_repo_remote     TEXT,
                project_default_branch  TEXT
            )"""
        )
        conn.execute(
            """INSERT INTO knowledge_topics_v15 (
                id, filename, storage_filename, title, summary, created_at, updated_at,
                source_episodes, fact_count, access_count, confidence,
                namespace_slug, namespace_sharing_mode,
                app_client_name, app_client_type, app_client_provider, app_client_external_key,
                agent_name, agent_external_key, session_external_key, session_kind,
                project_slug, project_display_name, project_root_uri,
                project_repo_remote, project_default_branch
            )
            SELECT
                id, filename, filename, title, summary, created_at, updated_at,
                source_episodes, fact_count, access_count, confidence,
                namespace_slug, namespace_sharing_mode,
                app_client_name, app_client_type, app_client_provider, app_client_external_key,
                agent_name, agent_external_key, session_external_key, session_kind,
                project_slug, project_display_name, project_root_uri,
                project_repo_remote, project_default_branch
            FROM knowledge_topics"""
        )
        conn.execute("DROP TABLE knowledge_topics")
        conn.execute("ALTER TABLE knowledge_topics_v15 RENAME TO knowledge_topics")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_filename ON knowledge_topics(filename)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_storage_filename "
            "ON knowledge_topics(storage_filename)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_topics_scope_ns_project "
            "ON knowledge_topics(namespace_slug, project_slug)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_topics_scope_app "
            "ON knowledge_topics(namespace_slug, project_slug, app_client_name, app_client_type)"
        )
    finally:
        if foreign_keys_enabled:
            conn.execute("PRAGMA foreign_keys=ON")


def _apply_episode_index_visibility_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v16: stage episode visibility on vector durability."""
    _add_column_if_missing(
        conn,
        table_name="episodes",
        column_name="indexed",
        column_sql="indexed INTEGER NOT NULL DEFAULT 1",
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_indexed ON episodes(indexed)")


# ── Episode CRUD ─────────────────────────────────────────────────────────────

def insert_episode(
    content: str,
    content_type: str = "exchange",
    tags: list[str] | None = None,
    surprise_score: float = 0.5,
    source_session: str | None = None,
    scope: Mapping[str, Any] | None = None,
    episode_id: str | None = None,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
    access_count: int | None = None,
    consolidated: int | None = None,
    consolidated_at: str | None = None,
    consolidated_to: str | None = None,
    deleted: int | None = None,
    consolidation_attempts: int | None = None,
    last_consolidation_attempt: str | None = None,
    protected: int | None = None,
    indexed: int | bool | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    if episode_id is None:
        episode_id = str(uuid.uuid4())
    created_ts = created_at or _now()
    updated_ts = updated_at or created_ts
    scope_row = _coerce_scope_row(scope)
    def _insert(active_conn: sqlite3.Connection) -> None:
        active_conn.execute(
            """INSERT INTO episodes
               (id, created_at, updated_at, content, content_type, tags,
                surprise_score, indexed, access_count, source_session,
                consolidated, consolidated_at, consolidated_to, deleted,
                consolidation_attempts, last_consolidation_attempt, protected,
                namespace_slug, namespace_sharing_mode,
                app_client_name, app_client_type, app_client_provider, app_client_external_key,
                agent_name, agent_external_key,
                session_external_key, session_kind,
                project_slug, project_display_name, project_root_uri,
                project_repo_remote, project_default_branch)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                created_ts,
                updated_ts,
                content,
                content_type,
                json.dumps(tags or []),
                surprise_score,
                1 if indexed is None else int(bool(indexed)),
                0 if access_count is None else int(access_count),
                source_session,
                0 if consolidated is None else int(consolidated),
                consolidated_at,
                consolidated_to,
                0 if deleted is None else int(deleted),
                0 if consolidation_attempts is None else int(consolidation_attempts),
                last_consolidation_attempt,
                0 if protected is None else int(protected),
             scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
             scope_row["app_client_name"], scope_row["app_client_type"],
             scope_row["app_client_provider"], scope_row["app_client_external_key"],
             scope_row["agent_name"], scope_row["agent_external_key"],
             scope_row["session_external_key"], scope_row["session_kind"],
             scope_row["project_slug"], scope_row["project_display_name"],
             scope_row["project_root_uri"], scope_row["project_repo_remote"],
             scope_row["project_default_branch"],
            ),
        )
        # FTS insert within the same transaction for atomicity
        fts_insert(episode_id, content)
    if conn is None:
        with get_connection() as managed_conn:
            _insert(managed_conn)
    else:
        _insert(conn)
    return episode_id


def get_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
    *,
    include_unindexed: bool = False,
) -> dict[str, Any] | None:
    conditions = ["id = ?", "deleted = 0"]
    if not include_unindexed:
        conditions.append("indexed = 1")
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM episodes WHERE {where_clause}",  # nosec B608
            params,
        ).fetchone()
    return dict(row) if row else None


def get_episodes_batch(
    episode_ids: list[str],
    *,
    include_unindexed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch multiple episodes in a single query. Returns {id: episode_dict}."""
    if not episode_ids:
        return {}
    conditions = ["id IN ({placeholders})", "deleted = 0"]
    if not include_unindexed:
        conditions.append("indexed = 1")
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        where_clause = " AND ".join(condition.format(placeholders=placeholders) for condition in conditions)
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE {where_clause}",  # nosec B608
            episode_ids,
        ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def get_unindexed_episodes(limit: int = 200) -> list[dict[str, Any]]:
    """Return non-deleted episodes whose vectors are not yet marked durable."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM episodes
               WHERE deleted = 0 AND indexed = 0
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_episode_indexed(
    episode_ids: Sequence[str],
    *,
    indexed: bool = True,
) -> int:
    """Mark episodes visible once their vectors are durably persisted."""
    if not episode_ids:
        return 0
    placeholders = ",".join("?" for _ in episode_ids)
    with get_connection() as conn:
        cursor = conn.execute(
            f"""UPDATE episodes
                SET indexed = ?
                WHERE id IN ({placeholders})""",  # nosec B608
            [1 if indexed else 0, *episode_ids],
        )
    return int(cursor.rowcount or 0)


def get_existing_episode_ids(
    episode_ids: Sequence[str],
    *,
    include_deleted: bool = False,
) -> set[str]:
    """Return the subset of provided episode IDs that currently exist in SQLite."""
    if not episode_ids:
        return set()
    placeholders = ",".join("?" for _ in episode_ids)
    conditions = [f"id IN ({placeholders})"]  # nosec B608
    if not include_deleted:
        conditions.append("deleted = 0")
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id FROM episodes WHERE {where_clause}",  # nosec B608
            list(episode_ids),
        ).fetchall()
    return {str(row["id"]) for row in rows}


def get_unconsolidated_episodes(limit: int = 200, max_attempts: int = 5) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM episodes
               WHERE consolidated = 0 AND deleted = 0 AND indexed = 1 AND consolidation_attempts < ?
               ORDER BY created_at DESC LIMIT ?""",
            (max_attempts, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def increment_access(episode_ids: list[str]) -> None:
    if not episode_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        query = f"""UPDATE episodes SET access_count = access_count + 1,
            updated_at = ? WHERE id IN ({placeholders})"""  # nosec B608
        conn.execute(
            query,
            [_now()] + episode_ids,
        )


def mark_consolidated(episode_ids: list[str], topic_filename: str) -> None:
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        query = f"""UPDATE episodes SET consolidated = 1,
            consolidated_at = ?, consolidated_to = ?, updated_at = ?
            WHERE id IN ({placeholders})"""  # nosec B608
        conn.execute(
            query,
            [now, topic_filename, now] + episode_ids,
        )


def mark_pruned(episode_ids: list[str]) -> None:
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        query = f"""UPDATE episodes SET consolidated = 2, updated_at = ?
            WHERE id IN ({placeholders}) AND consolidated = 1"""  # nosec B608
        conn.execute(
            query,
            [now] + episode_ids,
        )


def soft_delete_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
) -> bool:
    conditions = ["id = ?", "deleted = 0", "indexed = 1"]
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    now = _now()
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE episodes SET deleted = 1, updated_at = ? WHERE {where_clause}",  # nosec B608
            [now, *params],
        )
        deleted = bool(cursor.rowcount and cursor.rowcount > 0)
        if deleted:
            fts_delete(episode_id)
    return deleted


def restore_soft_deleted_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
) -> bool:
    """Restore a soft-deleted episode.

    Used as a compensating action when vector tombstoning fails during forget().
    """
    conditions = ["id = ?", "deleted = 1", "indexed = 1"]
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    now = _now()

    content: str | None = None
    restored = False
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT content FROM episodes WHERE {where_clause}",  # nosec B608
            params,
        ).fetchone()
        if row is None:
            return False
        content = str(row["content"])
        cursor = conn.execute(
            f"UPDATE episodes SET deleted = 0, updated_at = ? WHERE {where_clause}",  # nosec B608
            [now, *params],
        )
        restored = bool(cursor.rowcount and cursor.rowcount > 0)

    if restored and content is not None:
        fts_insert(episode_id, content)
    return restored


def hard_delete_episode(episode_id: str) -> bool:
    """Permanently delete an episode from the database.

    Used for rollback when FAISS add fails — soft-delete would leave an orphan
    that dedup checks still find.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM episodes WHERE id = ?", (episode_id,)
        )
        deleted = bool(cursor.rowcount and cursor.rowcount > 0)
        if deleted:
            fts_delete(episode_id)
    return deleted


# ── FTS5 Full-Text Search ─────────────────────────────────────────────────

def fts_available() -> bool:
    """Check if the FTS5 virtual table exists. Result is cached."""
    global _fts5_available
    if _fts5_available is not None:
        return _fts5_available
    with _fts5_lock:
        if _fts5_available is not None:
            return _fts5_available
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes_fts'"
                ).fetchone()
                _fts5_available = row is not None
        except Exception:
            _fts5_available = False
    return _fts5_available


def _reset_fts5_cache() -> None:
    """Reset the FTS5 availability cache. Used by tests."""
    global _fts5_available
    with _fts5_lock:
        _fts5_available = None


def fts_insert(episode_id: str, content: str) -> None:
    """Insert an episode into the FTS5 index."""
    if not fts_available():
        return
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO episodes_fts(episode_id, content) VALUES (?, ?)",
                (episode_id, content),
            )
    except Exception as e:
        logger.warning("FTS5 insert failed for %s: %s", episode_id, e)


def fts_delete(episode_id: str) -> None:
    """Delete an episode from the FTS5 index."""
    if not fts_available():
        return
    try:
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM episodes_fts WHERE episode_id = ?",
                (episode_id,),
            )
    except Exception as e:
        logger.warning("FTS5 delete failed for %s: %s", episode_id, e)


_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query for FTS5 MATCH.

    Splits into terms, strips non-word characters, drops single-char tokens,
    double-quotes FTS5 reserved operators so they're treated as literals,
    and joins with OR so documents matching any term are returned.
    """
    terms = re.findall(r'\w+', query)
    terms = [t for t in terms if len(t) > 1]
    if not terms:
        return ""
    # Double-quote terms that match FTS5 operators so they're treated as literals
    safe_terms = [f'"{t}"' if t.upper() in _FTS5_OPERATORS else t for t in terms]
    return " OR ".join(safe_terms)


def fts_search(query: str, limit: int = 50) -> list[tuple[str, float]]:
    """BM25 keyword search over episodes.

    Returns list of (episode_id, normalized_bm25_score) sorted by relevance.
    Normalization: score = raw / (raw + 1.0) where raw = -bm25().
    """
    if not fts_available():
        return []
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT episode_id, bm25(episodes_fts) as rank
                   FROM episodes_fts
                   WHERE episodes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (sanitized, limit),
            ).fetchall()
        results = []
        for row in rows:
            raw = -row["rank"]  # bm25() returns negative values; negate for positive
            normalized = raw / (raw + 1.0) if raw > 0 else 0.0
            results.append((row["episode_id"], normalized))
        return results
    except Exception as e:
        logger.warning("FTS5 search failed: %s", e)
        return []


def fts_rebuild() -> None:
    """Rebuild the FTS5 index from the episodes table."""
    if not fts_available():
        return
    with get_connection() as conn:
        conn.execute("DELETE FROM episodes_fts")
        conn.execute(
            """INSERT INTO episodes_fts(episode_id, content)
               SELECT id, content FROM episodes WHERE deleted = 0"""
        )


def get_prunable_episodes(days: int = 30) -> list[dict[str, Any]]:
    """Episodes that are consolidated and older than `days`, excluding protected ones."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM episodes
               WHERE consolidated = 1 AND consolidated_at < ? AND deleted = 0 AND indexed = 1
                 AND protected = 0""",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def protect_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
) -> bool:
    """Mark an episode as protected from pruning. Returns True if found."""
    now = _now()
    conditions = ["id = ?", "deleted = 0"]
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE episodes SET protected = 1, updated_at = ? WHERE {where_clause}",  # nosec B608
            [now, *params],
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def protect_by_tag(
    tag: str,
    scope: Mapping[str, Any] | None = None,
) -> int:
    """Mark all episodes with a given tag as protected. Returns count updated."""
    now = _now()
    # Tags are stored as JSON arrays, use LIKE with the tag value.
    # Escape LIKE wildcards in the tag to prevent injection (e.g. tag="%" matching all).
    escaped = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f'%"{escaped}"%'
    conditions = [
        "tags LIKE ? ESCAPE '\\'",
        "deleted = 0",
        "indexed = 1",
        "protected = 0",
    ]
    params: list[Any] = [pattern]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE episodes SET protected = 1, updated_at = ? "
            f"WHERE {where_clause}",  # nosec B608
            [now, *params],
        )
    return cursor.rowcount or 0


def get_low_confidence_records(threshold: float = 0.5) -> list[dict[str, Any]]:
    """Return active records below the confidence threshold."""
    now = _now()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
               FROM knowledge_records kr
               JOIN knowledge_topics kt ON kr.topic_id = kt.id
               WHERE kr.deleted = 0
                 AND (kr.valid_from IS NULL OR julianday(kr.valid_from) <= julianday(?))
                 AND (kr.valid_until IS NULL OR julianday(kr.valid_until) > julianday(?))
                 AND kr.confidence < ?
               ORDER BY kr.confidence ASC""",
            (now, now, threshold),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Knowledge Topic CRUD ────────────────────────────────────────────────────

def upsert_knowledge_topic(
    filename: str,
    title: str,
    summary: str,
    source_episodes: list[str],
    fact_count: int = 0,
    confidence: float = 0.8,
    scope: Mapping[str, Any] | None = None,
    topic_id: str | None = None,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
    access_count: int | None = None,
) -> str:
    now = _now()
    updated_ts = updated_at or now
    scope_row = _coerce_scope_row(scope)
    storage_filename = _topic_storage_filename(filename, scope_row)
    resolved_topic_id: str | None = topic_id
    with get_connection() as conn:
        existing = None
        if resolved_topic_id is not None:
            existing = conn.execute(
                "SELECT id, source_episodes, access_count, storage_filename "
                "FROM knowledge_topics WHERE id = ?",
                (resolved_topic_id,),
            ).fetchone()
        if existing is None:
            conditions = ["(filename = ? OR storage_filename = ?)"]
            params: list[Any] = [filename, filename]
            _apply_exact_scope_filters(conditions, params, scope_row)
            existing = conn.execute(
                f"""SELECT id, source_episodes, access_count, storage_filename
                    FROM knowledge_topics
                    WHERE {' AND '.join(conditions)}
                    ORDER BY updated_at DESC, id ASC
                    LIMIT 1""",  # nosec B608
                params,
            ).fetchone()

        if existing:
            resolved_topic_id = str(existing["id"])
            old_sources = json.loads(existing["source_episodes"])
            merged = list(set(old_sources + source_episodes))
            persisted_storage_filename = str(existing["storage_filename"] or storage_filename)
            conn.execute(
                """UPDATE knowledge_topics
                   SET title = ?, summary = ?, updated_at = ?,
                       source_episodes = ?, fact_count = ?, access_count = ?, confidence = ?,
                       storage_filename = ?,
                       namespace_slug = ?, namespace_sharing_mode = ?,
                       app_client_name = ?, app_client_type = ?, app_client_provider = ?, app_client_external_key = ?,
                       agent_name = ?, agent_external_key = ?,
                       session_external_key = ?, session_kind = ?,
                       project_slug = ?, project_display_name = ?, project_root_uri = ?,
                       project_repo_remote = ?, project_default_branch = ?
                   WHERE id = ?""",
                (
                    title,
                    summary,
                    updated_ts,
                    json.dumps(merged),
                    fact_count,
                    int(existing["access_count"]) if access_count is None else int(access_count),
                    confidence,
                    persisted_storage_filename,
                 scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
                 scope_row["app_client_name"], scope_row["app_client_type"],
                 scope_row["app_client_provider"], scope_row["app_client_external_key"],
                 scope_row["agent_name"], scope_row["agent_external_key"],
                 scope_row["session_external_key"], scope_row["session_kind"],
                 scope_row["project_slug"], scope_row["project_display_name"],
                 scope_row["project_root_uri"], scope_row["project_repo_remote"],
                 scope_row["project_default_branch"], resolved_topic_id,
                ),
            )
        else:
            resolved_topic_id = resolved_topic_id or str(uuid.uuid4())
            created_ts = created_at or updated_ts
            try:
                conn.execute(
                    """INSERT INTO knowledge_topics
                       (id, filename, storage_filename, title, summary, created_at, updated_at,
                        source_episodes, fact_count, access_count, confidence,
                        namespace_slug, namespace_sharing_mode,
                        app_client_name, app_client_type, app_client_provider, app_client_external_key,
                        agent_name, agent_external_key,
                        session_external_key, session_kind,
                        project_slug, project_display_name, project_root_uri,
                        project_repo_remote, project_default_branch)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        resolved_topic_id,
                        filename,
                        storage_filename,
                        title,
                        summary,
                        created_ts,
                        updated_ts,
                        json.dumps(source_episodes),
                        fact_count,
                        0 if access_count is None else int(access_count),
                        confidence,
                     scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
                     scope_row["app_client_name"], scope_row["app_client_type"],
                     scope_row["app_client_provider"], scope_row["app_client_external_key"],
                     scope_row["agent_name"], scope_row["agent_external_key"],
                     scope_row["session_external_key"], scope_row["session_kind"],
                     scope_row["project_slug"], scope_row["project_display_name"],
                     scope_row["project_root_uri"], scope_row["project_repo_remote"],
                     scope_row["project_default_branch"],
                    ),
                )
            except sqlite3.IntegrityError:
                # Race: concurrent insert won — fall back to update
                conditions = ["(filename = ? OR storage_filename = ?)"]
                params = [filename, filename]
                _apply_exact_scope_filters(conditions, params, scope_row)
                existing = conn.execute(
                    f"""SELECT id, source_episodes, access_count, storage_filename
                        FROM knowledge_topics
                        WHERE {' AND '.join(conditions)}
                        ORDER BY updated_at DESC, id ASC
                        LIMIT 1""",  # nosec B608
                    params,
                ).fetchone()
                if existing is None:
                    raise
                resolved_topic_id = str(existing["id"])
                old_sources = json.loads(existing["source_episodes"])
                merged = list(set(old_sources + source_episodes))
                persisted_storage_filename = str(existing["storage_filename"] or storage_filename)
                conn.execute(
                    """UPDATE knowledge_topics
                       SET title = ?, summary = ?, updated_at = ?,
                           source_episodes = ?, fact_count = ?, access_count = ?, confidence = ?,
                           storage_filename = ?,
                           namespace_slug = ?, namespace_sharing_mode = ?,
                           app_client_name = ?, app_client_type = ?, app_client_provider = ?, app_client_external_key = ?,
                           agent_name = ?, agent_external_key = ?,
                           session_external_key = ?, session_kind = ?,
                           project_slug = ?, project_display_name = ?, project_root_uri = ?,
                           project_repo_remote = ?, project_default_branch = ?
                       WHERE id = ?""",
                    (
                        title,
                        summary,
                        updated_ts,
                        json.dumps(merged),
                        fact_count,
                        int(existing["access_count"]) if access_count is None else int(access_count),
                        confidence,
                        persisted_storage_filename,
                     scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
                     scope_row["app_client_name"], scope_row["app_client_type"],
                     scope_row["app_client_provider"], scope_row["app_client_external_key"],
                     scope_row["agent_name"], scope_row["agent_external_key"],
                     scope_row["session_external_key"], scope_row["session_kind"],
                     scope_row["project_slug"], scope_row["project_display_name"],
                     scope_row["project_root_uri"], scope_row["project_repo_remote"],
                     scope_row["project_default_branch"], resolved_topic_id,
                    ),
                )
    if resolved_topic_id is None:
        raise RuntimeError("Failed to resolve knowledge topic id during upsert")
    return resolved_topic_id


def get_all_knowledge_topics(
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    _apply_scope_filters(conditions, params, scope)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM knowledge_topics {where} ORDER BY updated_at DESC",  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_knowledge_topic(
    filename: str,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    conditions = ["(filename = ? OR storage_filename = ?)"]
    params: list[Any] = [filename, filename]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM knowledge_topics WHERE {where_clause}",  # nosec B608
            params,
        ).fetchone()
    return dict(row) if row else None


def get_knowledge_topics_by_name(
    filename: str,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    conditions = ["(filename = ? OR storage_filename = ?)"]
    params: list[Any] = [filename, filename]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT * FROM knowledge_topics
                WHERE {where_clause}
                ORDER BY updated_at DESC, id ASC""",  # nosec B608
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def increment_topic_access(
    filenames: list[str],
    *,
    scope: Mapping[str, Any] | None = None,
) -> None:
    """Increment topic access counters using legacy filename-based lookup.

    This compatibility surface is still used by call sites that only have the
    topic filename. Prefer ``increment_topic_access_by_ids`` for unambiguous
    updates when topic IDs are available.
    """
    deduped_filenames: list[str] = []
    seen: set[str] = set()
    for filename in filenames:
        token = str(filename or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped_filenames.append(token)

    if not deduped_filenames:
        return

    now = _now()
    with get_connection() as conn:
        # Keep chunks well under SQLite's default parameter limit.
        for start in range(0, len(deduped_filenames), 250):
            chunk = deduped_filenames[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            conditions: list[str] = [
                f"(filename IN ({placeholders}) OR storage_filename IN ({placeholders}))",
            ]
            params: list[Any] = [now, *chunk, *chunk]
            _apply_scope_filters(conditions, params, scope)
            where_clause = " AND ".join(conditions)
            conn.execute(
                f"""UPDATE knowledge_topics
                    SET access_count = access_count + 1,
                        updated_at = ?
                    WHERE {where_clause}""",  # nosec B608
                params,
            )


def increment_topic_access_by_ids(topic_ids: list[str]) -> None:
    if not topic_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in topic_ids)
        query = f"""UPDATE knowledge_topics SET access_count = access_count + 1,
            updated_at = ? WHERE id IN ({placeholders})"""  # nosec B608
        conn.execute(
            query,
            [_now()] + topic_ids,
        )


# ── Knowledge Record CRUD ─────────────────────────────────────────────────

def insert_knowledge_records(
    topic_id: str,
    records: list[dict[str, Any]],
    source_episodes: list[str] | None = None,
    scope: Mapping[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Insert multiple knowledge records for a topic.

    Each record dict must have: record_type, content (JSON-serializable dict),
    embedding_text. Optional: confidence, valid_from.

    Returns list of inserted record IDs.
    """
    if not records:
        return []
    now = _now()
    ids: list[str] = []

    def _coerce_int_column(value: object | None, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, (int, float, bool)):
            return int(value)
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return int(value)
            except ValueError:
                return default
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    def _insert_rows(active_conn: sqlite3.Connection) -> None:
        for rec in records:
            rec_id = str(rec.get("id") or uuid.uuid4())
            content = rec["content"] if isinstance(rec["content"], str) else json.dumps(rec["content"])
            valid_from = rec.get("valid_from")
            valid_until = rec.get("valid_until")
            created_ts = str(rec.get("created_at") or now)
            updated_ts = str(rec.get("updated_at") or created_ts)
            access_count_value = _coerce_int_column(rec.get("access_count"))
            deleted_value = _coerce_int_column(rec.get("deleted"))
            record_scope_row = _coerce_scope_row(
                rec["scope"] if isinstance(rec.get("scope"), Mapping) else scope
            )
            record_source_episodes_raw = rec.get("source_episodes", source_episodes or [])
            if isinstance(record_source_episodes_raw, str):
                record_source_episodes = parse_json_list(record_source_episodes_raw)
            else:
                record_source_episodes = list(record_source_episodes_raw or [])
            active_conn.execute(
                """INSERT INTO knowledge_records
                   (id, topic_id, record_type, content, embedding_text,
                    source_episodes, confidence, created_at, updated_at, access_count,
                    deleted, valid_from, valid_until,
                    namespace_slug, namespace_sharing_mode,
                    app_client_name, app_client_type, app_client_provider, app_client_external_key,
                    agent_name, agent_external_key,
                    session_external_key, session_kind,
                    project_slug, project_display_name, project_root_uri,
                    project_repo_remote, project_default_branch)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec_id,
                    topic_id,
                    rec["record_type"],
                    content,
                    rec["embedding_text"],
                    json.dumps(record_source_episodes),
                    rec.get("confidence", 0.8),
                    created_ts,
                    updated_ts,
                    access_count_value,
                    deleted_value,
                    valid_from,
                    valid_until,
                    record_scope_row["namespace_slug"],
                    record_scope_row["namespace_sharing_mode"],
                    record_scope_row["app_client_name"],
                    record_scope_row["app_client_type"],
                    record_scope_row["app_client_provider"],
                    record_scope_row["app_client_external_key"],
                    record_scope_row["agent_name"],
                    record_scope_row["agent_external_key"],
                    record_scope_row["session_external_key"],
                    record_scope_row["session_kind"],
                    record_scope_row["project_slug"],
                    record_scope_row["project_display_name"],
                    record_scope_row["project_root_uri"],
                    record_scope_row["project_repo_remote"],
                    record_scope_row["project_default_branch"],
                ),
            )
            ids.append(rec_id)

    if conn is None:
        with get_connection() as managed_conn:
            _insert_rows(managed_conn)
    else:
        _insert_rows(conn)
    return ids


def expire_record(record_id: str, valid_until: str | None = None) -> bool:
    """Set valid_until on a record, marking it as temporally superseded.

    Unlike soft-delete, expired records retain their content and can be
    retrieved with include_expired=True for historical queries.

    Returns True if a record was updated.
    """
    ts = valid_until or _now()
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE knowledge_records SET valid_until = ?, updated_at = ? WHERE id = ? AND deleted = 0",
            (ts, _now(), record_id),
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def insert_contradiction(
    topic_id: str | None,
    old_record_id: str | None,
    new_record_id: str | None,
    old_content: str,
    new_content: str,
    resolution: str = "expired_old",
    reason: str | None = None,
) -> str:
    """Log a detected contradiction between knowledge records."""
    contradiction_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO contradiction_log
               (id, topic_id, old_record_id, new_record_id,
                old_content, new_content, resolution, reason, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (contradiction_id, topic_id, old_record_id, new_record_id,
             old_content, new_content, resolution, reason, _now()),
        )
    return contradiction_id


def get_contradictions(
    topic_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve logged contradictions, optionally filtered by topic.

    Args:
        topic_id: If provided, filter to contradictions for this topic.
        limit: Max results (default 50).

    Returns:
        List of contradiction dicts, newest first.
    """
    with get_connection() as conn:
        if topic_id:
            rows = conn.execute(
                """SELECT cl.*, kt.title as topic_title, kt.filename as topic_filename
                   FROM contradiction_log cl
                   LEFT JOIN knowledge_topics kt ON cl.topic_id = kt.id
                   WHERE cl.topic_id = ?
                   ORDER BY cl.detected_at DESC LIMIT ?""",
                (topic_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT cl.*, kt.title as topic_title, kt.filename as topic_filename
                   FROM contradiction_log cl
                   LEFT JOIN knowledge_topics kt ON cl.topic_id = kt.id
                   ORDER BY cl.detected_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_recently_contradicted_topic_ids(days: int = 30) -> set[str]:
    """Return topic IDs that have had contradictions detected within the last N days."""
    from datetime import timedelta
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff_dt.isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT topic_id FROM contradiction_log
               WHERE topic_id IS NOT NULL AND detected_at >= ?""",
            (cutoff_str,),
        ).fetchall()
    return {row["topic_id"] for row in rows}


def count_contradictions_since(since: str) -> int:
    """Return contradiction count with detected_at >= since."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS c
               FROM contradiction_log
               WHERE detected_at >= ?""",
            (since,),
        ).fetchone()
    return int(row["c"]) if row else 0


def update_tag_cooccurrence(tags: list[str]) -> None:
    """Update co-occurrence counts for all tag pairs in a set.

    Maintains the invariant tag_a < tag_b for consistent pair ordering.
    """
    if len(tags) < 2:
        return
    now = _now()
    # Generate unique sorted pairs
    unique_tags = sorted(set(tags))
    pairs = []
    for i in range(len(unique_tags)):
        for j in range(i + 1, len(unique_tags)):
            pairs.append((unique_tags[i], unique_tags[j]))

    with get_connection() as conn:
        for tag_a, tag_b in pairs:
            conn.execute(
                """INSERT INTO tag_cooccurrence (tag_a, tag_b, count, last_seen)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(tag_a, tag_b)
                   DO UPDATE SET count = count + 1, last_seen = ?""",
                (tag_a, tag_b, now, now),
            )


def get_cooccurring_tags(tags: list[str], min_count: int = 2) -> dict[str, int]:
    """Find tags that frequently co-occur with the given tags.

    Args:
        tags: Tags to find co-occurrences for.
        min_count: Minimum co-occurrence count to include.

    Returns:
        Dict mapping co-occurring tag to total co-occurrence count.
    """
    if not tags:
        return {}
    with get_connection() as conn:
        placeholders_a = ",".join("?" for _ in tags)
        placeholders_b = ",".join("?" for _ in tags)
        query = f"""SELECT tag_b as tag, SUM(count) as total
            FROM tag_cooccurrence
            WHERE tag_a IN ({placeholders_a}) AND count >= ?
            GROUP BY tag_b
            UNION ALL
            SELECT tag_a as tag, SUM(count) as total
            FROM tag_cooccurrence
            WHERE tag_b IN ({placeholders_b}) AND count >= ?
            GROUP BY tag_a"""  # nosec B608
        rows = conn.execute(
            query,
            [*tags, min_count, *tags, min_count],
        ).fetchall()

    # Aggregate results and exclude the input tags themselves
    tag_set = set(tags)
    result: dict[str, int] = {}
    for row in rows:
        tag = row["tag"]
        if tag not in tag_set:
            result[tag] = result.get(tag, 0) + row["total"]
    return result


def get_tag_pairs_in_set(tags: list[str], min_count: int = 2) -> list[tuple[str, str, int]]:
    """Find co-occurrence pairs where both tags are within the given set.

    Used to discover intra-candidate tag clusters during recall.

    Returns:
        List of (tag_a, tag_b, count) tuples.
    """
    if len(tags) < 2:
        return []
    with get_connection() as conn:
        placeholders_a = ",".join("?" for _ in tags)
        placeholders_b = ",".join("?" for _ in tags)
        query = f"""SELECT tag_a, tag_b, count FROM tag_cooccurrence
            WHERE tag_a IN ({placeholders_a})
              AND tag_b IN ({placeholders_b})
              AND count >= ?"""  # nosec B608
        rows = conn.execute(
            query,
            [*tags, *tags, min_count],
        ).fetchall()
    return [(row["tag_a"], row["tag_b"], row["count"]) for row in rows]


def get_all_active_records(
    include_expired: bool = False,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return all non-deleted knowledge records.

    Args:
        include_expired: If False (default), exclude records where
            the validity window does not include the current time.
    """
    now = _now()
    base_conditions: list[str] = ["kr.deleted = 0"]
    base_params: list[Any] = []
    _apply_scope_filters(base_conditions, base_params, scope, table_alias="kr")
    with get_connection() as conn:
        if include_expired:
            where_clause = " AND ".join(base_conditions)
            query = f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
               FROM knowledge_records kr
               JOIN knowledge_topics kt ON kr.topic_id = kt.id
               WHERE {where_clause}
               ORDER BY kr.updated_at DESC"""  # nosec B608
            rows = conn.execute(
                query,
                base_params,
            ).fetchall()
        else:
            timed_conditions = [
                *base_conditions,
                "(kr.valid_from IS NULL OR julianday(kr.valid_from) <= julianday(?))",
                "(kr.valid_until IS NULL OR julianday(kr.valid_until) > julianday(?))",
            ]
            timed_params = [*base_params, now, now]
            where_clause = " AND ".join(timed_conditions)
            query = f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
               FROM knowledge_records kr
               JOIN knowledge_topics kt ON kr.topic_id = kt.id
               WHERE {where_clause}
               ORDER BY kr.updated_at DESC"""  # nosec B608
            rows = conn.execute(
                query,
                timed_params,
            ).fetchall()
    return [dict(r) for r in rows]


def get_records_as_of(
    as_of: str,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return knowledge records as they existed at a specific point in time.

    Returns non-deleted records that were created on or before ``as_of`` and
    had not yet been superseded (expired) at that time.  This enables
    "what did I believe about X at time T?" queries.

    A record is considered valid at time T when:
    - ``created_at <= T`` (the record existed)
    - ``valid_from IS NULL OR valid_from <= T`` (it had become visible)
    - ``valid_until IS NULL OR valid_until > T`` (not yet superseded)

    Args:
        as_of: ISO 8601 datetime string representing the point in time.
    """
    as_of_utc = _normalize_utc_timestamp(as_of)
    conditions: list[str] = [
        "kr.deleted = 0",
        "julianday(kr.created_at) <= julianday(?)",
        "(kr.valid_from IS NULL OR julianday(kr.valid_from) <= julianday(?))",
        "(kr.valid_until IS NULL OR julianday(kr.valid_until) > julianday(?))",
    ]
    params: list[Any] = [as_of_utc, as_of_utc, as_of_utc]
    _apply_scope_filters(conditions, params, scope, table_alias="kr")
    where_clause = " AND ".join(conditions)
    query = f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
       FROM knowledge_records kr
       JOIN knowledge_topics kt ON kr.topic_id = kt.id
       WHERE {where_clause}
       ORDER BY kr.updated_at DESC"""  # nosec B608
    with get_connection() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_records_by_topic(topic_id: str, include_expired: bool = False) -> list[dict[str, Any]]:
    """Return all active records for a specific topic.

    Args:
        include_expired: If False (default), exclude records outside the current
            validity window.
    """
    with get_connection() as conn:
        if include_expired:
            rows = conn.execute(
                "SELECT * FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_id,),
            ).fetchall()
        else:
            now = _now()
            rows = conn.execute(
                """SELECT * FROM knowledge_records WHERE topic_id = ? AND deleted = 0
                   AND (valid_from IS NULL OR julianday(valid_from) <= julianday(?))
                   AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))""",
                (topic_id, now, now),
            ).fetchall()
    return [dict(r) for r in rows]


def soft_delete_records_by_topic(topic_id: str) -> int:
    """Soft-delete all records for a topic. Returns count of affected rows."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE knowledge_records SET deleted = 1, updated_at = ? WHERE topic_id = ? AND deleted = 0",
            (_now(), topic_id),
        )
    return int(cursor.rowcount)


def soft_delete_records_by_ids(record_ids: list[str]) -> int:
    """Soft-delete specific records by their IDs. Returns count of affected rows."""
    if not record_ids:
        return 0
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in record_ids)
        cursor = conn.execute(
            f"UPDATE knowledge_records SET deleted = 1, updated_at = ? WHERE id IN ({placeholders}) AND deleted = 0",  # nosec B608
            [now] + record_ids,
        )
    return int(cursor.rowcount)


def increment_record_access(record_ids: list[str]) -> None:
    """Increment access count for the given records."""
    if not record_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in record_ids)
        query = f"""UPDATE knowledge_records SET access_count = access_count + 1,
            updated_at = ? WHERE id IN ({placeholders})"""  # nosec B608
        conn.execute(
            query,
            [_now()] + record_ids,
        )


def get_record_count(include_expired: bool = False) -> int:
    """Return count of active (non-deleted) knowledge records."""
    with get_connection() as conn:
        if include_expired:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_records WHERE deleted = 0"
            ).fetchone()
        else:
            now = _now()
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM knowledge_records
                   WHERE deleted = 0
                     AND (valid_from IS NULL OR julianday(valid_from) <= julianday(?))
                     AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))""",
                (now, now),
            ).fetchone()
    return row["cnt"] if row else 0


# -- Claim Graph CRUD ---------------------------------------------------------

def upsert_claim(
    claim_id: str,
    claim_type: str,
    canonical_text: str,
    payload: dict[str, Any] | str | None = None,
    status: str = "active",
    confidence: float = 0.8,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> str:
    """Insert or update a claim row by ID."""
    now = _now()
    payload_text = payload if isinstance(payload, str) else json.dumps(payload or {})
    valid_from_ts = valid_from or now

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO claims
               (id, claim_type, canonical_text, payload, status, confidence,
                valid_from, valid_until, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   claim_type = excluded.claim_type,
                   canonical_text = excluded.canonical_text,
                   payload = excluded.payload,
                   status = excluded.status,
                   confidence = excluded.confidence,
                   valid_from = excluded.valid_from,
                   valid_until = excluded.valid_until,
                   updated_at = excluded.updated_at""",
            (
                claim_id,
                claim_type,
                canonical_text,
                payload_text,
                status,
                confidence,
                valid_from_ts,
                valid_until,
                now,
                now,
            ),
        )
    return claim_id


def get_active_claims(
    claim_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return currently active claims, optionally filtered by type."""
    now = _now()
    bounded_limit = max(1, int(limit))
    bounded_offset = max(0, int(offset))
    conditions = [
        "status = ?",
        "julianday(valid_from) <= julianday(?)",
        "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
    ]
    params: list[Any] = ["active", now, now]

    if claim_type:
        conditions.append("claim_type = ?")
        params.append(claim_type)

    where = " AND ".join(conditions)
    params.extend([bounded_limit, bounded_offset])
    query = f"""SELECT * FROM claims
        WHERE {where}
        ORDER BY updated_at DESC, id ASC
        LIMIT ? OFFSET ?"""  # nosec B608

    with get_connection() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_claims_as_of(
    as_of: str,
    claim_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return claims valid at a specific point in time."""
    as_of_utc = _normalize_utc_timestamp(as_of)
    bounded_limit = max(1, int(limit))
    bounded_offset = max(0, int(offset))
    conditions = [
        "julianday(valid_from) <= julianday(?)",
        "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
    ]
    params: list[Any] = [as_of_utc, as_of_utc]

    if claim_type:
        conditions.append("claim_type = ?")
        params.append(claim_type)

    where = " AND ".join(conditions)
    params.extend([bounded_limit, bounded_offset])
    query = f"""SELECT c.*,
                   CASE
                       WHEN c.status = 'challenged'
                            AND COALESCE(ch.first_challenged_at, julianday(c.updated_at)) > julianday(?)
                           THEN 'active'
                       WHEN c.status = 'challenged'
                           THEN 'challenged'
                       WHEN c.status = 'expired'
                            AND ch.first_challenged_at IS NOT NULL
                            AND ch.first_challenged_at <= julianday(?)
                           THEN 'challenged'
                       WHEN c.status = 'expired'
                           THEN 'active'
                       ELSE c.status
                   END AS snapshot_status
            FROM claims c
            LEFT JOIN (
                SELECT claim_id, MIN(julianday(created_at)) AS first_challenged_at
                FROM claim_events
                WHERE event_type = 'challenged'
                GROUP BY claim_id
            ) ch ON ch.claim_id = c.id
            WHERE {where}
            ORDER BY c.updated_at DESC, c.id ASC
            LIMIT ? OFFSET ?"""  # nosec B608

    with get_connection() as conn:
        rows = conn.execute(
            query,
            [as_of_utc, as_of_utc, *params],
        ).fetchall()
    claims = [dict(r) for r in rows]
    for claim in claims:
        claim["status"] = claim.pop("snapshot_status")
    return claims


def get_claim_source_scope_rows(
    claim_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return scope metadata for each claim based on its provenance sources."""
    if not claim_ids:
        return {}
    placeholders = ",".join("?" for _ in claim_ids)
    query = f"""SELECT
                cs.claim_id,
                COALESCE(e.namespace_slug, kr.namespace_slug, kt.namespace_slug) AS namespace_slug,
                COALESCE(e.project_slug, kr.project_slug, kt.project_slug) AS project_slug,
                COALESCE(e.app_client_name, kr.app_client_name, kt.app_client_name) AS app_client_name,
                COALESCE(e.app_client_type, kr.app_client_type, kt.app_client_type) AS app_client_type,
                COALESCE(e.app_client_provider, kr.app_client_provider, kt.app_client_provider) AS app_client_provider,
                COALESCE(e.app_client_external_key, kr.app_client_external_key, kt.app_client_external_key) AS app_client_external_key,
                COALESCE(e.agent_name, kr.agent_name, kt.agent_name) AS agent_name,
                COALESCE(e.agent_external_key, kr.agent_external_key, kt.agent_external_key) AS agent_external_key,
                COALESCE(e.session_external_key, kr.session_external_key, kt.session_external_key) AS session_external_key,
                COALESCE(e.session_kind, kr.session_kind, kt.session_kind) AS session_kind
            FROM claim_sources cs
            LEFT JOIN episodes e ON cs.source_episode_id = e.id
            LEFT JOIN knowledge_records kr ON cs.source_record_id = kr.id
            LEFT JOIN knowledge_topics kt ON cs.source_topic_id = kt.id
            WHERE cs.claim_id IN ({placeholders})"""  # nosec B608
    with get_connection() as conn:
        rows = conn.execute(
            query,
            list(claim_ids),
        ).fetchall()

    grouped: dict[str, list[dict[str, Any]]] = {str(cid): [] for cid in claim_ids}
    for row in rows:
        grouped[str(row["claim_id"])].append(dict(row))
    return grouped


def expire_claim(claim_id: str, valid_until: str | None = None) -> bool:
    """Expire a claim by setting status=expired and valid_until."""
    ts = valid_until or _now()
    with get_connection() as conn:
        cursor = conn.execute(
            """UPDATE claims
               SET status = 'expired',
                   valid_until = CASE
                       WHEN valid_until IS NULL OR valid_until > ? THEN ?
                       ELSE valid_until
                   END,
                   updated_at = ?
               WHERE id = ?""",
            (ts, ts, _now(), claim_id),
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def count_active_challenged_claims(as_of: str | None = None) -> int:
    """Return count of challenged claims that are still temporally valid."""
    ts = _normalize_utc_timestamp(as_of or _now())
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS c
               FROM claims
               WHERE status = 'challenged'
                 AND julianday(valid_from) <= julianday(?)
                 AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))""",
            (ts, ts),
        ).fetchone()
    return int(row["c"]) if row else 0


def auto_expire_stale_challenged_claims(
    *,
    max_age_hours: float,
    max_claims: int = 200,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Expire stale challenged claims and record audit events.

    Claims remain in `challenged` status until an explicit resolution. To prevent
    unbounded challenged backlogs, this helper expires challenged claims whose
    earliest challenge event is older than the configured age threshold.
    """
    age_hours = max(float(max_age_hours), 0.0)
    if age_hours <= 0:
        raise ValueError("max_age_hours must be > 0")

    limit = max(int(max_claims), 1)
    as_of_utc = _normalize_utc_timestamp(as_of or _now())
    as_of_dt = parse_datetime(as_of_utc)
    cutoff = (as_of_dt - timedelta(hours=age_hours)).isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT c.id
                 FROM claims c
                WHERE c.status = 'challenged'
                  AND julianday(c.valid_from) <= julianday(?)
                  AND (c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))
                  AND julianday(
                        COALESCE(
                            (
                                SELECT MIN(ce.created_at)
                                  FROM claim_events ce
                                 WHERE ce.claim_id = c.id
                                   AND ce.event_type = 'challenged'
                            ),
                            c.updated_at
                        )
                      ) <= julianday(?)
                ORDER BY c.updated_at ASC, c.id ASC
                LIMIT ?""",
            (as_of_utc, as_of_utc, cutoff, limit),
        ).fetchall()
        stale_ids = [row["id"] for row in rows]
        if not stale_ids:
            return {
                "as_of": as_of_utc,
                "cutoff": cutoff,
                "max_age_hours": age_hours,
                "max_claims": limit,
                "expired_count": 0,
                "expired_claim_ids": [],
            }

        placeholders = ",".join("?" for _ in stale_ids)
        conn.execute(
            f"""UPDATE claims
                    SET status = 'expired',
                        valid_until = CASE
                            WHEN valid_until IS NULL OR julianday(valid_until) > julianday(?) THEN ?
                            ELSE valid_until
                        END,
                        updated_at = ?
                  WHERE id IN ({placeholders})
                    AND status = 'challenged'""",  # nosec B608
            [as_of_utc, as_of_utc, as_of_utc, *stale_ids],
        )

        details_payload = json.dumps(
            {
                "policy": "auto_expire_stale_challenged",
                "max_age_hours": age_hours,
                "cutoff": cutoff,
                "expired_at": as_of_utc,
            },
            default=str,
        )
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    claim_id,
                    "auto_expired_challenged",
                    details_payload,
                    as_of_utc,
                )
                for claim_id in stale_ids
            ],
        )

    return {
        "as_of": as_of_utc,
        "cutoff": cutoff,
        "max_age_hours": age_hours,
        "max_claims": limit,
        "expired_count": len(stale_ids),
        "expired_claim_ids": stale_ids,
    }


def insert_claim_edge(
    from_claim_id: str,
    to_claim_id: str,
    edge_type: str,
    confidence: float = 1.0,
    details: dict[str, Any] | str | None = None,
    edge_id: str | None = None,
) -> str:
    """Insert an edge between two claims."""
    eid = edge_id or str(uuid.uuid4())
    details_text = details if isinstance(details, str) else (
        json.dumps(details) if details is not None else None
    )
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO claim_edges
               (id, from_claim_id, to_claim_id, edge_type, confidence, details, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, from_claim_id, to_claim_id, edge_type, confidence, details_text, _now()),
        )
    return eid


def insert_claim_sources(claim_id: str, sources: list[dict[str, Any]]) -> list[str]:
    """Insert source links for a claim. Returns inserted source IDs."""
    if not sources:
        return []

    now = _now()
    source_ids: list[str] = []
    with get_connection() as conn:
        for source in sources:
            source_id = str(source.get("id") or uuid.uuid4())
            source_episode_id = source.get("source_episode_id") or source.get("episode_id")
            source_topic_id = source.get("source_topic_id") or source.get("topic_id")
            source_record_id = source.get("source_record_id") or source.get("record_id")

            conn.execute(
                """INSERT INTO claim_sources
                   (id, claim_id, source_episode_id, source_topic_id, source_record_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    source_id,
                    claim_id,
                    source_episode_id,
                    source_topic_id,
                    source_record_id,
                    now,
                ),
            )
            source_ids.append(source_id)
    return source_ids


def insert_claim_event(
    claim_id: str,
    event_type: str,
    details: dict[str, Any] | str | None = None,
    event_id: str | None = None,
    created_at: str | None = None,
) -> str:
    """Insert a claim lifecycle event."""
    eid = event_id or str(uuid.uuid4())
    details_text = details if isinstance(details, str) else (
        json.dumps(details) if details is not None else None
    )
    ts = created_at or _now()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (eid, claim_id, event_type, details_text, ts),
        )
    return eid


def insert_claim_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Insert multiple claim lifecycle events in a single transaction."""
    if not events:
        return []

    now = _now()
    normalized_rows: list[tuple[str, str, str, str | None, str]] = []
    inserted_ids: list[str] = []
    for event in events:
        claim_id = str(event.get("claim_id") or "").strip()
        event_type = str(event.get("event_type") or "").strip()
        if not claim_id or not event_type:
            continue

        event_id = str(event.get("id") or uuid.uuid4())
        details_raw = event.get("details")
        details_text = (
            details_raw
            if isinstance(details_raw, str)
            else (json.dumps(details_raw) if details_raw is not None else None)
        )
        created_at_raw = event.get("created_at")
        created_at = str(created_at_raw) if created_at_raw is not None else now

        normalized_rows.append((event_id, claim_id, event_type, details_text, created_at))
        inserted_ids.append(event_id)

    if not normalized_rows:
        return []

    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            normalized_rows,
        )

    return inserted_ids


def insert_episode_anchors(episode_id: str, anchors: Sequence[Mapping[str, Any]]) -> list[str]:
    """Insert anchors for an episode. Duplicate anchors are ignored."""
    if not anchors:
        return []

    normalized: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor in anchors:
        anchor_type = str(anchor.get("anchor_type") or anchor.get("type") or "").strip()
        anchor_value = str(anchor.get("anchor_value") or anchor.get("value") or "").strip()
        if not anchor_type or not anchor_value:
            continue

        pair = (anchor_type, anchor_value)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        anchor_id = str(
            anchor.get("id")
            or uuid.uuid5(uuid.NAMESPACE_URL, f"{episode_id}:{anchor_type}:{anchor_value}")
        )
        normalized.append((anchor_id, anchor_type, anchor_value))

    if not normalized:
        return []

    inserted_ids: list[str] = []
    with get_connection() as conn:
        for anchor_id, anchor_type, anchor_value in normalized:
            cursor = conn.execute(
                """INSERT INTO episode_anchors
                   (id, episode_id, anchor_type, anchor_value, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(episode_id, anchor_type, anchor_value) DO NOTHING""",
                (anchor_id, episode_id, anchor_type, anchor_value, _now()),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                inserted_ids.append(anchor_id)
    return inserted_ids


def get_claims_by_anchor(
    anchor_type: str | None = None,
    anchor_value: str | None = None,
    include_expired: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch claims linked to episodes matching the provided anchor filter."""
    if not anchor_type and not anchor_value:
        return []

    conditions: list[str] = []
    params: list[Any] = []
    if anchor_type:
        conditions.append("ea.anchor_type = ?")
        params.append(anchor_type)
    if anchor_value:
        conditions.append("ea.anchor_value = ?")
        params.append(anchor_value)
    if not include_expired:
        now = _now()
        conditions.append("julianday(c.valid_from) <= julianday(?)")
        conditions.append("(c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))")
        params.extend([now, now])

    where = " AND ".join(conditions)
    params.append(limit)
    query = f"""SELECT DISTINCT
                c.id, c.claim_type, c.canonical_text, c.payload, c.status,
                c.confidence, c.valid_from, c.valid_until, c.created_at, c.updated_at
            FROM claims c
            JOIN claim_sources cs ON cs.claim_id = c.id
            JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
            WHERE {where}
            ORDER BY c.updated_at DESC, c.id ASC
            LIMIT ?"""  # nosec B608

    with get_connection() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_claims_by_anchor_values(
    anchor_type: str,
    anchor_values: Sequence[str],
    include_expired: bool = False,
    scope: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch claims linked to episodes matching any anchor value for one anchor type."""
    anchor_type_token = str(anchor_type or "").strip()
    if not anchor_type_token:
        return []

    deduped_values: list[str] = []
    seen_values: set[str] = set()
    for value in anchor_values:
        token = str(value or "").strip()
        if not token or token in seen_values:
            continue
        seen_values.add(token)
        deduped_values.append(token)

    if not deduped_values:
        return []

    max_results = limit if limit is None else max(1, int(limit))
    params_prefix: list[Any] = [anchor_type_token]
    common_conditions = ["ea.anchor_type = ?"]
    _apply_scope_filters(common_conditions, params_prefix, scope, table_alias="e")
    if not include_expired:
        now = _now()
        common_conditions.extend(
            [
                "julianday(c.valid_from) <= julianday(?)",
                "(c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))",
            ]
        )
        params_prefix.extend([now, now])

    rows: list[dict[str, Any]] = []
    remaining = max_results
    # Keep chunks well under SQLite's default parameter limit.
    with get_connection() as conn:
        for start in range(0, len(deduped_values), 250):
            if remaining is not None and remaining <= 0:
                break
            chunk = deduped_values[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            where = " AND ".join([*common_conditions, f"ea.anchor_value IN ({placeholders})"])
            query_params: list[Any] = [*params_prefix, *chunk]
            sql = f"""SELECT DISTINCT
                        c.id, c.claim_type, c.canonical_text, c.payload, c.status,
                        c.confidence, c.valid_from, c.valid_until, c.created_at, c.updated_at,
                        ea.anchor_value
                    FROM claims c
                    JOIN claim_sources cs ON cs.claim_id = c.id
                    JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
                    JOIN episodes e ON e.id = ea.episode_id
                    WHERE {where}
                    ORDER BY c.updated_at DESC, c.id ASC"""  # nosec B608
            if remaining is not None:
                sql += " LIMIT ?"
                query_params.append(remaining)

            chunk_rows = conn.execute(sql, query_params).fetchall()

            mapped_chunk = [dict(row) for row in chunk_rows]
            rows.extend(mapped_chunk)
            if remaining is not None:
                remaining -= len(mapped_chunk)

    return rows


def mark_claims_challenged_by_ids(
    claim_ids: Sequence[str],
    challenged_at: str | None = None,
) -> list[str]:
    """Mark active claims as challenged for a known set of impacted claim IDs."""
    deduped_ids: list[str] = []
    seen_ids: set[str] = set()
    for claim_id in claim_ids:
        token = str(claim_id or "").strip()
        if not token or token in seen_ids:
            continue
        seen_ids.add(token)
        deduped_ids.append(token)
    if not deduped_ids:
        return []

    challenged_ts = _normalize_utc_timestamp(challenged_at or _now())
    challenged_ids: list[str] = []
    with get_connection() as conn:
        for start in range(0, len(deduped_ids), 250):
            chunk = deduped_ids[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            active_rows = conn.execute(
                f"""SELECT id
                    FROM claims
                    WHERE id IN ({placeholders})
                      AND status = 'active'
                      AND julianday(valid_from) <= julianday(?)
                      AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))
                    ORDER BY id ASC""",  # nosec B608
                [*chunk, challenged_ts, challenged_ts],
            ).fetchall()
            active_ids = [row["id"] for row in active_rows]
            if not active_ids:
                continue

            active_placeholders = ",".join("?" for _ in active_ids)
            conn.execute(
                f"""UPDATE claims
                    SET status = 'challenged', updated_at = ?
                    WHERE id IN ({active_placeholders})
                      AND status = 'active'""",  # nosec B608
                [challenged_ts, *active_ids],
            )
            conn.executemany(
                """INSERT INTO claim_events
                   (id, claim_id, event_type, details, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        str(uuid.uuid4()),
                        claim_id,
                        "challenged",
                        json.dumps({"challenged_at": challenged_ts}),
                        challenged_ts,
                    )
                    for claim_id in active_ids
                ],
            )
            challenged_ids.extend(active_ids)

    return sorted(challenged_ids)


def mark_claims_challenged_by_anchors(
    anchors: list[dict[str, Any]],
    challenged_at: str | None = None,
) -> list[str]:
    """Mark active claims as challenged when linked episode anchors match."""
    if not anchors:
        return []

    anchor_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor in anchors:
        anchor_type = str(anchor.get("anchor_type") or anchor.get("type") or "").strip()
        anchor_value = str(anchor.get("anchor_value") or anchor.get("value") or "").strip()
        if not anchor_type or not anchor_value:
            continue
        pair = (anchor_type, anchor_value)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        anchor_pairs.append(pair)

    if not anchor_pairs:
        return []

    challenged_ts = _normalize_utc_timestamp(challenged_at or _now())
    anchor_clauses = []
    anchor_params: list[Any] = []
    for anchor_type, anchor_value in anchor_pairs:
        anchor_clauses.append("(ea.anchor_type = ? AND ea.anchor_value = ?)")
        anchor_params.extend([anchor_type, anchor_value])
    select_query = f"""SELECT DISTINCT c.id
            FROM claims c
            JOIN claim_sources cs ON cs.claim_id = c.id
            JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
            WHERE c.status = 'active'
              AND julianday(c.valid_from) <= julianday(?)
              AND (c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))
              AND ({' OR '.join(anchor_clauses)})
            ORDER BY c.id ASC"""  # nosec B608

    with get_connection() as conn:
        rows = conn.execute(
            select_query,
            [challenged_ts, challenged_ts, *anchor_params],
        ).fetchall()

        claim_ids = [row["id"] for row in rows]
        if not claim_ids:
            return []

        placeholders = ",".join("?" for _ in claim_ids)
        update_query = f"""UPDATE claims
            SET status = 'challenged', updated_at = ?
            WHERE id IN ({placeholders})
              AND status = 'active'"""  # nosec B608
        conn.execute(
            update_query,
            [challenged_ts, *claim_ids],
        )
        # Record the transition so temporal as_of queries can reconstruct the prior active state.
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    claim_id,
                    "challenged",
                    json.dumps({"challenged_at": challenged_ts}),
                    challenged_ts,
                )
                for claim_id in claim_ids
            ],
        )

    return claim_ids


# -- Consolidation Run Tracking -----------------------------------------------

_SCHEDULER_ROW_ID = "global"


def _default_stale_consolidation_timeout_seconds(max_duration_seconds: float) -> float:
    """Return stale-run timeout with a grace window above configured max duration."""
    safe_max_duration = max(float(max_duration_seconds), 1.0)
    # Give the worker a 5-minute grace period while still recovering quickly
    # after process crashes. Keep a sane lower bound for very small durations.
    return max(600.0, safe_max_duration + 300.0)


def reconcile_stale_consolidation_state(
    *,
    stale_timeout_seconds: float | None = None,
    as_of: str | datetime | None = None,
) -> dict[str, Any]:
    """Recover stale running consolidation state after interrupted execution.

    Marks long-running orphaned consolidation runs as failed and clears stale
    scheduler leases/status so automatic scheduling can continue.
    """
    cfg = _get_config()
    timeout_seconds = (
        _default_stale_consolidation_timeout_seconds(cfg.CONSOLIDATION_MAX_DURATION)
        if stale_timeout_seconds is None
        else max(float(stale_timeout_seconds), 1.0)
    )

    if isinstance(as_of, str):
        as_of_dt = parse_datetime(as_of)
    elif isinstance(as_of, datetime):
        as_of_dt = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
    else:
        as_of_dt = datetime.now(timezone.utc)
    as_of_dt = as_of_dt.astimezone(timezone.utc)

    as_of_iso = as_of_dt.isoformat()
    cutoff_iso = (as_of_dt - timedelta(seconds=timeout_seconds)).isoformat()
    stale_run_message = (
        "Recovered stale running consolidation run after exceeding timeout "
        f"({int(timeout_seconds)}s)."
    )
    stale_scheduler_message = (
        "Recovered stale running scheduler state after exceeding timeout "
        f"({int(timeout_seconds)}s)."
    )

    stale_run_ids: list[str] = []
    scheduler_recovered = False

    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)

        stale_rows = conn.execute(
            """SELECT id
               FROM consolidation_runs
               WHERE status = ?
                 AND completed_at IS NULL
                 AND julianday(started_at) <= julianday(?)""",
            (RUN_STATUS_RUNNING, cutoff_iso),
        ).fetchall()
        stale_run_ids = [str(row["id"]) for row in stale_rows]

        if stale_run_ids:
            conn.executemany(
                """UPDATE consolidation_runs
                    SET status = ?,
                        completed_at = ?,
                        error_message = COALESCE(error_message, ?)
                    WHERE id = ?""",
                [
                    (RUN_STATUS_FAILED, as_of_iso, stale_run_message, stale_run_id)
                    for stale_run_id in stale_run_ids
                ],
            )

        scheduler_cursor = conn.execute(
            """UPDATE consolidation_scheduler
               SET last_run_completed_at = COALESCE(last_run_completed_at, ?),
                   last_status = ?,
                   last_error = COALESCE(last_error, ?),
                   next_due_at = ?,
                   lease_owner = NULL,
                   lease_expires_at = NULL,
                   updated_at = ?
               WHERE id = ?
                 AND last_status = ?
                 AND last_run_started_at IS NOT NULL
                 AND julianday(last_run_started_at) <= julianday(?)
                 AND (
                   lease_owner IS NULL
                   OR lease_expires_at IS NULL
                   OR julianday(lease_expires_at) <= julianday(?)
                 )""",
            (
                as_of_iso,
                RUN_STATUS_FAILED,
                stale_scheduler_message,
                as_of_iso,
                as_of_iso,
                _SCHEDULER_ROW_ID,
                RUN_STATUS_RUNNING,
                cutoff_iso,
                as_of_iso,
            ),
        )
        scheduler_recovered = bool(scheduler_cursor.rowcount and scheduler_cursor.rowcount > 0)

    if stale_run_ids or scheduler_recovered:
        logger.warning(
            "Recovered stale consolidation state (stale_runs=%d, scheduler_recovered=%s, timeout_seconds=%.0f)",
            len(stale_run_ids),
            scheduler_recovered,
            timeout_seconds,
        )

    return {
        "stale_timeout_seconds": timeout_seconds,
        "stale_runs_marked_failed": len(stale_run_ids),
        "stale_run_ids": stale_run_ids,
        "scheduler_state_recovered": scheduler_recovered,
        "cutoff": cutoff_iso,
        "as_of": as_of_iso,
    }


def _ensure_consolidation_scheduler_row(conn: sqlite3.Connection) -> sqlite3.Row:
    """Ensure the singleton scheduler row exists and return it."""
    row = conn.execute(
        "SELECT * FROM consolidation_scheduler WHERE id = ?",
        (_SCHEDULER_ROW_ID,),
    ).fetchone()
    if row is not None:
        return cast(sqlite3.Row, row)

    now = _now()
    conn.execute(
        """INSERT INTO consolidation_scheduler
           (id, last_status, next_due_at, updated_at)
           VALUES (?, 'idle', ?, ?)""",
        (_SCHEDULER_ROW_ID, now, now),
    )
    created = conn.execute(
        "SELECT * FROM consolidation_scheduler WHERE id = ?",
        (_SCHEDULER_ROW_ID,),
    ).fetchone()
    if created is None:
        raise RuntimeError("Failed to initialize consolidation scheduler state")
    return cast(sqlite3.Row, created)


def get_consolidation_scheduler_state() -> dict[str, Any]:
    """Return persisted scheduler state for automatic consolidation."""
    reconcile_stale_consolidation_state()
    with get_connection() as conn:
        row = _ensure_consolidation_scheduler_row(conn)
    return dict(row)


def try_acquire_consolidation_lease(owner: str, lease_seconds: float) -> bool:
    """Try to acquire the scheduler lease for a consolidation worker.

    Returns True when the lease is acquired, False when held by another owner.
    """
    owner_token = owner.strip()
    if not owner_token:
        raise ValueError("owner must be non-empty")

    lease_ttl = max(float(lease_seconds), 1.0)
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    lease_expires_at = (now_dt + timedelta(seconds=lease_ttl)).isoformat()

    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        cursor = conn.execute(
            """UPDATE consolidation_scheduler
               SET lease_owner = ?, lease_expires_at = ?, updated_at = ?
               WHERE id = ?
                 AND (
                   lease_owner IS NULL
                   OR lease_expires_at IS NULL
                   OR julianday(lease_expires_at) <= julianday(?)
                   OR lease_owner = ?
                 )""",
            (
                owner_token,
                lease_expires_at,
                now,
                _SCHEDULER_ROW_ID,
                now,
                owner_token,
            ),
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def mark_consolidation_scheduler_started(
    owner: str,
    *,
    trigger_reason: str,
    utility_score: float | None = None,
    started_at: str | None = None,
) -> None:
    """Persist scheduler state when a consolidation run starts."""
    owner_token = owner.strip()
    if not owner_token:
        raise ValueError("owner must be non-empty")

    ts = started_at or _now()
    score_value = float(utility_score) if utility_score is not None else None
    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        conn.execute(
            """UPDATE consolidation_scheduler
               SET last_run_started_at = ?,
                   last_status = ?,
                   last_error = NULL,
                   last_trigger = ?,
                   last_utility_score = ?,
                   updated_at = ?
               WHERE id = ?
                 AND (lease_owner = ? OR lease_owner IS NULL)""",
            (
                ts,
                RUN_STATUS_RUNNING,
                trigger_reason,
                score_value,
                ts,
                _SCHEDULER_ROW_ID,
                owner_token,
            ),
        )


def mark_consolidation_scheduler_finished(
    owner: str,
    *,
    status: RunStatus,
    interval_hours: float,
    error_message: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Persist scheduler state and release the lease after a run completes."""
    owner_token = owner.strip()
    if not owner_token:
        raise ValueError("owner must be non-empty")

    completed_ts = completed_at or _now()
    completed_dt = parse_datetime(completed_ts)
    next_due = (completed_dt + timedelta(hours=max(float(interval_hours), 0.01))).isoformat()

    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        conn.execute(
            """UPDATE consolidation_scheduler
               SET last_run_completed_at = ?,
                   last_status = ?,
                   last_error = ?,
                   next_due_at = ?,
                   lease_owner = NULL,
                   lease_expires_at = NULL,
                   updated_at = ?
               WHERE id = ?
                 AND (lease_owner = ? OR lease_owner IS NULL)""",
            (
                completed_ts,
                status,
                error_message,
                next_due,
                completed_ts,
                _SCHEDULER_ROW_ID,
                owner_token,
            ),
        )


def release_consolidation_lease(owner: str) -> None:
    """Release the scheduler lease without mutating run outcome fields."""
    owner_token = owner.strip()
    if not owner_token:
        return
    now = _now()
    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        conn.execute(
            """UPDATE consolidation_scheduler
               SET lease_owner = NULL,
                   lease_expires_at = NULL,
                   updated_at = ?
               WHERE id = ?
                 AND lease_owner = ?""",
            (now, _SCHEDULER_ROW_ID, owner_token),
        )

def start_consolidation_run() -> str:
    run_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO consolidation_runs (id, started_at) VALUES (?, ?)",
            (run_id, _now()),
        )
    return run_id


def complete_consolidation_run(
    run_id: str,
    episodes_processed: int = 0,
    clusters_formed: int = 0,
    topics_created: int = 0,
    topics_updated: int = 0,
    episodes_pruned: int = 0,
    status: RunStatus = RUN_STATUS_COMPLETED,
    error_message: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE consolidation_runs
               SET completed_at = ?, episodes_processed = ?,
                   clusters_formed = ?, topics_created = ?,
                   topics_updated = ?, episodes_pruned = ?,
                   status = ?, error_message = ?
               WHERE id = ?""",
            (_now(), episodes_processed, clusters_formed, topics_created,
             topics_updated, episodes_pruned, status, error_message, run_id),
        )


def get_last_consolidation_run() -> dict[str, Any] | None:
    reconcile_stale_consolidation_state()
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM consolidation_runs
               ORDER BY started_at DESC LIMIT 1"""
        ).fetchone()
    return dict(row) if row else None


def get_recent_consolidation_runs(limit: int = 5) -> list[dict[str, Any]]:
    """Return recent consolidation runs as activity summaries."""
    reconcile_stale_consolidation_state()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, started_at, completed_at, status,
                      episodes_processed, clusters_formed,
                      topics_created, topics_updated, episodes_pruned,
                      error_message
               FROM consolidation_runs
               ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Export / Bulk queries ────────────────────────────────────────────────────

def get_all_episodes(
    include_deleted: bool = False,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return all episodes for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if not include_deleted:
        conditions.append("deleted = 0")
    _apply_scope_filters(conditions, params, scope)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM episodes {where_clause} ORDER BY created_at",  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claims(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claims for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"id IN ({placeholders})")  # nosec B608
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_type, canonical_text, payload, status, confidence,
                      valid_from, valid_until, created_at, updated_at
               FROM claims
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claim_edges(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claim edges for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"from_claim_id IN ({placeholders})")  # nosec B608
        params.extend(claim_ids)
        conditions.append(f"to_claim_id IN ({placeholders})")  # nosec B608
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, from_claim_id, to_claim_id, edge_type, confidence, details, created_at
               FROM claim_edges
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claim_sources(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claim source links for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"claim_id IN ({placeholders})")  # nosec B608
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_id, source_episode_id, source_topic_id, source_record_id, created_at
               FROM claim_sources
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claim_events(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claim lifecycle events for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"claim_id IN ({placeholders})")  # nosec B608
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_id, event_type, details, created_at
               FROM claim_events
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_episode_anchors(
    episode_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all episode anchors for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if episode_ids is not None:
        if not episode_ids:
            return []
        placeholders = ",".join("?" for _ in episode_ids)
        conditions.append(f"episode_id IN ({placeholders})")  # nosec B608
        params.extend(episode_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, episode_id, anchor_type, anchor_value, created_at
               FROM episode_anchors
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),  # nosec B608
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def import_claim_graph_snapshot(
    *,
    claims: Sequence[Mapping[str, Any]] = (),
    claim_edges: Sequence[Mapping[str, Any]] = (),
    claim_sources: Sequence[Mapping[str, Any]] = (),
    claim_events: Sequence[Mapping[str, Any]] = (),
    episode_anchors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, int]:
    """Import claim graph entities from an export snapshot.

    Existing rows are preserved or updated by primary key:
    - claims: upsert by claim ID
    - edges/sources/events/anchors: insert with conflict-ignore by unique key/ID
    """
    now = _now()
    imported = {
        "claims": 0,
        "claim_edges": 0,
        "claim_sources": 0,
        "claim_events": 0,
        "episode_anchors": 0,
    }

    with get_connection() as conn:
        for claim in claims:
            claim_id = str(claim.get("id") or "").strip()
            if not claim_id:
                continue
            payload_raw = claim.get("payload")
            payload_text = payload_raw if isinstance(payload_raw, str) else json.dumps(payload_raw or {})
            claim_confidence_raw = claim.get("confidence", 0.8)
            claim_confidence = 0.8 if claim_confidence_raw is None else float(claim_confidence_raw)

            created_at = str(claim.get("created_at") or now)
            updated_at = str(claim.get("updated_at") or created_at)
            valid_from = str(claim.get("valid_from") or created_at)
            valid_until_raw = claim.get("valid_until")
            valid_until = str(valid_until_raw) if valid_until_raw is not None else None

            conn.execute(
                """INSERT INTO claims
                   (id, claim_type, canonical_text, payload, status, confidence,
                    valid_from, valid_until, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       claim_type = excluded.claim_type,
                       canonical_text = excluded.canonical_text,
                       payload = excluded.payload,
                       status = excluded.status,
                       confidence = excluded.confidence,
                       valid_from = excluded.valid_from,
                       valid_until = excluded.valid_until,
                       updated_at = excluded.updated_at""",
                (
                    claim_id,
                    str(claim.get("claim_type") or "fact"),
                    str(claim.get("canonical_text") or ""),
                    payload_text,
                    str(claim.get("status") or "active"),
                    claim_confidence,
                    valid_from,
                    valid_until,
                    created_at,
                    updated_at,
                ),
            )
            imported["claims"] += 1

        for edge in claim_edges:
            edge_id = str(edge.get("id") or uuid.uuid4())
            from_claim_id = str(edge.get("from_claim_id") or "").strip()
            to_claim_id = str(edge.get("to_claim_id") or "").strip()
            edge_type = str(edge.get("edge_type") or "").strip()
            if not from_claim_id or not to_claim_id or not edge_type:
                continue

            details_raw = edge.get("details")
            details_text = details_raw if isinstance(details_raw, str) else (
                json.dumps(details_raw) if details_raw is not None else None
            )
            created_at = str(edge.get("created_at") or now)
            edge_confidence_raw = edge.get("confidence", 1.0)
            edge_confidence = 1.0 if edge_confidence_raw is None else float(edge_confidence_raw)

            cursor = conn.execute(
                """INSERT INTO claim_edges
                   (id, from_claim_id, to_claim_id, edge_type, confidence, details, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (
                    edge_id,
                    from_claim_id,
                    to_claim_id,
                    edge_type,
                    edge_confidence,
                    details_text,
                    created_at,
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["claim_edges"] += 1

        for source in claim_sources:
            source_id = str(source.get("id") or uuid.uuid4())
            claim_id = str(source.get("claim_id") or "").strip()
            if not claim_id:
                continue

            source_episode_id = source.get("source_episode_id")
            source_topic_id = source.get("source_topic_id")
            source_record_id = source.get("source_record_id")
            created_at = str(source.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO claim_sources
                   (id, claim_id, source_episode_id, source_topic_id, source_record_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (
                    source_id,
                    claim_id,
                    str(source_episode_id) if source_episode_id is not None else None,
                    str(source_topic_id) if source_topic_id is not None else None,
                    str(source_record_id) if source_record_id is not None else None,
                    created_at,
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["claim_sources"] += 1

        for event in claim_events:
            event_id = str(event.get("id") or uuid.uuid4())
            claim_id = str(event.get("claim_id") or "").strip()
            event_type = str(event.get("event_type") or "").strip()
            if not claim_id or not event_type:
                continue

            details_raw = event.get("details")
            details_text = details_raw if isinstance(details_raw, str) else (
                json.dumps(details_raw) if details_raw is not None else None
            )
            created_at = str(event.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO claim_events
                   (id, claim_id, event_type, details, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (event_id, claim_id, event_type, details_text, created_at),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["claim_events"] += 1

        for anchor in episode_anchors:
            anchor_id = str(anchor.get("id") or uuid.uuid4())
            episode_id = str(anchor.get("episode_id") or "").strip()
            anchor_type = str(anchor.get("anchor_type") or "").strip()
            anchor_value = str(anchor.get("anchor_value") or "").strip()
            if not episode_id or not anchor_type or not anchor_value:
                continue
            created_at = str(anchor.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO episode_anchors
                   (id, episode_id, anchor_type, anchor_value, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (anchor_id, episode_id, anchor_type, anchor_value, created_at),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["episode_anchors"] += 1

    return imported


def get_all_active_episodes() -> list[dict[str, Any]]:
    """Return all non-deleted, non-pruned episodes for surprise adjustment."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, surprise_score, access_count, created_at, updated_at, consolidated
               FROM episodes WHERE deleted = 0 AND consolidated != 2"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_surprise_scores(updates: list[tuple[float, str]]) -> None:
    """Batch update surprise scores. updates = [(new_score, episode_id), ...]"""
    if not updates:
        return
    now = _now()
    with get_connection() as conn:
        conn.executemany(
            "UPDATE episodes SET surprise_score = ?, updated_at = ? WHERE id = ?",
            [(score, now, eid) for score, eid in updates],
        )


# ── Stats ────────────────────────────────────────────────────────────────────

def get_stats(scope: Mapping[str, Any] | None = None) -> StatsDict:
    with get_connection() as conn:
        episode_conditions: list[str] = []
        episode_params: list[Any] = []
        _apply_scope_filters(episode_conditions, episode_params, scope)
        episode_where = f"WHERE {' AND '.join(episode_conditions)}" if episode_conditions else ""
        ep_counts = conn.execute(
            f"""SELECT
                 COUNT(*) FILTER (WHERE deleted = 0) as total,
                 COUNT(*) FILTER (WHERE consolidated = 0 AND deleted = 0) as pending,
                 COUNT(*) FILTER (WHERE consolidated = 1 AND deleted = 0) as consolidated,
                 COUNT(*) FILTER (WHERE consolidated = 2 OR deleted = 1) as pruned
               FROM episodes {episode_where}""",  # nosec B608
            episode_params,
        ).fetchone()
        topic_conditions: list[str] = []
        topic_params: list[Any] = []
        _apply_scope_filters(topic_conditions, topic_params, scope)
        topic_where = f"WHERE {' AND '.join(topic_conditions)}" if topic_conditions else ""
        kt_counts = conn.execute(
            f"""SELECT COUNT(*) as total_topics,
                      COALESCE(SUM(fact_count), 0) as total_facts
               FROM knowledge_topics {topic_where}""",  # nosec B608
            topic_params,
        ).fetchone()
        record_conditions: list[str] = []
        record_params: list[Any] = []
        _apply_scope_filters(record_conditions, record_params, scope)
        record_where = f"WHERE {' AND '.join(record_conditions)}" if record_conditions else ""
        rec_counts = conn.execute(
            f"""SELECT
                 COUNT(*) FILTER (WHERE deleted = 0) as total_records,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'fact') as facts,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'solution') as solutions,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'preference') as preferences,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'procedure') as procedures
               FROM knowledge_records {record_where}""",  # nosec B608
            record_params,
        ).fetchone()

    return {
        "episodic_buffer": {
            "total": ep_counts["total"],
            "pending_consolidation": ep_counts["pending"],
            "consolidated": ep_counts["consolidated"],
            "pruned": ep_counts["pruned"],
        },
        "knowledge_base": {
            "total_topics": kt_counts["total_topics"],
            "total_facts": kt_counts["total_facts"],
            "total_records": rec_counts["total_records"],
            "records_by_type": {
                "facts": rec_counts["facts"],
                "solutions": rec_counts["solutions"],
                "preferences": rec_counts["preferences"],
                "procedures": rec_counts["procedures"],
            },
        },
    }


# ── Consolidation Attempt Tracking ────────────────────────────────────────────

def increment_consolidation_attempts(episode_ids: list[str]) -> None:
    """Record a failed consolidation attempt for the given episodes."""
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        conn.execute(
            f"UPDATE episodes SET consolidation_attempts = consolidation_attempts + 1, "  # nosec B608
            f"last_consolidation_attempt = ? WHERE id IN ({placeholders})",
            [now] + episode_ids,
        )


def reset_stale_consolidation_attempts(max_attempts: int = 5, stale_hours: int = 24) -> int:
    """Reset consolidation_attempts for episodes stuck at max that haven't been
    retried recently. This allows episodes to be reconsolidated after the LLM
    backend recovers from an outage.

    Returns:
        Number of episodes reset.
    """
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=stale_hours)).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """UPDATE episodes SET consolidation_attempts = 0, last_consolidation_attempt = NULL
               WHERE consolidation_attempts >= ? AND deleted = 0 AND consolidated = 0
               AND (last_consolidation_attempt IS NULL OR last_consolidation_attempt < ?)""",
            (max_attempts, cutoff),
        )
    return int(cursor.rowcount)


def get_median_access_count() -> float:
    """Compute the median access_count across all active episodes using SQL."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT access_count FROM episodes
               WHERE deleted = 0 AND consolidated != 2
               ORDER BY access_count
               LIMIT 1 OFFSET (
                   SELECT COUNT(*) / 2 FROM episodes
                   WHERE deleted = 0 AND consolidated != 2
               )"""
        ).fetchone()
    return float(row["access_count"]) if row else 0.0


def get_active_episodes_paginated(offset: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    """Return a page of non-deleted, non-pruned episodes for surprise adjustment."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, surprise_score, access_count, created_at, updated_at, consolidated
               FROM episodes WHERE deleted = 0 AND consolidated != 2
               ORDER BY id LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def insert_consolidation_metrics(
    run_id: str,
    clusters_succeeded: int,
    clusters_failed: int,
    avg_confidence: float,
    episodes_processed: int,
    duration_seconds: float,
    api_calls: int,
    topics_created: int,
    topics_updated: int,
    episodes_pruned: int,
) -> str:
    """Insert a consolidation run metrics record."""
    metric_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO consolidation_metrics "
            "(id, run_id, timestamp, clusters_succeeded, clusters_failed, "
            "avg_confidence, episodes_processed, duration_seconds, api_calls, "
            "topics_created, topics_updated, episodes_pruned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (metric_id, run_id, _now(), clusters_succeeded, clusters_failed,
             avg_confidence, episodes_processed, duration_seconds, api_calls,
             topics_created, topics_updated, episodes_pruned),
        )
    return metric_id


def search_episodes(
    query: str | None = None,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    scope: Mapping[str, Any] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Keyword/metadata search over episodes. No embeddings required.

    Args:
        query: Text substring to search in episode content (case-insensitive).
        content_types: Filter to specific content types.
        tags: Filter to episodes with at least one matching tag.
        after: Only episodes created after this ISO date.
        before: Only episodes created before this ISO date.
        limit: Max results.

    Returns:
        List of episode dicts, ordered by created_at descending.
    """
    conditions: list[str] = ["deleted = 0"]
    params: list[Any] = []
    _apply_scope_filters(conditions, params, scope)

    if query:
        # Escape LIKE wildcards in user input to prevent unintended pattern matching
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions.append("content LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    if content_types:
        placeholders = ",".join("?" for _ in content_types)
        conditions.append(f"content_type IN ({placeholders})")
        params.extend(content_types)

    if after:
        conditions.append("created_at > ?")
        params.append(after)

    if before:
        conditions.append("created_at < ?")
        params.append(before)

    if limit <= 0:
        return []

    where = " AND ".join(conditions)
    base_sql = f"SELECT * FROM episodes WHERE {where} ORDER BY created_at DESC"  # nosec B608

    if not tags:
        with get_connection() as conn:
            rows = conn.execute(
                f"{base_sql} LIMIT ?",  # nosec B608
                [*params, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    requested_tags = set(tags)
    results: list[dict[str, Any]] = []
    offset = 0
    page_size = min(max(limit * 5, 50), 500)
    paged_sql = f"{base_sql} LIMIT ? OFFSET ?"  # nosec B608

    with get_connection() as conn:
        while len(results) < limit:
            rows = conn.execute(
                paged_sql,
                [*params, page_size, offset],
            ).fetchall()
            if not rows:
                break

            offset += len(rows)
            for row in rows:
                ep = dict(row)
                ep_tags = parse_json_list(ep["tags"])
                if not requested_tags.intersection(ep_tags):
                    continue
                results.append(ep)
                if len(results) >= limit:
                    break

            if len(rows) < page_size:
                break

    return results


def get_consolidation_metrics(limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent consolidation metrics, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM consolidation_metrics ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
