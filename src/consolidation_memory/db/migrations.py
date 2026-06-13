"""Schema versioning, migrations, and ensure_schema entry point."""

from __future__ import annotations

import logging
import sqlite3

from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import (
    _DEFAULT_APP_CLIENT_NAME,
    _DEFAULT_APP_CLIENT_TYPE,
    _DEFAULT_NAMESPACE_SHARING_MODE,
    _DEFAULT_NAMESPACE_SLUG,
    _default_project_slug,
)

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 20

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
    17: [
        """CREATE TABLE IF NOT EXISTS action_outcomes (
            id                      TEXT PRIMARY KEY,
            action_key              TEXT NOT NULL,
            action_summary          TEXT NOT NULL,
            outcome_type            TEXT NOT NULL,
            summary                 TEXT,
            details                 TEXT,
            confidence              REAL NOT NULL DEFAULT 0.8,
            provenance              TEXT NOT NULL DEFAULT '{}',
            observed_at             TEXT NOT NULL,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL,
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
        )""",
        "CREATE INDEX IF NOT EXISTS idx_action_outcomes_action_key ON action_outcomes(action_key)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcomes_type_observed ON action_outcomes(outcome_type, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcomes_scope_ns_project ON action_outcomes(namespace_slug, project_slug)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcomes_scope_app ON action_outcomes(namespace_slug, project_slug, app_client_name, app_client_type)",
        """CREATE TABLE IF NOT EXISTS action_outcome_sources (
            id                  TEXT PRIMARY KEY,
            outcome_id          TEXT NOT NULL,
            source_claim_id     TEXT,
            source_record_id    TEXT,
            source_episode_id   TEXT,
            created_at          TEXT NOT NULL,
            FOREIGN KEY (outcome_id) REFERENCES action_outcomes(id) ON DELETE CASCADE,
            FOREIGN KEY (source_claim_id) REFERENCES claims(id),
            FOREIGN KEY (source_record_id) REFERENCES knowledge_records(id),
            FOREIGN KEY (source_episode_id) REFERENCES episodes(id),
            UNIQUE(outcome_id, source_claim_id, source_record_id, source_episode_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_action_outcome_sources_outcome ON action_outcome_sources(outcome_id)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcome_sources_claim ON action_outcome_sources(source_claim_id)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcome_sources_record ON action_outcome_sources(source_record_id)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcome_sources_episode ON action_outcome_sources(source_episode_id)",
        """CREATE TABLE IF NOT EXISTS action_outcome_refs (
            id              TEXT PRIMARY KEY,
            outcome_id      TEXT NOT NULL,
            ref_type        TEXT NOT NULL,
            ref_key         TEXT NOT NULL,
            ref_value       TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (outcome_id) REFERENCES action_outcomes(id) ON DELETE CASCADE,
            UNIQUE(outcome_id, ref_type, ref_key, ref_value)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_action_outcome_refs_outcome ON action_outcome_refs(outcome_id)",
        "CREATE INDEX IF NOT EXISTS idx_action_outcome_refs_lookup ON action_outcome_refs(ref_type, ref_value)",
    ],
    # Migration 18 is applied specially in _apply_migration() so fast-path
    # metric columns are added idempotently on replay.
    18: [],
    # Migration 19 is applied specially in _apply_migration() so the precision
    # column is added idempotently when replaying from an older version marker.
    19: [],
    # Migration 20 is applied specially in _apply_migration() so the scheduler
    # trigger breakdown column is added idempotently on replay.
    20: [],
}

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
    if version == 18:
        _apply_consolidation_metrics_fast_path_migration(conn)
        return
    if version == 19:
        _apply_claim_precision_migration(conn)
        return
    if version == 20:
        _apply_scheduler_trigger_breakdown_migration(conn)
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
    _apply_consolidation_metrics_fast_path_migration(conn)
    _apply_claim_precision_migration(conn)
    _apply_scheduler_trigger_breakdown_migration(conn)


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
                    )""",
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


def _apply_consolidation_metrics_fast_path_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v18: fast-path vs LLM fallback counters on consolidation metrics."""
    _add_column_if_missing(
        conn,
        table_name="consolidation_metrics",
        column_name="fast_path_hits",
        column_sql="fast_path_hits INTEGER NOT NULL DEFAULT 0",
    )
    _add_column_if_missing(
        conn,
        table_name="consolidation_metrics",
        column_name="llm_fallbacks",
        column_sql="llm_fallbacks INTEGER NOT NULL DEFAULT 0",
    )


def _apply_claim_precision_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v19: persisted trust precision on claims."""
    _add_column_if_missing(
        conn,
        table_name="claims",
        column_name="precision",
        column_sql="precision REAL NOT NULL DEFAULT 1.0",
    )


def _apply_scheduler_trigger_breakdown_migration(conn: sqlite3.Connection) -> None:
    """Apply schema v20: persisted utility breakdown for last consolidation trigger."""
    _add_column_if_missing(
        conn,
        table_name="consolidation_scheduler",
        column_name="last_trigger_breakdown",
        column_sql="last_trigger_breakdown TEXT",
    )
