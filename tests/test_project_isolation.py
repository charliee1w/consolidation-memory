"""Tests for multi-project namespace support: validation, project switching, and isolation."""

import sqlite3
import threading
from contextlib import closing
from unittest.mock import patch

import pytest

from consolidation_memory.config import (
    validate_project_name,
    get_active_project,
    set_active_project,
    get_config,
    maybe_migrate_to_projects,
)
from consolidation_memory import config
import consolidation_memory.database as database
from tests.helpers import make_normalized_vec as _make_normalized_vec


# ── Project name validation ──────────────────────────────────────────────────


class TestProjectNameValidation:
    """Tests for validate_project_name()."""

    @pytest.mark.parametrize(
        "name",
        [
            "default",
            "my-project",
            "project_1",
            "a",
            "0test",
            "abc-def_ghi",
            "a" * 64,  # max length
        ],
    )
    def test_valid_names(self, name):
        assert validate_project_name(name) == name

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_project_name("")

    @pytest.mark.parametrize(
        "name",
        [
            "UPPER",
            "Mixed",
            "has space",
            "special!char",
            "path/../traversal",
            "a/b",
            "a\\b",
            ".hidden",
        ],
    )
    def test_invalid_special_chars(self, name):
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name(name)

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("a" * 65)

    def test_starts_with_hyphen_raises(self):
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("-bad-start")

    def test_starts_with_underscore_raises(self):
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("_bad-start")

    def test_path_traversal_raises(self):
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("..")


# ── set_active_project ───────────────────────────────────────────────────────


class TestSetActiveProject:
    """Tests for set_active_project() and get_active_project()."""

    def test_default_project(self, tmp_data_dir):
        """The fixture sets up 'default' as the active project."""
        assert get_active_project() == "default"

    def test_set_custom_project(self, tmp_data_dir):
        """Switching to a custom project updates active_project and paths."""
        cfg = get_config()
        base = cfg._base_data_dir
        result = set_active_project("my-project")
        assert result == "my-project"
        assert get_active_project() == "my-project"
        assert cfg.DATA_DIR == base / "projects" / "my-project"

    def test_all_paths_recalculated(self, tmp_data_dir):
        """All path fields should point into the new project directory."""
        cfg = get_config()
        base = cfg._base_data_dir
        set_active_project("alpha")
        expected_data = base / "projects" / "alpha"

        assert cfg.DATA_DIR == expected_data
        assert cfg.DB_PATH == expected_data / "memory.db"
        assert cfg.FAISS_INDEX_PATH == expected_data / "faiss_index.bin"
        assert cfg.FAISS_ID_MAP_PATH == expected_data / "faiss_id_map.json"
        assert cfg.FAISS_TOMBSTONE_PATH == expected_data / "faiss_tombstones.json"
        assert cfg.FAISS_RELOAD_SIGNAL == expected_data / ".faiss_reload"
        assert cfg.FAISS_WRITE_LOCK_PATH == expected_data / ".faiss_write.lock"
        assert cfg.KNOWLEDGE_DIR == expected_data / "knowledge"
        assert cfg.KNOWLEDGE_VERSIONS_DIR == expected_data / "knowledge" / "versions"
        assert cfg.CONSOLIDATION_LOG_DIR == expected_data / "consolidation_logs"
        assert cfg.BACKUP_DIR == expected_data / "backups"

    def test_log_dir_shared_across_projects(self, tmp_data_dir):
        """LOG_DIR is shared and must NOT change when project changes."""
        cfg = get_config()
        log_before = cfg.LOG_DIR
        set_active_project("other-project")
        assert cfg.LOG_DIR == log_before

    def test_none_resolves_to_env_default(self, tmp_data_dir):
        """None without env var falls back to 'default'."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove the env var if set
            import os
            os.environ.pop("CONSOLIDATION_MEMORY_PROJECT", None)
            result = set_active_project(None)
        assert result == "default"
        assert get_active_project() == "default"

    def test_none_resolves_from_env_var(self, tmp_data_dir):
        """None with env var reads from CONSOLIDATION_MEMORY_PROJECT."""
        with patch.dict("os.environ", {"CONSOLIDATION_MEMORY_PROJECT": "from-env"}):
            result = set_active_project(None)
        assert result == "from-env"
        assert get_active_project() == "from-env"

    def test_invalid_name_raises(self, tmp_data_dir):
        """Invalid project names must raise ValueError."""
        with pytest.raises(ValueError):
            set_active_project("INVALID")
        with pytest.raises(ValueError):
            set_active_project("")
        with pytest.raises(ValueError):
            set_active_project("../escape")

    def test_switch_closes_thread_local_db_connection(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema, get_connection, insert_episode

        cfg = get_config()
        ensure_schema()
        insert_episode("alpha-content")
        with get_connection() as conn_alpha:
            alpha_conn_id = id(conn_alpha)

        set_active_project("beta")
        ensure_schema()
        insert_episode("beta-content")
        with get_connection() as conn_beta:
            beta_conn_id = id(conn_beta)

        assert beta_conn_id != alpha_conn_id

        database.close_all_connections()

        alpha_db = cfg._base_data_dir / "projects" / "default" / "memory.db"
        beta_db = cfg._base_data_dir / "projects" / "beta" / "memory.db"

        with closing(sqlite3.connect(alpha_db)) as conn:
            alpha_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        with closing(sqlite3.connect(beta_db)) as conn:
            beta_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

        assert alpha_count == 1
        assert beta_count == 1

    def test_close_all_connections_closes_worker_thread_connection(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema, get_connection

        ensure_schema()
        ready = threading.Event()
        release = threading.Event()
        holder: dict[str, sqlite3.Connection] = {}

        def worker() -> None:
            with get_connection() as conn:
                conn.execute("SELECT 1")
                holder["conn"] = conn
            ready.set()
            release.wait(timeout=5)

        thread = threading.Thread(target=worker)
        thread.start()
        assert ready.wait(timeout=5)
        assert "conn" in holder

        database.close_all_connections()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            holder["conn"].execute("SELECT 1")

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


# ── Project isolation (end-to-end) ──────────────────────────────────────────


class TestProjectIsolation:
    """Store in project A should not appear in project B recalls."""

    @patch("consolidation_memory.backends.encode_documents")
    @patch("consolidation_memory.backends.encode_query")
    def test_store_in_a_not_visible_in_b(self, mock_query, mock_embed, tmp_data_dir):
        from consolidation_memory.client import MemoryClient

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)
        mock_query.return_value = vec

        # Store in project "alpha"
        database.close_all_connections()
        config.set_active_project("alpha")
        with MemoryClient(auto_consolidate=False) as client_a:
            client_a.store("secret alpha fact", content_type="fact", tags=["alpha"])

        # Recall in project "beta" — should find nothing
        database.close_all_connections()
        config.set_active_project("beta")
        with MemoryClient(auto_consolidate=False) as client_b:
            result = client_b.recall("secret alpha fact", n_results=10)
            assert len(result.episodes) == 0

    @patch("consolidation_memory.backends.encode_documents")
    @patch("consolidation_memory.backends.encode_query")
    def test_store_in_a_visible_in_a(self, mock_query, mock_embed, tmp_data_dir):
        from consolidation_memory.client import MemoryClient

        vec = _make_normalized_vec(seed=99)
        mock_embed.return_value = vec.reshape(1, -1)
        mock_query.return_value = vec

        database.close_all_connections()
        config.set_active_project("alpha")
        with MemoryClient(auto_consolidate=False) as client:
            client.store("visible alpha fact", content_type="fact", tags=["alpha"])

        # Re-open same project — should find it
        database.close_all_connections()
        config.set_active_project("alpha")
        with MemoryClient(auto_consolidate=False) as client:
            result = client.recall("visible alpha fact", n_results=10)
            assert len(result.episodes) > 0
            assert "visible alpha fact" in result.episodes[0]["content"]

    @patch("consolidation_memory.backends.encode_documents")
    def test_default_project_works(self, mock_embed, tmp_data_dir):
        from consolidation_memory.client import MemoryClient

        vec = _make_normalized_vec(seed=7)
        mock_embed.return_value = vec.reshape(1, -1)

        database.close_all_connections()
        config.set_active_project("default")
        with MemoryClient(auto_consolidate=False) as client:
            store_result = client.store("default project fact", content_type="fact")
            assert store_result.status == "stored"


# ── Migration from flat layout ──────────────────────────────────────────────


class TestMigration:
    """Test auto-migration from flat DATA_DIR to projects/default/."""

    def test_migrate_flat_to_projects(self, tmp_path):
        base = tmp_path / "migration_test"
        base.mkdir()
        # Create flat layout
        (base / "memory.db").write_text("fake db")
        (base / "faiss_index.bin").write_bytes(b"fake index")
        (base / "faiss_id_map.json").write_text("[]")
        (base / "faiss_tombstones.json").write_text("[]")
        (base / "knowledge").mkdir()
        (base / "knowledge" / "topic.md").write_text("knowledge")
        (base / "backups").mkdir()
        (base / "consolidation_logs").mkdir()

        migrated = maybe_migrate_to_projects(base)
        assert migrated is True

        default_dir = base / "projects" / "default"
        assert (default_dir / "memory.db").exists()
        assert (default_dir / "faiss_index.bin").exists()
        assert (default_dir / "knowledge" / "topic.md").exists()
        assert not (base / "memory.db").exists()  # moved, not copied

    def test_no_migration_if_already_projects(self, tmp_path):
        base = tmp_path / "migration_test"
        base.mkdir()
        (base / "projects").mkdir()
        (base / "memory.db").write_text("fake db")

        migrated = maybe_migrate_to_projects(base)
        assert migrated is False

    def test_no_migration_if_no_flat_data(self, tmp_path):
        base = tmp_path / "migration_test"
        base.mkdir()

        migrated = maybe_migrate_to_projects(base)
        assert migrated is False
