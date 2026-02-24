"""Shared fixtures for all test modules."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path):
    """Override config paths to use temp directory for ALL tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "knowledge").mkdir()
    (data_dir / "knowledge" / "versions").mkdir()
    (data_dir / "consolidation_logs").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "backups").mkdir()

    db_path = data_dir / "memory.db"
    faiss_idx = data_dir / "faiss_index.bin"
    faiss_map = data_dir / "faiss_id_map.json"
    faiss_tombstone = data_dir / "faiss_tombstones.json"
    faiss_reload = data_dir / ".faiss_reload"
    knowledge = data_dir / "knowledge"
    knowledge_versions = knowledge / "versions"
    consol_log = data_dir / "consolidation_logs"
    log_dir = tmp_path / "logs"
    backup_dir = tmp_path / "backups"

    # Tests use 384-dim vectors regardless of user config
    test_dim = 384

    patches = [
        patch("consolidation_memory.config.DATA_DIR", data_dir),
        patch("consolidation_memory.config.DB_PATH", db_path),
        patch("consolidation_memory.config.FAISS_INDEX_PATH", faiss_idx),
        patch("consolidation_memory.config.FAISS_ID_MAP_PATH", faiss_map),
        patch("consolidation_memory.config.FAISS_TOMBSTONE_PATH", faiss_tombstone),
        patch("consolidation_memory.config.FAISS_RELOAD_SIGNAL", faiss_reload),
        patch("consolidation_memory.config.KNOWLEDGE_DIR", knowledge),
        patch("consolidation_memory.config.KNOWLEDGE_VERSIONS_DIR", knowledge_versions),
        patch("consolidation_memory.config.CONSOLIDATION_LOG_DIR", consol_log),
        patch("consolidation_memory.config.LOG_DIR", log_dir),
        patch("consolidation_memory.config.BACKUP_DIR", backup_dir),
        patch("consolidation_memory.config.EMBEDDING_DIMENSION", test_dim),
        patch("consolidation_memory.config.EMBEDDING_BACKEND", "fastembed"),
        # Patch modules that import these at module level
        patch("consolidation_memory.database.DB_PATH", db_path),
        patch("consolidation_memory.vector_store.FAISS_INDEX_PATH", faiss_idx),
        patch("consolidation_memory.vector_store.FAISS_ID_MAP_PATH", faiss_map),
        patch("consolidation_memory.vector_store.FAISS_TOMBSTONE_PATH", faiss_tombstone),
        patch("consolidation_memory.vector_store.FAISS_RELOAD_SIGNAL", faiss_reload),
        patch("consolidation_memory.vector_store.EMBEDDING_DIMENSION", test_dim),
        patch("consolidation_memory.consolidation.KNOWLEDGE_DIR", knowledge),
        patch("consolidation_memory.consolidation.KNOWLEDGE_VERSIONS_DIR", knowledge_versions),
        patch("consolidation_memory.consolidation.CONSOLIDATION_LOG_DIR", consol_log),
        patch("consolidation_memory.consolidation.CONSOLIDATION_PRUNE_ENABLED", False),
    ]

    for p in patches:
        p.start()

    # Clear thread-local DB connection from previous tests
    import consolidation_memory.database as database
    if hasattr(database._local, "conn"):
        try:
            database._local.conn.close()
        except Exception:
            pass
        database._local.conn = None

    yield tmp_path

    for p in patches:
        p.stop()

    if hasattr(database._local, "conn"):
        try:
            database._local.conn.close()
        except Exception:
            pass
        database._local.conn = None
