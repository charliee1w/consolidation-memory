"""Shared fixtures for all test modules."""

import pytest

from consolidation_memory.config import reset_config


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path):
    """Override config paths to use temp directory for ALL tests."""
    data_dir = tmp_path / "data" / "projects" / "default"
    data_dir.mkdir(parents=True)
    (data_dir / "knowledge").mkdir()
    (data_dir / "knowledge" / "versions").mkdir()
    (data_dir / "consolidation_logs").mkdir()
    (tmp_path / "data" / "logs").mkdir(parents=True)
    (data_dir / "backups").mkdir()

    # Tests use 384-dim vectors regardless of user config
    reset_config(
        _base_data_dir=tmp_path / "data",
        active_project="default",
        EMBEDDING_DIMENSION=384,
        EMBEDDING_BACKEND="fastembed",
    )

    # Close all thread-local DB connections from previous tests
    import consolidation_memory.database as database
    database.close_all_connections()

    # Reset backends and circuit breakers so state doesn't leak between tests
    from consolidation_memory.backends import reset_backends
    reset_backends()

    # Reset module-level caches so state doesn't leak between tests
    from consolidation_memory import claim_cache, topic_cache, record_cache
    topic_cache.invalidate()
    record_cache.invalidate()
    claim_cache.invalidate()

    # Reset plugin manager singleton so registered plugins don't leak between tests
    from consolidation_memory.plugins import reset_plugin_manager
    reset_plugin_manager()

    yield tmp_path

    # Close ALL connections (including those from spawned threads)
    database.close_all_connections()
