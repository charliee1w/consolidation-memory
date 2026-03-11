"""Regression tests for knowledge topic file path resolution surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def test_context_assembler_reads_storage_filename_when_logical_file_missing(tmp_data_dir):
    from consolidation_memory.config import get_config
    from consolidation_memory.context_assembler import _search_knowledge

    cfg = get_config()
    query_vec = np.ones(384, dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    topic = {
        "title": "Storage-backed Topic",
        "filename": "logical.md",
        "storage_filename": "logical__storage.md",
        "summary": "storage path test",
        "confidence": 0.9,
        "source_episodes": "[]",
    }

    storage_path = cfg.KNOWLEDGE_DIR / topic["storage_filename"]
    storage_path.write_text("content from storage filename", encoding="utf-8")

    with (
        patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
        patch("consolidation_memory.context_assembler.increment_topic_access"),
        patch("consolidation_memory.context_assembler._apply_evolving_topic_signals"),
    ):
        mock_tc.get_topic_vecs.return_value = ([topic], np.stack([query_vec]))
        topics, _warnings = _search_knowledge("storage path test", query_vec)

    assert len(topics) == 1
    assert topics[0]["content"] == "content from storage filename"


def test_export_reads_storage_filename_when_logical_file_missing(tmp_data_dir):
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.config import get_config
    from consolidation_memory.database import ensure_schema, get_knowledge_topic, upsert_knowledge_topic

    ensure_schema()
    cfg = get_config()
    filename = "export-storage.md"
    upsert_knowledge_topic(
        filename=filename,
        title="Export Storage",
        summary="storage-backed export",
        source_episodes=[],
    )

    topic = get_knowledge_topic(filename)
    assert topic is not None
    storage_filename = str(topic["storage_filename"])

    storage_path = cfg.KNOWLEDGE_DIR / storage_filename
    storage_path.write_text("# Exported\nstorage content", encoding="utf-8")
    logical_path = cfg.KNOWLEDGE_DIR / filename
    if logical_path.exists():
        logical_path.unlink()

    client = MemoryClient(auto_consolidate=False)
    try:
        result = client.export()
    finally:
        client.close()

    snapshot = json.loads(Path(result.path).read_text(encoding="utf-8"))
    exported_topic = next(t for t in snapshot["knowledge_topics"] if t["filename"] == filename)
    assert exported_topic["file_content"] == "# Exported\nstorage content"
