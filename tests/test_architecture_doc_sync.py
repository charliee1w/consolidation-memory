"""Tests that keep architecture documentation aligned with live implementation."""

from __future__ import annotations

from pathlib import Path

from consolidation_memory.database import CURRENT_SCHEMA_VERSION


def _architecture_doc() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")


def test_architecture_doc_mentions_current_schema_version():
    doc = _architecture_doc()
    assert f"CURRENT_SCHEMA_VERSION = {CURRENT_SCHEMA_VERSION}" in doc


def test_architecture_doc_mentions_cross_process_faiss_write_lease():
    doc = _architecture_doc()
    assert ".faiss_write.lock" in doc


def test_architecture_doc_mentions_scheduler_and_claim_tables():
    doc = _architecture_doc()
    for table_name in (
        "`claims`",
        "`claim_edges`",
        "`claim_sources`",
        "`claim_events`",
        "`episode_anchors`",
        "`action_outcomes`",
        "`action_outcome_sources`",
        "`action_outcome_refs`",
        "`consolidation_scheduler`",
    ):
        assert table_name in doc
