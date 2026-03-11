"""Minimal recall-audit plugin example."""

from __future__ import annotations

from typing import Any

from consolidation_memory.plugins import PluginBase


class RecallAuditPlugin(PluginBase):
    name = "recall-audit"

    def on_recall(self, query: str, result: Any) -> None:
        episode_count = len(getattr(result, "episodes", []))
        topic_count = len(getattr(result, "knowledge", []))
        record_count = len(getattr(result, "records", []))
        claim_count = len(getattr(result, "claims", []))
        print(
            f"[{self.name}] query={query!r} "
            f"episodes={episode_count} topics={topic_count} "
            f"records={record_count} claims={claim_count}"
        )
