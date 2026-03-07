"""Minimal plugin example for builder onboarding.

Use this as a starting point for extension work.
"""

from __future__ import annotations

from consolidation_memory.plugins import PluginBase


class RecallAuditPlugin(PluginBase):
    name = "recall-audit"

    def on_recall(self, query: str, result: object) -> None:
        # Replace print with your own telemetry sink.
        print(f"[{self.name}] query={query!r}")
