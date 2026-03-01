"""Consolidation package.

Public API: ``run_consolidation`` only.
Internal functions should be imported from their submodules directly.
"""

from consolidation_memory.consolidation.engine import run_consolidation

__all__ = ["run_consolidation"]
