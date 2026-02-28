"""Consolidation Memory — persistent semantic memory for AI conversations."""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("consolidation-memory")

# Lazy imports to avoid pulling in heavy deps (faiss, numpy) on bare import.
_LAZY_IMPORTS = {
    "MemoryClient": "consolidation_memory.client",
    "StoreResult": "consolidation_memory.types",
    "RecallResult": "consolidation_memory.types",
    "ForgetResult": "consolidation_memory.types",
    "StatusResult": "consolidation_memory.types",
    "ExportResult": "consolidation_memory.types",
    "CorrectResult": "consolidation_memory.types",
    "SearchResult": "consolidation_memory.types",
    "BatchStoreResult": "consolidation_memory.types",
    "ConsolidationReport": "consolidation_memory.types",
    "ConsolidationQuality": "consolidation_memory.types",
    "EpisodicBufferStats": "consolidation_memory.types",
    "KnowledgeBaseStats": "consolidation_memory.types",
    "HealthStatus": "consolidation_memory.types",
    "StatsDict": "consolidation_memory.types",
}

__all__ = ["__version__", *_LAZY_IMPORTS]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
