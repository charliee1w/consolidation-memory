"""Consolidation Memory — persistent semantic memory for AI conversations."""

__version__ = "0.1.0"

# Lazy imports to avoid pulling in heavy deps (faiss, numpy) on bare import.
_LAZY_IMPORTS = {
    "MemoryClient": "consolidation_memory.client",
    "StoreResult": "consolidation_memory.types",
    "RecallResult": "consolidation_memory.types",
    "ForgetResult": "consolidation_memory.types",
    "StatusResult": "consolidation_memory.types",
    "ExportResult": "consolidation_memory.types",
    "CorrectResult": "consolidation_memory.types",
}

__all__ = ["__version__", *_LAZY_IMPORTS]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
