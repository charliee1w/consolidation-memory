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
    "ClaimBrowseResult": "consolidation_memory.types",
    "ClaimSearchResult": "consolidation_memory.types",
    "BatchStoreResult": "consolidation_memory.types",
    "ConsolidationReport": "consolidation_memory.types",
    "ConsolidationQuality": "consolidation_memory.types",
    "EpisodicBufferStats": "consolidation_memory.types",
    "KnowledgeBaseStats": "consolidation_memory.types",
    "HealthStatus": "consolidation_memory.types",
    "StatsDict": "consolidation_memory.types",
    "CompactResult": "consolidation_memory.types",
    "BrowseResult": "consolidation_memory.types",
    "TopicDetailResult": "consolidation_memory.types",
    "TimelineResult": "consolidation_memory.types",
    "DecayReportResult": "consolidation_memory.types",
    "ProtectResult": "consolidation_memory.types",
    "ContradictionResult": "consolidation_memory.types",
    "ContentType": "consolidation_memory.types",
    "RecordType": "consolidation_memory.types",
    "NamespaceScope": "consolidation_memory.types",
    "AppClientScope": "consolidation_memory.types",
    "AgentScope": "consolidation_memory.types",
    "SessionScope": "consolidation_memory.types",
    "ProjectRepoScope": "consolidation_memory.types",
    "ScopeEnvelope": "consolidation_memory.types",
    "ResolvedScopeEnvelope": "consolidation_memory.types",
    "MemoryOperationContext": "consolidation_memory.types",
    "coerce_scope_envelope": "consolidation_memory.types",
    "RunStatus": "consolidation_memory.types",
    "RUN_STATUS_RUNNING": "consolidation_memory.types",
    "RUN_STATUS_COMPLETED": "consolidation_memory.types",
    "RUN_STATUS_FAILED": "consolidation_memory.types",
}

__all__ = ["__version__", *_LAZY_IMPORTS]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
