"""Consolidation Memory — engineering knowledge layer for agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Sentinel used when the distribution metadata is unavailable, e.g. running from
# a source checkout that was never installed, or from a subprocess whose
# interpreter cannot see the dist-info. It stays a valid PEP 440 version so every
# consumer that formats or reports ``__version__`` keeps working, and it sorts
# below every real 0.x release so it can never masquerade as a newer build.
_UNKNOWN_VERSION = "0.0.0"

try:
    __version__ = _pkg_version("consolidation-memory")
except PackageNotFoundError:
    # Only a missing distribution is tolerated; any other lookup failure is a
    # real error and must stay visible.
    __version__ = _UNKNOWN_VERSION

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
    "OutcomeRecordResult": "consolidation_memory.types",
    "OutcomeBrowseResult": "consolidation_memory.types",
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
    "ConsolidationLogResult": "consolidation_memory.types",
    "HygieneApplyResult": "consolidation_memory.types",
    "ContentType": "consolidation_memory.types",
    "RecordType": "consolidation_memory.types",
    "OutcomeType": "consolidation_memory.types",
    "OUTCOME_TYPES": "consolidation_memory.types",
    "NamespaceScope": "consolidation_memory.types",
    "AppClientScope": "consolidation_memory.types",
    "AgentScope": "consolidation_memory.types",
    "SessionScope": "consolidation_memory.types",
    "ProjectRepoScope": "consolidation_memory.types",
    "PolicyScope": "consolidation_memory.types",
    "ScopeEnvelope": "consolidation_memory.types",
    "ResolvedScopeEnvelope": "consolidation_memory.types",
    "MemoryOperationContext": "consolidation_memory.types",
    "coerce_scope_envelope": "consolidation_memory.types",
    "RunStatus": "consolidation_memory.types",
    "RUN_STATUS_RUNNING": "consolidation_memory.types",
    "RUN_STATUS_COMPLETED": "consolidation_memory.types",
    "RUN_STATUS_FAILED": "consolidation_memory.types",
}

# The suppression below is deliberate: the names are the keys of a dict[str, str],
# so every entry is a string by construction, but the rule cannot see through the
# unpacking, and the concatenated form trips RUF005 instead.
__all__ = ["__version__", *list(_LAZY_IMPORTS)]  # noqa: PLE0604


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
