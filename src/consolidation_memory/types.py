"""Result types for consolidation-memory operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict


class ContentType(str, Enum):
    """Valid episode content types."""
    EXCHANGE = "exchange"
    FACT = "fact"
    SOLUTION = "solution"
    PREFERENCE = "preference"


class RecordType(str, Enum):
    """Valid knowledge record types."""
    FACT = "fact"
    SOLUTION = "solution"
    PREFERENCE = "preference"
    PROCEDURE = "procedure"


# ── TypedDicts for structured dict returns ────────────────────────────────────


class EpisodicBufferStats(TypedDict):
    """Stats for the episodic buffer (episodes table)."""
    total: int
    pending_consolidation: int
    consolidated: int
    pruned: int


class RecordsByType(TypedDict):
    """Breakdown of knowledge records by type."""
    facts: int
    solutions: int
    preferences: int
    procedures: int


class KnowledgeBaseStats(TypedDict):
    """Stats for the knowledge base (topics + records)."""
    total_topics: int
    total_facts: int
    total_records: int
    records_by_type: RecordsByType


class StatsDict(TypedDict):
    """Return type of database.get_stats()."""
    episodic_buffer: EpisodicBufferStats
    knowledge_base: KnowledgeBaseStats


class HealthStatus(TypedDict):
    """Health assessment of the memory system."""
    status: str  # "healthy" | "degraded" | "error" — dynamically computed
    issues: list[str]
    backend_reachable: bool


class ConsolidationQuality(TypedDict):
    """Aggregate quality stats from recent consolidation metrics."""
    runs_analyzed: int
    total_clusters_processed: int
    success_rate: float
    avg_confidence: float
    total_api_calls: int
    total_episodes_processed: int


class ConsolidationReport(TypedDict, total=False):
    """Return type of run_consolidation().

    On success, contains detailed run metrics. On early exit or error,
    contains a subset with ``status`` and optionally ``message`` / ``episodes``.
    """
    # Present in success reports
    run_id: str
    timestamp: str
    episodes_loaded: int
    episodes_with_vectors: int
    clusters_total: int
    clusters_valid: int
    clusters_failed: int
    topics_created: int
    topics_updated: int
    episodes_pruned: int
    surprise_adjusted: int
    api_calls: int
    failed_episode_ids: list[str]
    # Present in early exit / error cases
    status: str
    message: str
    episodes: int


# ── Dataclass result types ────────────────────────────────────────────────────


@dataclass
class StoreResult:
    """Result of a memory store operation."""

    status: Literal["stored", "duplicate_detected", "backend_unavailable"]
    id: str | None = None
    content_type: str | None = None
    tags: list[str] = field(default_factory=list)
    existing_id: str | None = None
    similarity: float | None = None
    message: str | None = None


@dataclass
class RecallResult:
    """Result of a memory recall operation."""

    episodes: list[dict[str, Any]] = field(default_factory=list)
    knowledge: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    total_episodes: int = 0
    total_knowledge_topics: int = 0
    message: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ForgetResult:
    """Result of a memory forget operation."""

    status: Literal["forgotten", "not_found"]
    id: str = ""


@dataclass
class StatusResult:
    """Result of a memory status query."""

    episodic_buffer: EpisodicBufferStats | None = None
    knowledge_base: KnowledgeBaseStats | None = None
    last_consolidation: dict[str, Any] | None = None
    embedding_backend: str = ""
    embedding_model: str = ""
    faiss_index_size: int = 0
    faiss_tombstones: int = 0
    db_size_mb: float = 0.0
    version: str = ""
    health: HealthStatus | None = None
    consolidation_metrics: list[dict[str, Any]] = field(default_factory=list)
    consolidation_quality: ConsolidationQuality | dict[str, Any] | None = None
    recent_activity: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExportResult:
    """Result of a memory export operation."""

    status: Literal["exported"]
    path: str = ""
    episodes: int = 0
    knowledge_topics: int = 0


@dataclass
class CorrectResult:
    """Result of a knowledge correction operation."""

    status: Literal["corrected", "not_found", "error"]
    filename: str | None = None
    title: str | None = None
    message: str | None = None


@dataclass
class SearchResult:
    """Result of a keyword/metadata search operation."""

    episodes: list[dict[str, Any]] = field(default_factory=list)
    total_matches: int = 0
    query: str | None = None
    message: str | None = None


@dataclass
class BatchStoreResult:
    """Result of a batch memory store operation."""

    status: Literal["stored", "backend_unavailable"]
    stored: int = 0
    duplicates: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CompactResult:
    """Result of a FAISS compaction operation."""

    status: Literal["compacted", "no_tombstones"]
    tombstones_removed: int = 0
    index_size: int = 0


@dataclass
class BrowseResult:
    """Result of browsing knowledge topics."""

    topics: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0


@dataclass
class TopicDetailResult:
    """Result of reading a single knowledge topic."""

    status: Literal["ok", "not_found", "error"]
    filename: str = ""
    content: str = ""
    message: str = ""


@dataclass
class TimelineResult:
    """Result of a temporal timeline query."""

    query: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    message: str = ""


@dataclass
class DecayReportResult:
    """Result of a decay/pruning report."""

    prunable_episodes: int = 0
    low_confidence_records: int = 0
    protected_episodes: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtectResult:
    """Result of protecting episodes from pruning."""

    status: Literal["protected", "not_found", "error"]
    protected_count: int = 0
    message: str = ""


@dataclass
class ContradictionResult:
    """Result of querying the contradiction audit log."""

    contradictions: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    topic: str | None = None
