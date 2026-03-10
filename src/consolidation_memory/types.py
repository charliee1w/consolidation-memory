"""Result types for consolidation-memory operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, TypedDict


# ── Consolidation run status values ─────────────────────────────────────────
# These match the strings stored in the consolidation_runs.status DB column.

RunStatus = Literal["running", "completed", "failed"]

RUN_STATUS_RUNNING: RunStatus = "running"
RUN_STATUS_COMPLETED: RunStatus = "completed"
RUN_STATUS_FAILED: RunStatus = "failed"


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


# ── Canonical domain-model scope skeleton ────────────────────────────────────

NamespaceSharingMode = Literal["private", "shared", "team", "managed"]
AppClientType = Literal[
    "mcp",
    "python_sdk",
    "rest",
    "openai_agents",
    "langgraph",
    "adk",
    "letta",
    "cli",
    "other",
]
SessionKind = Literal["conversation", "thread", "workflow", "job"]
PolicyReadVisibility = Literal["private", "namespace", "project"]
PolicyWriteMode = Literal["allow", "deny"]
PolicyResolutionSource = Literal["scope_policy", "persisted_acl"]


@dataclass(frozen=True)
class NamespaceScope:
    """Top-level namespace identity for intentional sharing boundaries."""

    id: str | None = None
    slug: str = "default"
    display_name: str | None = None
    sharing_mode: NamespaceSharingMode = "private"


@dataclass(frozen=True)
class AppClientScope:
    """Calling application or client identity."""

    id: str | None = None
    app_type: AppClientType = "python_sdk"
    name: str = "legacy_client"
    provider: str | None = None
    external_key: str | None = None


@dataclass(frozen=True)
class AgentScope:
    """Logical agent identity inside an app client."""

    id: str | None = None
    name: str | None = None
    external_key: str | None = None
    model_provider: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class SessionScope:
    """Short-lived interaction context identity."""

    id: str | None = None
    external_key: str | None = None
    session_kind: SessionKind = "conversation"


@dataclass(frozen=True)
class ProjectRepoScope:
    """Project or repository context identity."""

    id: str | None = None
    slug: str | None = None
    display_name: str | None = None
    root_uri: str | None = None
    repo_remote: str | None = None
    default_branch: str | None = None


@dataclass(frozen=True)
class PolicyScope:
    """Explicit access-policy controls layered on top of scope identity."""

    read_visibility: PolicyReadVisibility = "private"
    write_mode: PolicyWriteMode = "allow"


@dataclass(frozen=True)
class ScopeEnvelope:
    """Canonical scope envelope for read/write operations."""

    namespace: NamespaceScope = field(default_factory=NamespaceScope)
    app_client: AppClientScope = field(default_factory=AppClientScope)
    agent: AgentScope | None = None
    session: SessionScope | None = None
    project: ProjectRepoScope | None = None
    policy: PolicyScope | None = None


@dataclass(frozen=True)
class ResolvedScopeEnvelope:
    """Resolved scope envelope with backward-compatible defaults applied."""

    namespace: NamespaceScope
    app_client: AppClientScope
    project: ProjectRepoScope
    agent: AgentScope | None = None
    session: SessionScope | None = None
    policy: PolicyScope = field(default_factory=PolicyScope)
    policy_source: PolicyResolutionSource = "scope_policy"
    policy_acl_matches: int = 0


@dataclass(frozen=True)
class MemoryOperationContext:
    """Service-layer operation context placeholder for later canonicalization."""

    scope: ResolvedScopeEnvelope


class ScopeResolver(Protocol):
    """Protocol for components that resolve canonical scope envelopes."""

    def resolve_scope(
        self,
        scope: ScopeEnvelope | Mapping[str, Any] | None = None,
    ) -> ResolvedScopeEnvelope:
        """Resolve scope input into a stable envelope."""


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_namespace_sharing_mode(value: object) -> NamespaceSharingMode:
    cleaned = _clean_str(value)
    if cleaned == "private":
        return "private"
    if cleaned == "shared":
        return "shared"
    if cleaned == "team":
        return "team"
    if cleaned == "managed":
        return "managed"
    return "private"


def _coerce_app_client_type(value: object) -> AppClientType:
    cleaned = _clean_str(value)
    if cleaned == "mcp":
        return "mcp"
    if cleaned == "python_sdk":
        return "python_sdk"
    if cleaned == "rest":
        return "rest"
    if cleaned == "openai_agents":
        return "openai_agents"
    if cleaned == "langgraph":
        return "langgraph"
    if cleaned == "adk":
        return "adk"
    if cleaned == "letta":
        return "letta"
    if cleaned == "cli":
        return "cli"
    if cleaned == "other":
        return "other"
    return "python_sdk"


def _coerce_session_kind(value: object) -> SessionKind:
    cleaned = _clean_str(value)
    if cleaned == "conversation":
        return "conversation"
    if cleaned == "thread":
        return "thread"
    if cleaned == "workflow":
        return "workflow"
    if cleaned == "job":
        return "job"
    return "conversation"


def _coerce_policy_read_visibility(value: object) -> PolicyReadVisibility:
    cleaned = _clean_str(value)
    if cleaned == "private":
        return "private"
    if cleaned == "namespace":
        return "namespace"
    if cleaned == "project":
        return "project"
    return "private"


def _coerce_policy_write_mode(value: object) -> PolicyWriteMode:
    cleaned = _clean_str(value)
    if cleaned == "deny":
        return "deny"
    if cleaned == "allow":
        return "allow"
    return "allow"


def coerce_scope_envelope(
    scope: ScopeEnvelope | Mapping[str, Any] | None,
) -> ScopeEnvelope | None:
    """Coerce user/tool input into a typed scope envelope.

    Accepts:
      - ``None`` (returns ``None``)
      - an already-typed ``ScopeEnvelope``
      - a dict-like structure with canonical keys
    """
    if scope is None:
        return None
    if isinstance(scope, ScopeEnvelope):
        return scope
    if not isinstance(scope, Mapping):
        raise TypeError("scope must be a ScopeEnvelope, mapping, or None")

    namespace_raw = scope.get("namespace")
    if isinstance(namespace_raw, NamespaceScope):
        namespace = namespace_raw
    elif isinstance(namespace_raw, str):
        namespace = NamespaceScope(slug=namespace_raw)
    else:
        ns = _as_mapping(namespace_raw)
        namespace = NamespaceScope(
            id=_clean_str(ns.get("id")),
            slug=_clean_str(ns.get("slug")) or "default",
            display_name=_clean_str(ns.get("display_name")),
            sharing_mode=_coerce_namespace_sharing_mode(ns.get("sharing_mode")),
        )

    app_raw = scope.get("app_client", scope.get("app"))
    if isinstance(app_raw, AppClientScope):
        app_client = app_raw
    elif isinstance(app_raw, str):
        app_client = AppClientScope(name=app_raw)
    else:
        app = _as_mapping(app_raw)
        app_client = AppClientScope(
            id=_clean_str(app.get("id")),
            app_type=_coerce_app_client_type(app.get("app_type")),
            name=_clean_str(app.get("name")) or "legacy_client",
            provider=_clean_str(app.get("provider")),
            external_key=_clean_str(app.get("external_key")),
        )

    agent_raw = scope.get("agent")
    if isinstance(agent_raw, AgentScope):
        agent = agent_raw
    else:
        agent_map = _as_mapping(agent_raw)
        agent = None
        if agent_map:
            agent = AgentScope(
                id=_clean_str(agent_map.get("id")),
                name=_clean_str(agent_map.get("name")),
                external_key=_clean_str(agent_map.get("external_key")),
                model_provider=_clean_str(agent_map.get("model_provider")),
                model_name=_clean_str(agent_map.get("model_name")),
            )

    session_raw = scope.get("session")
    if isinstance(session_raw, SessionScope):
        session = session_raw
    else:
        session_map = _as_mapping(session_raw)
        session = None
        if session_map:
            session = SessionScope(
                id=_clean_str(session_map.get("id")),
                external_key=_clean_str(session_map.get("external_key")),
                session_kind=_coerce_session_kind(session_map.get("session_kind")),
            )

    project_raw = scope.get("project", scope.get("project_repo"))
    if isinstance(project_raw, ProjectRepoScope):
        project = project_raw
    else:
        project_map = _as_mapping(project_raw)
        project = None
        if project_map:
            project = ProjectRepoScope(
                id=_clean_str(project_map.get("id")),
                slug=_clean_str(project_map.get("slug")),
                display_name=_clean_str(project_map.get("display_name")),
                root_uri=_clean_str(project_map.get("root_uri")),
                repo_remote=_clean_str(project_map.get("repo_remote")),
                default_branch=_clean_str(project_map.get("default_branch")),
            )

    policy_raw = scope.get("policy")
    if isinstance(policy_raw, PolicyScope):
        policy = policy_raw
    else:
        policy_map = _as_mapping(policy_raw)
        policy = None
        if policy_map:
            policy = PolicyScope(
                read_visibility=_coerce_policy_read_visibility(
                    policy_map.get("read_visibility")
                ),
                write_mode=_coerce_policy_write_mode(policy_map.get("write_mode")),
            )

    return ScopeEnvelope(
        namespace=namespace,
        app_client=app_client,
        agent=agent,
        session=session,
        project=project,
        policy=policy,
    )


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


class ClaimQueryResult(TypedDict):
    """Row shape for claim-query database results."""
    id: str
    claim_type: str
    canonical_text: str
    payload: str
    status: str
    confidence: float
    valid_from: str
    valid_until: str | None
    created_at: str
    updated_at: str


class DriftAnchor(TypedDict):
    """Anchor used for drift impact lookups."""
    anchor_type: str
    anchor_value: str


class DriftClaimImpact(TypedDict):
    """Per-claim drift impact status transition."""
    claim_id: str
    previous_status: str
    new_status: str
    matched_anchors: list[DriftAnchor]


class DriftOutput(TypedDict):
    """Aggregated drift-detection output payload."""
    checked_anchors: list[DriftAnchor]
    impacted_claim_ids: list[str]
    challenged_claim_ids: list[str]
    impacts: list[DriftClaimImpact]


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

    status: Literal["stored", "duplicate_detected", "backend_unavailable", "write_denied"]
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
    claims: list[dict[str, Any]] = field(default_factory=list)
    total_episodes: int = 0
    total_knowledge_topics: int = 0
    message: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ForgetResult:
    """Result of a memory forget operation."""

    status: Literal["forgotten", "not_found", "write_denied"]
    id: str = ""
    message: str | None = None


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
    utility_scheduler: dict[str, Any] | None = None
    knowledge_consistency: dict[str, Any] | None = None
    scaling: dict[str, Any] | None = None


@dataclass
class ExportResult:
    """Result of a memory export operation."""

    status: Literal["exported"]
    path: str = ""
    episodes: int = 0
    knowledge_topics: int = 0
    claims: int = 0
    claim_edges: int = 0
    claim_sources: int = 0
    claim_events: int = 0
    episode_anchors: int = 0


@dataclass
class CorrectResult:
    """Result of a knowledge correction operation."""

    status: Literal["corrected", "not_found", "error", "write_denied"]
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
class ClaimBrowseResult:
    """Result of browsing claims."""

    claims: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    claim_type: str | None = None
    as_of: str | None = None
    message: str | None = None


@dataclass
class ClaimSearchResult:
    """Result of searching claims."""

    claims: list[dict[str, Any]] = field(default_factory=list)
    total_matches: int = 0
    query: str | None = None
    claim_type: str | None = None
    as_of: str | None = None
    message: str | None = None


@dataclass
class BatchStoreResult:
    """Result of a batch memory store operation."""

    status: Literal["stored", "backend_unavailable", "write_denied"]
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

    status: Literal["protected", "not_found", "error", "write_denied"]
    protected_count: int = 0
    message: str = ""


@dataclass
class ContradictionResult:
    """Result of querying the contradiction audit log."""

    contradictions: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    topic: str | None = None


@dataclass
class ConsolidationLogResult:
    """Result of querying the consolidation changelog."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    message: str = ""
