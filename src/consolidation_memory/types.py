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


OutcomeType = Literal[
    "success",
    "failure",
    "partial_success",
    "reverted",
    "superseded",
]

OUTCOME_TYPES: tuple[OutcomeType, ...] = (
    "success",
    "failure",
    "partial_success",
    "reverted",
    "superseded",
)


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


_NAMESPACE_SCOPE_FLAT_KEYS: dict[str, str] = {
    "namespace_id": "id",
    "namespace_slug": "slug",
    "namespace_display_name": "display_name",
    "namespace_sharing_mode": "sharing_mode",
}
_APP_CLIENT_SCOPE_FLAT_KEYS: dict[str, str] = {
    "app_client_id": "id",
    "app_client_type": "app_type",
    "app_client_name": "name",
    "app_client_provider": "provider",
    "app_client_external_key": "external_key",
}
_AGENT_SCOPE_FLAT_KEYS: dict[str, str] = {
    "agent_id": "id",
    "agent_name": "name",
    "agent_external_key": "external_key",
    "agent_model_provider": "model_provider",
    "agent_model_name": "model_name",
}
_SESSION_SCOPE_FLAT_KEYS: dict[str, str] = {
    "session_id": "id",
    "session_external_key": "external_key",
    "session_kind": "session_kind",
}
_PROJECT_SCOPE_FLAT_KEYS: dict[str, str] = {
    "project_id": "id",
    "project_slug": "slug",
    "project_display_name": "display_name",
    "project_root_uri": "root_uri",
    "project_repo_remote": "repo_remote",
    "project_default_branch": "default_branch",
}
_POLICY_SCOPE_FLAT_KEYS: dict[str, str] = {
    "read_visibility": "read_visibility",
    "write_mode": "write_mode",
}


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"scope.{field_name} must be an object when provided")


def _normalize_scope_mapping(scope: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in scope.items():
        if not isinstance(key, str):
            raise TypeError("scope keys must be strings")
        normalized[key] = value
    return normalized


def _merge_scope_section(
    scope: Mapping[str, Any],
    *,
    canonical_key: str,
    alias_key: str | None = None,
    flat_keys: Mapping[str, str] | None = None,
    string_field: str | None = None,
) -> object:
    raw = scope.get(canonical_key)
    if raw is None and alias_key is not None:
        raw = scope.get(alias_key)

    if raw is not None and not isinstance(raw, (Mapping, str)):
        return raw

    merged: dict[str, Any] = {}
    if flat_keys is not None:
        for flat_key, nested_key in flat_keys.items():
            value = scope.get(flat_key)
            if value is not None:
                merged[nested_key] = value

    if isinstance(raw, str):
        if string_field is None:
            return raw
        merged.setdefault(string_field, raw)
    elif isinstance(raw, Mapping):
        for key, value in raw.items():
            if not isinstance(key, str):
                raise TypeError(f"scope.{canonical_key} keys must be strings")
            merged[key] = value

    return merged or raw


def _coerce_optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"scope.{field_name} must be a string when provided")
    cleaned = value.strip()
    return cleaned or None


def _coerce_namespace_sharing_mode(value: object) -> NamespaceSharingMode:
    cleaned = _coerce_optional_str(value, field_name="namespace.sharing_mode")
    if cleaned is None:
        return "private"
    if cleaned == "private":
        return "private"
    if cleaned == "shared":
        return "shared"
    if cleaned == "team":
        return "team"
    if cleaned == "managed":
        return "managed"
    raise ValueError(
        "scope.namespace.sharing_mode must be one of: private, shared, team, managed"
    )


def _coerce_app_client_type(value: object) -> AppClientType:
    cleaned = _coerce_optional_str(value, field_name="app_client.app_type")
    if cleaned is None:
        return "python_sdk"
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
    raise ValueError(
        "scope.app_client.app_type must be one of: "
        "mcp, python_sdk, rest, openai_agents, langgraph, adk, letta, cli, other"
    )


def _coerce_session_kind(value: object) -> SessionKind:
    cleaned = _coerce_optional_str(value, field_name="session.session_kind")
    if cleaned is None:
        return "conversation"
    if cleaned == "conversation":
        return "conversation"
    if cleaned == "thread":
        return "thread"
    if cleaned == "workflow":
        return "workflow"
    if cleaned == "job":
        return "job"
    raise ValueError("scope.session.session_kind must be one of: conversation, thread, workflow, job")


def _coerce_policy_read_visibility(value: object) -> PolicyReadVisibility:
    cleaned = _coerce_optional_str(value, field_name="policy.read_visibility")
    if cleaned is None:
        return "private"
    if cleaned == "private":
        return "private"
    if cleaned == "namespace":
        return "namespace"
    if cleaned == "project":
        return "project"
    raise ValueError("scope.policy.read_visibility must be one of: private, namespace, project")


def _coerce_policy_write_mode(value: object) -> PolicyWriteMode:
    cleaned = _coerce_optional_str(value, field_name="policy.write_mode")
    if cleaned is None:
        return "allow"
    if cleaned == "deny":
        return "deny"
    if cleaned == "allow":
        return "allow"
    raise ValueError("scope.policy.write_mode must be one of: allow, deny")


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
    normalized_scope = _normalize_scope_mapping(scope)

    namespace_raw = _merge_scope_section(
        normalized_scope,
        canonical_key="namespace",
        flat_keys=_NAMESPACE_SCOPE_FLAT_KEYS,
        string_field="slug",
    )
    if isinstance(namespace_raw, NamespaceScope):
        namespace = namespace_raw
    else:
        ns = _as_mapping(namespace_raw, field_name="namespace")
        namespace = NamespaceScope(
            id=_coerce_optional_str(ns.get("id"), field_name="namespace.id"),
            slug=_coerce_optional_str(ns.get("slug"), field_name="namespace.slug") or "default",
            display_name=_coerce_optional_str(
                ns.get("display_name"), field_name="namespace.display_name"
            ),
            sharing_mode=_coerce_namespace_sharing_mode(ns.get("sharing_mode")),
        )

    app_raw = _merge_scope_section(
        normalized_scope,
        canonical_key="app_client",
        alias_key="app",
        flat_keys=_APP_CLIENT_SCOPE_FLAT_KEYS,
        string_field="name",
    )
    if isinstance(app_raw, AppClientScope):
        app_client = app_raw
    else:
        app = _as_mapping(app_raw, field_name="app_client")
        app_client = AppClientScope(
            id=_coerce_optional_str(app.get("id"), field_name="app_client.id"),
            app_type=_coerce_app_client_type(app.get("app_type")),
            name=_coerce_optional_str(app.get("name"), field_name="app_client.name") or "legacy_client",
            provider=_coerce_optional_str(
                app.get("provider"), field_name="app_client.provider"
            ),
            external_key=_coerce_optional_str(
                app.get("external_key"), field_name="app_client.external_key"
            ),
        )

    agent_raw = _merge_scope_section(
        normalized_scope,
        canonical_key="agent",
        flat_keys=_AGENT_SCOPE_FLAT_KEYS,
        string_field="name",
    )
    if isinstance(agent_raw, AgentScope):
        agent = agent_raw
    else:
        agent_map = _as_mapping(agent_raw, field_name="agent")
        agent = None
        if agent_map:
            agent = AgentScope(
                id=_coerce_optional_str(agent_map.get("id"), field_name="agent.id"),
                name=_coerce_optional_str(agent_map.get("name"), field_name="agent.name"),
                external_key=_coerce_optional_str(
                    agent_map.get("external_key"), field_name="agent.external_key"
                ),
                model_provider=_coerce_optional_str(
                    agent_map.get("model_provider"), field_name="agent.model_provider"
                ),
                model_name=_coerce_optional_str(
                    agent_map.get("model_name"), field_name="agent.model_name"
                ),
            )

    session_raw = _merge_scope_section(
        normalized_scope,
        canonical_key="session",
        flat_keys=_SESSION_SCOPE_FLAT_KEYS,
        string_field="external_key",
    )
    if isinstance(session_raw, SessionScope):
        session = session_raw
    else:
        session_map = _as_mapping(session_raw, field_name="session")
        session = None
        if session_map:
            session = SessionScope(
                id=_coerce_optional_str(session_map.get("id"), field_name="session.id"),
                external_key=_coerce_optional_str(
                    session_map.get("external_key"), field_name="session.external_key"
                ),
                session_kind=_coerce_session_kind(session_map.get("session_kind")),
            )

    project_raw = _merge_scope_section(
        normalized_scope,
        canonical_key="project",
        alias_key="project_repo",
        flat_keys=_PROJECT_SCOPE_FLAT_KEYS,
        string_field="slug",
    )
    if isinstance(project_raw, ProjectRepoScope):
        project = project_raw
    else:
        project_map = _as_mapping(project_raw, field_name="project")
        project = None
        if project_map:
            project = ProjectRepoScope(
                id=_coerce_optional_str(project_map.get("id"), field_name="project.id"),
                slug=_coerce_optional_str(project_map.get("slug"), field_name="project.slug"),
                display_name=_coerce_optional_str(
                    project_map.get("display_name"), field_name="project.display_name"
                ),
                root_uri=_coerce_optional_str(
                    project_map.get("root_uri"), field_name="project.root_uri"
                ),
                repo_remote=_coerce_optional_str(
                    project_map.get("repo_remote"), field_name="project.repo_remote"
                ),
                default_branch=_coerce_optional_str(
                    project_map.get("default_branch"), field_name="project.default_branch"
                ),
            )

    policy_raw = _merge_scope_section(
        normalized_scope,
        canonical_key="policy",
        flat_keys=_POLICY_SCOPE_FLAT_KEYS,
    )
    if isinstance(policy_raw, PolicyScope):
        policy = policy_raw
    else:
        policy_map = _as_mapping(policy_raw, field_name="policy")
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
    trust_profile: dict[str, Any] | None = None


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
    action_outcomes: int = 0
    action_outcome_sources: int = 0
    action_outcome_refs: int = 0


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
class OutcomeRecordResult:
    """Result of recording an action outcome observation."""

    status: Literal["recorded", "write_denied"]
    id: str | None = None
    action_key: str | None = None
    outcome_type: OutcomeType | None = None
    observed_at: str | None = None
    message: str | None = None


@dataclass
class OutcomeBrowseResult:
    """Result of browsing outcome observations."""

    outcomes: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    outcome_type: OutcomeType | None = None
    action_key: str | None = None
    source_claim_id: str | None = None
    source_record_id: str | None = None
    source_episode_id: str | None = None
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
