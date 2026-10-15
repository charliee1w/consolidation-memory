"""Published output contracts for MCP tool results.

Each model mirrors one tool's dispatch payload:

- strict JSON types — a value of the wrong type is refused instead of coerced;
- ``extra="allow"`` — payload keys added upstream are kept, never dropped;
- required fields follow the shapes the dispatcher always emits (see
  ``types.py`` dataclasses and captured payloads);
- ``Field(description=...)`` documents every property for clients and models;
- contracts that publish a ``types.py`` result type are checked against it at
  import time, so the wire promise cannot drift from the payload.

The MCP server wraps each contract's JSON schema as ``anyOf[success, error]``
before publishing (see ``server._publish_output_schemas``), so tool execution
error payloads (``{"error": "..."}``) validate against the same schema.
"""

from __future__ import annotations

from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

from consolidation_memory.types import HygieneApplyResult

__all__ = [
    "AskOutput",
    "BatchStoreOutput",
    "BrowseOutput",
    "ClaimBrowseOutput",
    "ClaimSearchOutput",
    "CompactOutput",
    "ConsolidationLogOutput",
    "ConsolidationOutput",
    "ContradictionOutput",
    "CorrectOutput",
    "DecayReportOutput",
    "DriftAnchorOutput",
    "DriftClaimImpactOutput",
    "DriftScanOutput",
    "EpisodicBufferOutput",
    "ExportOutput",
    "ForgetOutput",
    "HealthOutput",
    "HygieneApplyOutput",
    "HygieneScanOutput",
    "KnowledgeBaseOutput",
    "OutcomeBrowseOutput",
    "OutcomeRecordOutput",
    "PolicyGrantOutput",
    "PolicyListOutput",
    "ProtectOutput",
    "RecallOutput",
    "RecordsByTypeOutput",
    "ScopeEnvelopeUsage",
    "ScopeListAgent",
    "ScopeListAppClient",
    "ScopeListNamespace",
    "ScopeListOutput",
    "ScopeListProject",
    "ScopeListSession",
    "ScopeUsageCounts",
    "ScopeUsageEntry",
    "SearchOutput",
    "StatusOutput",
    "StoreOutput",
    "TimelineOutput",
    "TopicDetailOutput",
]


class _Output(BaseModel):
    """Shared contract config: strict types, forward-compatible extra keys."""

    model_config = ConfigDict(extra="allow", strict=True)


# ── Nested building blocks ───────────────────────────────────────────────────


class EpisodicBufferOutput(_Output):
    """Row counts of the episodes table."""

    total: int = Field(description="Episodes currently stored.")
    pending_consolidation: int = Field(description="Episodes not yet consolidated.")
    consolidated: int = Field(description="Episodes already merged into knowledge.")
    pruned: int = Field(description="Episodes removed by decay/pruning.")


class RecordsByTypeOutput(_Output):
    """Knowledge record counts grouped by record type."""

    facts: int = Field(description="Records of type fact.")
    solutions: int = Field(description="Records of type solution.")
    preferences: int = Field(description="Records of type preference.")
    procedures: int = Field(description="Records of type procedure.")
    strategies: int = Field(description="Records of type strategy.")


class KnowledgeBaseOutput(_Output):
    """Totals for topics and records in the knowledge base."""

    total_topics: int = Field(description="Consolidated knowledge topics.")
    total_facts: int = Field(description="Facts extracted into knowledge.")
    total_records: int = Field(description="All knowledge records combined.")
    records_by_type: RecordsByTypeOutput = Field(description="Breakdown by record type.")


class HealthOutput(_Output):
    """Health assessment of the memory system."""

    status: str = Field(description="Overall state: healthy, degraded or error.")
    issues: list[str] = Field(description="Human-readable problems, empty when healthy.")
    backend_reachable: bool = Field(description="Whether the embedding backend responded.")


class DriftAnchorOutput(_Output):
    """A code anchor scanned for drift."""

    anchor_type: str = Field(description="Anchor kind, e.g. file path or symbol.")
    anchor_value: str = Field(description="The concrete anchor value that was checked.")


class DriftClaimImpactOutput(_Output):
    """Status transition of one claim impacted by drift."""

    claim_id: str = Field(description="Identifier of the impacted claim.")
    previous_status: str = Field(description="Claim status before the drift challenge.")
    new_status: str = Field(description="Claim status after the drift challenge.")
    matched_anchors: list[DriftAnchorOutput] = Field(
        description="Anchors that linked the code change to this claim."
    )


# ── Tool contracts ───────────────────────────────────────────────────────────


class StoreOutput(_Output):
    """Result of storing one episode (memory_store and memory_remember)."""

    status: Literal["stored", "duplicate_detected", "backend_unavailable", "write_denied"] = Field(
        description="Outcome: stored, duplicate_detected, backend_unavailable or write_denied."
    )
    id: str | None = Field(description="Episode id, null when nothing was written.")
    content_type: str | None = Field(description="Content type actually stored.")
    tags: list[str] = Field(description="Tags stored with the episode.")
    existing_id: str | None = Field(description="Id of the near-duplicate episode, if detected.")
    similarity: float | None = Field(description="Similarity score of the detected duplicate.")
    message: str | None = Field(description="Additional human-readable detail, if any.")
    shape_warnings: list[str] = Field(description="Non-blocking data quality warnings.")


class BatchStoreOutput(_Output):
    """Result of storing several episodes in one call."""

    status: Literal["stored", "backend_unavailable", "write_denied"] = Field(
        description="Overall outcome of the batch write."
    )
    stored: int = Field(description="Number of episodes written.")
    duplicates: int = Field(description="Number of episodes skipped as duplicates.")
    results: list[dict[str, Any]] = Field(description="Per-episode outcome rows.")


class RecallOutput(_Output):
    """Semantic recall envelope with episodes, knowledge, records and claims."""

    episodes: list[dict[str, Any]] = Field(description="Matching episodes, best score first.")
    knowledge: list[dict[str, Any]] = Field(description="Matching consolidated knowledge topics.")
    records: list[dict[str, Any]] = Field(description="Matching knowledge records.")
    claims: list[dict[str, Any]] = Field(description="Matching claims from the claim graph.")
    total_episodes: int = Field(description="Number of returned episodes.")
    total_knowledge_topics: int = Field(description="Number of returned knowledge topics.")
    message: str | None = Field(description="Notes about the query, e.g. fallbacks applied.")
    warnings: list[str] = Field(description="Degradation warnings, empty on the fast path.")
    entity_resolution: dict[str, Any] | None = Field(
        default=None, description="Entity linking details when the resolver ran."
    )


class AskOutput(_Output):
    """Simplified recall envelope for plain-language questions."""

    episodes: list[dict[str, Any]] = Field(description="Episode previews with id, kind and score.")
    knowledge: list[dict[str, Any]] = Field(description="Knowledge topic previews.")
    records: list[dict[str, Any]] = Field(description="Matching knowledge records.")
    claims: list[dict[str, Any]] = Field(description="Matching claims.")
    total_episodes: int = Field(description="Number of returned episodes.")
    query: str = Field(description="Normalized query that was executed.")
    message: str | None = Field(description="Notes about the query, if any.")
    warnings: list[str] = Field(description="Degradation warnings, empty on the fast path.")


class SearchOutput(_Output):
    """Keyword and metadata search over episodes."""

    episodes: list[dict[str, Any]] = Field(description="Matching episodes, newest first.")
    total_matches: int = Field(description="Number of returned episodes.")
    query: str | None = Field(description="The keyword query, null when browsing by filters.")
    message: str | None = Field(description="Additional notes, e.g. when nothing matched.")


class StatusOutput(_Output):
    """Full statistics snapshot of the memory system."""

    episodic_buffer: EpisodicBufferOutput | None = Field(
        description="Episode table counters."
    )
    knowledge_base: KnowledgeBaseOutput | None = Field(
        description="Topic and record counters."
    )
    last_consolidation: dict[str, Any] | None = Field(
        description="Summary of the most recent consolidation run, null before the first run."
    )
    embedding_backend: str = Field(description="Configured embedding backend name.")
    embedding_model: str = Field(description="Configured embedding model name.")
    faiss_index_size: int = Field(description="Vectors currently in the FAISS index.")
    faiss_tombstones: int = Field(description="Tombstoned vectors awaiting compaction.")
    db_size_mb: float = Field(description="SQLite database size in megabytes.")
    version: str = Field(description="Package version of consolidation-memory.")
    health: HealthOutput | None = Field(description="Health assessment, null when unavailable.")
    consolidation_metrics: list[dict[str, Any]] = Field(
        description="Recent consolidation metric rows."
    )
    consolidation_quality: dict[str, Any] | None = Field(
        description="Aggregate quality stats: runs_analyzed, success_rate, avg_confidence, rates."
    )
    fast_path_hits: int = Field(description="Consolidations answered without an LLM call.")
    llm_fallbacks: int = Field(description="Consolidations that fell back to the LLM.")
    recent_activity: list[dict[str, Any]] = Field(description="Recent activity rows.")
    utility_scheduler: dict[str, Any] | None = Field(
        description="Utility scheduling state: score, weights, signals, next_due_at, last trigger."
    )
    knowledge_consistency: dict[str, Any] | None = Field(
        description="Markdown consistency scan summary for the knowledge base."
    )
    scaling: dict[str, Any] | None = Field(description="Scaling/backpressure counters.")
    trust_profile: dict[str, Any] | None = Field(description="Trust and ACL profile snapshot.")


class ClaimBrowseOutput(_Output):
    """Paged view over the claim graph."""

    claims: list[dict[str, Any]] = Field(description="Claim rows matching the filters.")
    total: int = Field(description="Number of returned claims.")
    claim_type: str | None = Field(description="Applied claim_type filter, if any.")
    as_of: str | None = Field(description="Temporal snapshot used, if any.")
    message: str | None = Field(description="Additional notes, if any.")


class ClaimSearchOutput(_Output):
    """Full-text claim search with optional temporal filtering."""

    claims: list[dict[str, Any]] = Field(description="Claim rows matching the query.")
    total_matches: int = Field(description="Number of returned claims.")
    query: str | None = Field(description="The search query.")
    claim_type: str | None = Field(description="Applied claim_type filter, if any.")
    as_of: str | None = Field(description="Temporal snapshot used, if any.")
    message: str | None = Field(description="Additional notes, if any.")


class OutcomeRecordOutput(_Output):
    """Result of recording an action outcome observation."""

    status: Literal["recorded", "write_denied"] = Field(
        description="Outcome: recorded or write_denied."
    )
    id: str | None = Field(description="Id of the created outcome row, if recorded.")
    action_key: str | None = Field(description="Deduplication key for the action, if provided.")
    outcome_type: str | None = Field(description="Classified outcome type, if resolved.")
    observed_at: str | None = Field(description="Observation timestamp, null when defaulted.")
    message: str | None = Field(description="Additional human-readable detail, if any.")


class OutcomeBrowseOutput(_Output):
    """Paged view over recorded action outcomes."""

    outcomes: list[dict[str, Any]] = Field(description="Outcome rows matching the filters.")
    total: int = Field(description="Number of returned outcomes.")
    outcome_type: str | None = Field(description="Applied outcome_type filter, if any.")
    action_key: str | None = Field(description="Applied action_key filter, if any.")
    source_claim_id: str | None = Field(description="Applied claim source filter, if any.")
    source_record_id: str | None = Field(description="Applied record source filter, if any.")
    source_episode_id: str | None = Field(description="Applied episode source filter, if any.")
    as_of: str | None = Field(description="Temporal snapshot used, if any.")
    message: str | None = Field(description="Additional notes, if any.")


class DriftScanOutput(_Output):
    """Aggregated drift-detection report."""

    checked_anchors: list[DriftAnchorOutput] = Field(
        description="Anchors that were scanned for changes."
    )
    impacted_claim_ids: list[str] = Field(description="Claims whose anchors changed.")
    challenged_claim_ids: list[str] = Field(description="Claims moved to challenged status.")
    impacts: list[DriftClaimImpactOutput] = Field(
        description="Per-claim status transitions caused by the drift."
    )
    message: str | None = Field(
        default=None,
        description="Present on degraded runs: explains timeouts or fallback scans.",
    )
    error: str | None = Field(
        default=None,
        description=(
            "Present when the result is a failure, not a successful empty scan: "
            "carries the same text as the isError content block, so a client "
            "validating against the error arm of this schema can tell a timeout "
            "from an empty result."
        ),
    )


class ForgetOutput(_Output):
    """Result of marking an episode for removal."""

    status: Literal["forgotten", "not_found", "write_denied"] = Field(
        description="Outcome: forgotten, not_found or write_denied."
    )
    id: str = Field(description="Episode id the call referred to.")
    message: str | None = Field(description="Additional human-readable detail, if any.")


class ExportOutput(_Output):
    """Result of exporting a JSON snapshot of the corpus."""

    status: Literal["exported"] = Field(description="Outcome: exported.")
    path: str = Field(description="Filesystem path of the written snapshot.")
    episodes: int = Field(description="Episodes included in the snapshot.")
    knowledge_topics: int = Field(description="Knowledge topics included.")
    claims: int = Field(description="Claims included.")
    claim_edges: int = Field(description="Claim graph edges included.")
    claim_sources: int = Field(description="Claim provenance links included.")
    claim_events: int = Field(description="Claim lifecycle events included.")
    episode_anchors: int = Field(description="Episode code anchors included.")
    action_outcomes: int = Field(description="Action outcomes included.")
    action_outcome_sources: int = Field(description="Outcome provenance links included.")
    action_outcome_refs: int = Field(description="Outcome reference rows included.")


class CorrectOutput(_Output):
    """Result of correcting a knowledge topic."""

    status: Literal["corrected", "not_found", "error", "write_denied"] = Field(
        description="Outcome: corrected, not_found, error or write_denied."
    )
    filename: str | None = Field(description="Topic file the call referred to.")
    title: str | None = Field(description="Topic title after the correction, if applied.")
    message: str | None = Field(description="Additional human-readable detail, if any.")


class CompactOutput(_Output):
    """Result of compacting the FAISS index."""

    status: Literal["compacted", "no_tombstones"] = Field(
        description="Outcome: compacted or no_tombstones."
    )
    tombstones_removed: int = Field(description="Tombstoned vectors removed.")
    index_size: int = Field(description="Vectors remaining in the index.")


class ConsolidationOutput(_Output):
    """Consolidation run report; early exits carry only a subset of fields."""

    run_id: str | None = Field(default=None, description="Identifier of the consolidation run.")
    timestamp: str | None = Field(default=None, description="Run start time (ISO 8601).")
    episodes_loaded: int | None = Field(default=None, description="Episodes read for the run.")
    episodes_with_vectors: int | None = Field(
        default=None, description="Episodes that produced embeddings."
    )
    clusters_total: int | None = Field(default=None, description="Clusters formed.")
    clusters_valid: int | None = Field(default=None, description="Clusters that passed checks.")
    clusters_failed: int | None = Field(default=None, description="Clusters rejected by checks.")
    topics_created: int | None = Field(default=None, description="New topics written.")
    topics_updated: int | None = Field(default=None, description="Existing topics updated.")
    episodes_pruned: int | None = Field(default=None, description="Episodes pruned by decay.")
    surprise_adjusted: int | None = Field(
        default=None, description="Episodes whose surprise score was adjusted."
    )
    api_calls: int | None = Field(default=None, description="LLM API calls spent.")
    fast_path_hits: int | None = Field(
        default=None, description="Clusters answered without an LLM call."
    )
    llm_fallbacks: int | None = Field(default=None, description="Clusters that used the LLM.")
    failed_episode_ids: list[str] | None = Field(
        default=None, description="Episodes whose processing failed."
    )
    failure_linked_episodes_loaded: int | None = Field(
        default=None, description="Episodes pulled in to prioritize known failures."
    )
    clusters_prioritized_by_failures: int | None = Field(
        default=None, description="Clusters reprioritized because of linked failures."
    )
    status: str | None = Field(
        default=None, description="Early-exit state, e.g. already_running or error."
    )
    message: str | None = Field(default=None, description="Explanation for early exits.")
    episodes: int | None = Field(
        default=None, description="Episode count reported by an early exit."
    )


class ConsolidationLogOutput(_Output):
    """Recent consolidation activity as changelog entries."""

    entries: list[dict[str, Any]] = Field(description="Changelog rows, newest first.")
    total: int = Field(description="Number of returned entries.")
    message: str = Field(description="Notes when no entries are available.")


class DecayReportOutput(_Output):
    """What pruning would remove right now."""

    prunable_episodes: int = Field(description="Episodes eligible for decay pruning.")
    low_confidence_records: int = Field(description="Records below the confidence threshold.")
    protected_episodes: int = Field(description="Episodes marked immune to pruning.")
    details: dict[str, Any] = Field(
        description="Policy settings and candidate ids: prune_after_days, decay_policies, lists."
    )


class ProtectOutput(_Output):
    """Result of marking episodes immune to pruning."""

    status: Literal["protected", "not_found", "error", "write_denied"] = Field(
        description="Outcome: protected, not_found, error or write_denied."
    )
    protected_count: int = Field(description="Episodes actually marked protected.")
    message: str = Field(description="Human-readable outcome detail.")


class TimelineOutput(_Output):
    """Temporal view of how a topic evolved."""

    query: str = Field(description="Topic the timeline was built for.")
    entries: list[dict[str, Any]] = Field(description="Timeline entries, oldest first.")
    total: int = Field(description="Number of returned entries.")
    message: str = Field(description="Notes when no entries are available.")


class ContradictionOutput(_Output):
    """Detected contradictions from the audit log."""

    contradictions: list[dict[str, Any]] = Field(description="Contradiction rows.")
    total: int = Field(description="Number of returned contradictions.")
    topic: str | None = Field(description="Topic filter applied, if any.")


class BrowseOutput(_Output):
    """All knowledge topics with summaries."""

    topics: list[dict[str, Any]] = Field(description="Topic rows with title and summary.")
    total: int = Field(description="Number of returned topics.")


class TopicDetailOutput(_Output):
    """Full markdown content of one knowledge topic."""

    status: Literal["ok", "not_found", "error"] = Field(
        description="Outcome: ok, not_found or error."
    )
    filename: str = Field(description="Topic file name requested.")
    content: str = Field(description="Full markdown body, empty when not found.")
    message: str = Field(description="Human-readable outcome detail.")


class HygieneScanOutput(_Output):
    """Noise and orphan scan over the corpus."""

    status: str = Field(description="Scan outcome, e.g. ok.")
    episodes: dict[str, Any] = Field(
        description="Episode noise analysis: total_active, per-type counts, "
        "recommended_cleanup_ids, would_remain."
    )
    orphaned_claims: dict[str, Any] = Field(
        description="Claims without live provenance: count, ids, samples."
    )
    stale_episode_sources: dict[str, Any] = Field(
        description="Episode source rows that reference missing episodes."
    )


class HygieneApplyOutput(_Output):
    """Result of applying corpus hygiene cleanup.

    Mirrors ``types.HygieneApplyResult``, the payload
    ``corpus_hygiene.apply_corpus_hygiene`` actually builds; both modes send
    every property. ``_assert_mirrors_result_type`` fails at import if the two
    ever drift.
    """

    status: Literal["dry_run", "applied"] = Field(
        description="Outcome: dry_run (preview, corpus untouched) or applied (cleanup committed)."
    )
    episode_targets: int = Field(description="Episodes selected for cleanup.")
    episode_ids: list[str] = Field(description="Ids of the selected episodes.")
    forgotten: int = Field(description="Episodes actually forgotten; 0 for a dry run.")
    not_found: int = Field(description="Selected episode ids that were already gone; 0 for a dry run.")
    expire_orphans: bool = Field(description="Whether orphaned claims were expired.")
    orphan_repair: dict[str, Any] | None = Field(
        description="Orphan repair summary, null when not requested."
    )


class PolicyListOutput(_Output):
    """Persisted access policies and ACL bindings."""

    status: str = Field(description="Outcome, e.g. ok.")
    count: int = Field(description="Number of returned bindings.")
    policies: list[dict[str, Any]] = Field(description="Policy/ACL binding rows.")


class PolicyGrantOutput(_Output):
    """Result of creating or updating a policy ACL binding."""

    status: str = Field(description="Outcome, e.g. granted.")
    policy_id: str = Field(description="Id of the policy row.")
    principal_id: str = Field(description="Id of the principal row.")
    acl_entry_id: str = Field(description="Id of the created ACL binding.")
    namespace: str | None = Field(description="Namespace scope of the binding, if any.")
    project: str | None = Field(description="Project scope of the binding, if any.")
    principal_type: str = Field(description="Principal kind the binding applies to.")
    principal_key: str = Field(description="Concrete principal identifier.")
    write_mode: str | None = Field(description="Granted write mode, if any.")
    read_visibility: str | None = Field(description="Granted read visibility, if any.")


class ScopeListNamespace(_Output):
    """Namespace identity of a discovered scope."""

    slug: str = Field(description="Namespace slug, e.g. default.")
    sharing_mode: str | None = Field(
        description="Namespace sharing mode: private, shared, team or managed."
    )


class ScopeListAppClient(_Output):
    """Calling application identity of a discovered scope."""

    name: str = Field(description="App client name, e.g. legacy_client.")
    app_type: str = Field(description="App client kind: mcp, python_sdk, rest, cli, ...")
    provider: str | None = Field(description="Upstream provider identifier, when registered.")
    external_key: str | None = Field(description="External key of the app client, when set.")


class ScopeListAgent(_Output):
    """Agent identity of a discovered scope, null when rows carry no agent."""

    name: str | None = Field(description="Agent name, when recorded.")
    external_key: str | None = Field(description="Stable external agent key, when recorded.")


class ScopeListSession(_Output):
    """Session identity of a discovered scope, null when rows carry no session."""

    external_key: str | None = Field(description="Stable external session key, when recorded.")
    session_kind: str | None = Field(
        description="Session kind: conversation, thread, workflow or job."
    )


class ScopeListProject(_Output):
    """Project or repository identity of a discovered scope."""

    slug: str = Field(description="Project slug.")
    display_name: str | None = Field(description="Human-readable project name, when set.")
    root_uri: str | None = Field(description="Project root URI, when recorded.")
    repo_remote: str | None = Field(description="Git remote URL, when recorded.")
    default_branch: str | None = Field(description="Default branch, when recorded.")


class ScopeEnvelopeUsage(_Output):
    """Canonical scope envelope for a discovered scope.

    Shape matches the scope argument accepted by other tools, so an entry can
    be passed back as ``scope`` without transformation.
    """

    namespace: ScopeListNamespace = Field(description="Namespace identity.")
    app_client: ScopeListAppClient = Field(description="Calling app identity.")
    agent: ScopeListAgent | None = Field(
        description="Agent identity, null when rows carry no agent."
    )
    session: ScopeListSession | None = Field(
        description="Session identity, null when rows carry no session."
    )
    project: ScopeListProject = Field(description="Project identity.")


class ScopeUsageCounts(_Output):
    """Row counts of the scope per data table."""

    episodes: int = Field(description="Episodes stored under this scope.")
    records: int = Field(description="Knowledge records stored under this scope.")
    topics: int = Field(description="Knowledge topics stored under this scope.")


class ScopeUsageEntry(_Output):
    """One discovered scope with usage statistics."""

    scope: ScopeEnvelopeUsage = Field(
        description="Canonical scope envelope, reusable as the scope argument of other tools."
    )
    counts: ScopeUsageCounts = Field(description="Row counts per data table.")
    last_used_at: str | None = Field(
        description="Most recent created_at timestamp across the scope's rows (ISO 8601)."
    )


class ScopeListOutput(_Output):
    """Result of discovering existing scopes."""

    scopes: list[ScopeUsageEntry] = Field(
        description="Discovered scopes, most recently used first."
    )
    total: int = Field(description="Total number of discovered scopes before limiting.")
    offset: int = Field(description="0-based offset this window starts at.")
    message: str | None = Field(
        description="Set when the window does not cover every scope; explains how to page."
    )


def _assert_mirrors_result_type(contract: type[_Output], result_type: type[Any]) -> None:
    """Fail at import when a contract drifts from the result type it publishes.

    The dataclass is what the producer sends and the contract is what the wire
    promises. A field on one side only either rejects an otherwise valid
    payload — after the tool already applied its side effects — or advertises a
    key that never arrives, so the mismatch has to break the process instead of
    one client's cleanup.
    """
    hints = get_type_hints(result_type)
    declared = {name: spec.annotation for name, spec in contract.model_fields.items()}
    if set(declared) != set(hints):
        raise RuntimeError(
            f"{contract.__name__} does not mirror {result_type.__name__}: "
            f"missing={sorted(set(hints) - set(declared))} "
            f"extra={sorted(set(declared) - set(hints))}"
        )
    for name, annotation in declared.items():
        if annotation != hints[name]:
            raise RuntimeError(
                f"{contract.__name__}.{name} is {annotation!r}, but "
                f"{result_type.__name__} sends {hints[name]!r}"
            )
        if not contract.model_fields[name].is_required():
            raise RuntimeError(
                f"{contract.__name__}.{name} must stay required: the producer sends "
                "every field, so a missing key is a bug, not a valid payload"
            )


_assert_mirrors_result_type(HygieneApplyOutput, HygieneApplyResult)
