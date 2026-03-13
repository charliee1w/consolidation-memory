"""Pure Python client for consolidation-memory.

Provides MemoryClient — the single source of truth for all memory operations.
MCP server and REST API are thin wrappers over this class.

Usage::

    from consolidation_memory import MemoryClient

    with MemoryClient() as mem:
        mem.store("learned X about Y", content_type="fact", tags=["python"])
        results = mem.recall("how does Y work")
        print(results.episodes)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

import numpy as np

from consolidation_memory import __version__
from consolidation_memory.client_runtime import (
    check_embedding_backend as _check_embedding_backend_runtime,
    compute_consolidation_utility as _compute_consolidation_utility_runtime,
    compute_health as _compute_health_runtime,
    consolidation_loop as _consolidation_loop_runtime,
    _compute_force_thresholds as _compute_force_thresholds_runtime,
    finalize_auto_consolidation as _finalize_auto_consolidation_runtime,
    maybe_auto_consolidate as _maybe_auto_consolidate_runtime,
    probe_backend as _probe_backend_runtime,
    record_recall_signal as _record_recall_signal_runtime,
    recent_recall_signal_counts as _recent_recall_signal_counts_runtime,
    should_trigger_consolidation as _should_trigger_consolidation_runtime,
    should_trigger_scheduler_run as _should_trigger_scheduler_run_runtime,
    start_consolidation_thread as _start_consolidation_thread_runtime,
    submit_auto_consolidation as _submit_auto_consolidation_runtime,
)
from consolidation_memory.markdown_records import parse_markdown_records
from consolidation_memory.query_service import (
    CanonicalQueryService,
    ClaimBrowseQuery,
    ClaimSearchQuery,
    DriftQuery,
    EpisodeSearchQuery,
    RecallQuery,
)
from consolidation_memory.policy_engine import (
    principal_tokens_for_scope,
    resolve_effective_policy,
)
from consolidation_memory.utils import parse_datetime, parse_json_list
from consolidation_memory.types import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    AppClientScope,
    AgentScope,
    MemoryOperationContext,
    NamespaceScope,
    PolicyScope,
    ProjectRepoScope,
    ResolvedScopeEnvelope,
    ScopeEnvelope,
    SessionScope,
    coerce_scope_envelope,
    ContentType,
    ConsolidationLogResult,
    ConsolidationReport,
    ContradictionResult,
    HealthStatus,
    StoreResult,
    BatchStoreResult,
    RecallResult,
    SearchResult,
    ClaimBrowseResult,
    ClaimSearchResult,
    DriftOutput,
    ForgetResult,
    StatusResult,
    ExportResult,
    CorrectResult,
    CompactResult,
    BrowseResult,
    TopicDetailResult,
    TimelineResult,
    DecayReportResult,
    ProtectResult,
)

logger = logging.getLogger("consolidation_memory")

_DEFAULT_NAMESPACE_SLUG = "default"
_DEFAULT_APP_CLIENT_NAME = "legacy_client"
_SHARED_NAMESPACE_MODES = {"shared", "team", "managed"}
_WRITE_DENIED_MESSAGE = "Writes are denied by scope policy (write_mode='deny')."
_CLOSED_MESSAGE = "MemoryClient is closed."


def _normalize_content_type(ct: str) -> str:
    """Validate and normalize a content_type string.

    Returns the value unchanged if it is a valid ContentType, otherwise
    logs a warning and falls back to ``'exchange'``.
    """
    _valid_types = {t.value for t in ContentType}
    if ct in _valid_types:
        return ct
    logger.warning("Invalid content_type %r, defaulting to 'exchange'", ct)
    return ContentType.EXCHANGE.value


def _validate_embedding_batch(
    embeddings: object,
    *,
    expected_count: int,
    operation: str,
) -> np.ndarray:
    """Validate backend embedding output before any durable mutation."""
    from consolidation_memory.config import get_config

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"Embedding backend returned invalid shape {matrix.shape!r} for {operation}; "
            "expected a 2D matrix."
        )
    if matrix.shape[0] != expected_count:
        raise ValueError(
            f"Embedding backend returned {matrix.shape[0]} vectors for {expected_count} texts "
            f"during {operation}."
        )
    expected_dim = int(get_config().EMBEDDING_DIMENSION)
    if matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding backend returned dimension {matrix.shape[1]} during {operation}; "
            f"expected {expected_dim}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Embedding backend returned non-finite values during {operation}.")
    return matrix


def _write_temp_text(path: os.PathLike[str] | str, content: str) -> str:
    """Write text to a sibling temp file and return its path for atomic replace."""
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    prefix = f".{os.path.basename(target)}."
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return temp_path


def _normalize_scope_token(value: str | None) -> str | None:
    """Normalize optional scope strings to non-empty tokens."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolved_scope_to_db_row(scope: ResolvedScopeEnvelope) -> dict[str, str | None]:
    """Convert a resolved scope envelope into DB scope columns."""
    return {
        "namespace_slug": scope.namespace.slug,
        "namespace_sharing_mode": scope.namespace.sharing_mode,
        "app_client_name": scope.app_client.name,
        "app_client_type": scope.app_client.app_type,
        "app_client_provider": scope.app_client.provider,
        "app_client_external_key": scope.app_client.external_key,
        "agent_name": scope.agent.name if scope.agent else None,
        "agent_external_key": scope.agent.external_key if scope.agent else None,
        "session_external_key": scope.session.external_key if scope.session else None,
        "session_kind": scope.session.session_kind if scope.session else None,
        "project_slug": scope.project.slug,
        "project_display_name": scope.project.display_name,
        "project_root_uri": scope.project.root_uri,
        "project_repo_remote": scope.project.repo_remote,
        "project_default_branch": scope.project.default_branch,
    }


def _resolved_scope_to_query_filter(scope: ResolvedScopeEnvelope) -> dict[str, str | None]:
    """Build query-time scope filters from a resolved scope envelope."""
    read_visibility = scope.policy.read_visibility
    filters: dict[str, str | None] = {
        "namespace_slug": scope.namespace.slug,
    }

    # Visibility rules are additive and preserve legacy behavior:
    # - private: legacy semantics (namespace+project, private app isolation unless shared mode)
    # - project: namespace+project, cross-app visibility
    # - namespace: namespace-wide visibility across projects and apps
    if read_visibility != "namespace":
        filters["project_slug"] = scope.project.slug

    shared_namespace = scope.namespace.sharing_mode in _SHARED_NAMESPACE_MODES
    if read_visibility == "private" and not shared_namespace:
        filters["app_client_name"] = scope.app_client.name
        filters["app_client_type"] = scope.app_client.app_type
        if scope.app_client.provider:
            filters["app_client_provider"] = scope.app_client.provider
        if scope.app_client.external_key:
            filters["app_client_external_key"] = scope.app_client.external_key

    if scope.agent is not None:
        if scope.agent.external_key:
            filters["agent_external_key"] = scope.agent.external_key
        elif scope.agent.name:
            filters["agent_name"] = scope.agent.name

    if scope.session is not None:
        if scope.session.external_key:
            filters["session_external_key"] = scope.session.external_key
        filters["session_kind"] = scope.session.session_kind

    return filters


def _resolved_scope_to_mutation_filter(scope: ResolvedScopeEnvelope) -> dict[str, str | None]:
    """Build an exact-scope filter for mutation authorization.

    Writes are intentionally fail-closed: read visibility widening does not grant
    cross-project or cross-app mutation rights.
    """
    filters: dict[str, str | None] = {
        "namespace_slug": scope.namespace.slug,
        "project_slug": scope.project.slug,
        "app_client_name": scope.app_client.name,
        "app_client_type": scope.app_client.app_type,
        "app_client_provider": scope.app_client.provider,
        "app_client_external_key": scope.app_client.external_key,
    }
    if scope.agent is not None:
        if scope.agent.external_key:
            filters["agent_external_key"] = scope.agent.external_key
        elif scope.agent.name:
            filters["agent_name"] = scope.agent.name
    if scope.session is not None:
        if scope.session.external_key:
            filters["session_external_key"] = scope.session.external_key
        filters["session_kind"] = scope.session.session_kind
    return filters


def _write_denied_message(scope: ResolvedScopeEnvelope) -> str | None:
    """Return a policy denial message when writes are disabled for scope."""
    if scope.policy.write_mode == "deny":
        return _WRITE_DENIED_MESSAGE
    return None


def _row_matches_scope_filter(
    row: dict[str, object],
    scope_filter: dict[str, str | None],
) -> bool:
    """Return True when a DB row falls within the provided scope filter."""
    for key, expected in scope_filter.items():
        if expected is None:
            continue
        actual = row.get(key)
        if actual is None:
            return False
        if str(actual) != expected:
            return False
    return True


class MemoryClient:
    """Trust-calibrated working memory client.

    Owns the vector store, database lifecycle, and optional background
    consolidation thread.  All public methods are synchronous and thread-safe.

    Args:
        auto_consolidate: Start background consolidation thread. Default True
            (respects ``CONSOLIDATION_AUTO_RUN`` config).
    """

    _instance_lock = threading.Lock()
    _live_instances = 0

    def __init__(self, auto_consolidate: bool = True) -> None:
        from consolidation_memory.config import get_config
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.vector_store import VectorStore

        cfg = get_config()

        # Ensure directories
        for d in [cfg.DATA_DIR, cfg.KNOWLEDGE_DIR, cfg.CONSOLIDATION_LOG_DIR, cfg.LOG_DIR, cfg.BACKUP_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        ensure_schema()
        self._vector_store = VectorStore()
        self._query_service = CanonicalQueryService(self._vector_store)
        self._check_embedding_backend()

        # Consolidation threading
        self._consolidation_lock = threading.Lock()
        self._consolidation_stop = threading.Event()
        self._consolidation_thread: threading.Thread | None = None
        self._consolidation_pool: ThreadPoolExecutor | None = None
        self._consolidation_future: Future[ConsolidationReport] | None = None
        self._scheduler_owner = f"pid:{os.getpid()}:{uuid.uuid4().hex}"
        self._auto_consolidate_enabled = auto_consolidate

        # Shared executor for LLM calls (e.g. correct())
        self._llm_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm"
        )

        # Cached backend probe result: (is_reachable, timestamp)
        self._probe_cache: tuple[bool, float] | None = None
        self._probe_cache_ttl = 30.0  # seconds

        # Utility scheduler recall signals (recent misses/fallbacks).
        self._scheduler_signal_lock = threading.Lock()
        self._recall_miss_events: deque[float] = deque(maxlen=256)
        self._recall_fallback_events: deque[float] = deque(maxlen=256)
        self._state_lock = threading.Lock()
        self._closing = False
        self._closed = False

        if self._auto_consolidate_enabled and cfg.CONSOLIDATION_AUTO_RUN:
            self._start_consolidation_thread()

        logger.info(
            "MemoryClient initialized (vectors=%d, version=%s)",
            self._vector_store.size,
            __version__,
        )

        # Discover and load plugins, then fire startup hook
        with type(self)._instance_lock:
            type(self)._live_instances += 1
        from consolidation_memory.plugins import get_plugin_manager
        get_plugin_manager().acquire(client=self, auto_load=bool(cfg.PLUGINS_ENABLED))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop background threads. Call when done."""
        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True

        should_close_connections = False

        from consolidation_memory.plugins import get_plugin_manager
        from consolidation_memory.database import close_all_connections

        try:
            self._consolidation_stop.set()
            if self._consolidation_thread is not None:
                self._consolidation_thread.join(timeout=30)
                if self._consolidation_thread.is_alive():
                    logger.warning(
                        "Consolidation thread did not stop within 30s; "
                        "it will terminate when the process exits (daemon thread)."
                    )
                self._consolidation_thread = None
            if self._consolidation_pool is not None:
                self._consolidation_pool.shutdown(wait=True, cancel_futures=True)
                self._consolidation_pool = None
            if self._llm_executor is not None:
                self._llm_executor.shutdown(wait=True, cancel_futures=True)
                self._llm_executor = None
            try:
                get_plugin_manager().release()
            except Exception:
                logger.warning("Plugin shutdown hook failed", exc_info=True)
        finally:
            newly_closed = False
            with self._state_lock:
                if not self._closed:
                    newly_closed = True
                self._closed = True
                self._closing = False
            with type(self)._instance_lock:
                if newly_closed:
                    if type(self)._live_instances > 0:
                        type(self)._live_instances -= 1
                    should_close_connections = type(self)._live_instances == 0

            if should_close_connections:
                close_all_connections()

    def _ensure_open(self) -> None:
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError(_CLOSED_MESSAGE)

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────

    def resolve_scope(
        self,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> ResolvedScopeEnvelope:
        """Resolve canonical scope input using backward-compatible defaults.

        Scope envelope policy remains supported for compatibility. When
        persisted ACL rows match this scope/principal, they are authoritative
        for read/write policy semantics.
        """
        self._ensure_open()
        from consolidation_memory.config import get_active_project

        parsed_scope = coerce_scope_envelope(scope)
        if parsed_scope is None:
            parsed_scope = ScopeEnvelope()

        namespace_slug = (
            _normalize_scope_token(parsed_scope.namespace.slug) or _DEFAULT_NAMESPACE_SLUG
        )
        namespace = NamespaceScope(
            id=_normalize_scope_token(parsed_scope.namespace.id),
            slug=namespace_slug,
            display_name=parsed_scope.namespace.display_name or namespace_slug,
            sharing_mode=parsed_scope.namespace.sharing_mode,
        )

        app_type = parsed_scope.app_client.app_type
        app_name = (
            _normalize_scope_token(parsed_scope.app_client.name) or _DEFAULT_APP_CLIENT_NAME
        )
        app_client = AppClientScope(
            id=_normalize_scope_token(parsed_scope.app_client.id),
            app_type=app_type,
            name=app_name,
            provider=_normalize_scope_token(parsed_scope.app_client.provider),
            external_key=_normalize_scope_token(parsed_scope.app_client.external_key),
        )

        resolved_project = parsed_scope.project or ProjectRepoScope()
        active_project = _normalize_scope_token(get_active_project()) or "default"
        project_slug = (
            _normalize_scope_token(resolved_project.slug)
            or _normalize_scope_token(resolved_project.id)
            or active_project
        )
        project = ProjectRepoScope(
            id=_normalize_scope_token(resolved_project.id),
            slug=project_slug,
            display_name=resolved_project.display_name or project_slug,
            root_uri=_normalize_scope_token(resolved_project.root_uri),
            repo_remote=_normalize_scope_token(resolved_project.repo_remote),
            default_branch=_normalize_scope_token(resolved_project.default_branch),
        )

        agent: AgentScope | None = None
        if parsed_scope.agent is not None:
            agent = AgentScope(
                id=_normalize_scope_token(parsed_scope.agent.id),
                name=_normalize_scope_token(parsed_scope.agent.name),
                external_key=_normalize_scope_token(parsed_scope.agent.external_key),
                model_provider=_normalize_scope_token(parsed_scope.agent.model_provider),
                model_name=_normalize_scope_token(parsed_scope.agent.model_name),
            )

        session: SessionScope | None = None
        if parsed_scope.session is not None:
            session = SessionScope(
                id=_normalize_scope_token(parsed_scope.session.id),
                external_key=_normalize_scope_token(parsed_scope.session.external_key),
                session_kind=parsed_scope.session.session_kind,
            )

        parsed_policy = parsed_scope.policy or PolicyScope()
        envelope_policy = PolicyScope(
            read_visibility=parsed_policy.read_visibility,
            write_mode=parsed_policy.write_mode,
        )

        resolved_scope = ResolvedScopeEnvelope(
            namespace=namespace,
            app_client=app_client,
            project=project,
            agent=agent,
            session=session,
            policy=envelope_policy,
        )

        try:
            from consolidation_memory.database import get_matching_policy_acl_entries

            scope_row = _resolved_scope_to_db_row(resolved_scope)
            principals = principal_tokens_for_scope(resolved_scope)
            acl_rows = get_matching_policy_acl_entries(scope_row, principals)
            effective = resolve_effective_policy(envelope_policy, acl_rows)
            if effective.source == "persisted_acl" and effective.conflicts:
                logger.info(
                    "Resolved persisted ACL conflicts for namespace=%s project=%s: %s",
                    namespace.slug,
                    project.slug,
                    ", ".join(effective.conflicts),
                )
            return ResolvedScopeEnvelope(
                namespace=namespace,
                app_client=app_client,
                project=project,
                agent=agent,
                session=session,
                policy=effective.policy,
                policy_source=effective.source,
                policy_acl_matches=effective.matched_entries,
            )
        except Exception as exc:
            logger.warning(
                "Persisted policy lookup failed for namespace=%s project=%s; "
                "falling back to scope policy: %s",
                namespace.slug,
                project.slug,
                exc,
            )
            return resolved_scope

    def build_operation_context(
        self,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> MemoryOperationContext:
        """Build a canonical operation context for service-layer calls."""
        self._ensure_open()
        return MemoryOperationContext(scope=self.resolve_scope(scope))

    def store_with_scope(
        self,
        content: str,
        content_type: str = "exchange",
        tags: list[str] | None = None,
        surprise: float = 0.5,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> StoreResult:
        """Store an episode with explicit canonical scope metadata."""
        operation_context = self.build_operation_context(scope)
        return self._store_internal(
            content=content,
            content_type=content_type,
            tags=tags,
            surprise=surprise,
            resolved_scope=operation_context.scope,
        )

    def query_recall(
        self,
        query: str,
        n_results: int = 10,
        include_knowledge: bool = True,
        *,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        include_expired: bool = False,
        as_of: str | None = None,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> RecallResult:
        """Canonical recall entrypoint shared by all adapter surfaces."""
        operation_context = self.build_operation_context(scope)
        return self._recall_internal(
            query=query,
            n_results=n_results,
            include_knowledge=include_knowledge,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            include_expired=include_expired,
            as_of=as_of,
            resolved_scope=operation_context.scope,
        )

    def recall_with_scope(
        self,
        query: str,
        n_results: int = 10,
        include_knowledge: bool = True,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        include_expired: bool = False,
        as_of: str | None = None,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> RecallResult:
        """Backward-compatible scoped wrapper for canonical recall."""
        return self.query_recall(
            query=query,
            n_results=n_results,
            include_knowledge=include_knowledge,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            include_expired=include_expired,
            as_of=as_of,
            scope=scope,
        )

    def store_batch_with_scope(
        self,
        episodes: list[dict],
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> BatchStoreResult:
        """Store a batch of episodes with explicit canonical scope metadata."""
        operation_context = self.build_operation_context(scope)
        return self._store_batch_internal(
            episodes=episodes,
            resolved_scope=operation_context.scope,
        )

    def search_with_scope(
        self,
        query: str | None = None,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 20,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> SearchResult:
        """Backward-compatible scoped wrapper for canonical episode search."""
        return self.query_search(
            query=query,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            limit=limit,
            scope=scope,
        )

    def query_search(
        self,
        query: str | None = None,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 20,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> SearchResult:
        """Canonical episode keyword-search entrypoint for all adapters."""
        operation_context = self.build_operation_context(scope)
        return self._search_internal(
            query=query,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            limit=limit,
            resolved_scope=operation_context.scope,
        )

    def store(
        self,
        content: str,
        content_type: str = "exchange",
        tags: list[str] | None = None,
        surprise: float = 0.5,
    ) -> StoreResult:
        """Store a memory episode using backward-compatible default scope."""
        self._ensure_open()
        return self._store_internal(
            content=content,
            content_type=content_type,
            tags=tags,
            surprise=surprise,
            resolved_scope=self.resolve_scope(),
        )

    def _store_internal(
        self,
        *,
        content: str,
        content_type: str,
        tags: list[str] | None,
        surprise: float,
        resolved_scope: ResolvedScopeEnvelope,
    ) -> StoreResult:
        """Store a memory episode with resolved canonical scope."""
        from consolidation_memory.database import (
            get_episode,
            hard_delete_episode,
            insert_episode,
            mark_episode_indexed,
        )
        from consolidation_memory.backends import encode_documents
        from consolidation_memory.config import get_config

        cfg = get_config()
        denied_message = _write_denied_message(resolved_scope)
        if denied_message is not None:
            logger.warning(
                "Denied store for namespace=%s project=%s due to scope policy",
                resolved_scope.namespace.slug,
                resolved_scope.project.slug,
            )
            return StoreResult(status="write_denied", message=denied_message)

        self._vector_store.reload_if_stale()
        scope_row = _resolved_scope_to_db_row(resolved_scope)
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)

        try:
            embedding_matrix = _validate_embedding_batch(
                encode_documents([content]),
                expected_count=1,
                operation="store",
            )
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during store: %s", e)
            return StoreResult(
                status="backend_unavailable",
                message=f"Embedding backend unreachable: {e}",
            )
        embedding = embedding_matrix[0]

        # Dedup check is scope-aware to avoid false positives across isolated contexts.
        if cfg.DEDUP_ENABLED and self._vector_store.size > 0:
            matches = self._vector_store.search(embedding, k=10)
            for match_id, match_sim in matches:
                if match_sim >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                    existing = get_episode(match_id)
                    if existing is not None and _row_matches_scope_filter(existing, scope_filter):
                        logger.info(
                            "Duplicate detected (sim=%.4f >= %.2f): existing=%s scope=%s",
                            match_sim,
                            cfg.DEDUP_SIMILARITY_THRESHOLD,
                            match_id,
                            {k: v for k, v in scope_filter.items() if v is not None},
                        )
                        return StoreResult(
                            status="duplicate_detected",
                            existing_id=match_id,
                            similarity=round(match_sim, 4),
                            message="Content too similar to existing episode. Not stored.",
                        )
                else:
                    break  # Results are sorted by similarity; no point checking lower ones

        content_type = _normalize_content_type(content_type)

        episode_id = insert_episode(
            content=content,
            content_type=content_type,
            tags=tags,
            surprise_score=max(0.0, min(1.0, surprise)),
            scope=scope_row,
            indexed=False,
        )

        vector_added = False
        try:
            self._vector_store.add(episode_id, embedding)
            vector_added = True
            mark_episode_indexed([episode_id])
        except Exception as e:
            vector_rollback_ok = True
            if vector_added:
                try:
                    removed = self._vector_store.remove(episode_id)
                    if not removed:
                        vector_rollback_ok = False
                        logger.error(
                            "Store rollback could not tombstone vector for %s",
                            episode_id,
                        )
                except Exception:
                    vector_rollback_ok = False
                    logger.exception(
                        "Store rollback failed while tombstoning vector for %s",
                        episode_id,
                    )

            db_rollback_ok = True
            if not vector_added or vector_rollback_ok:
                try:
                    if not hard_delete_episode(episode_id):
                        db_rollback_ok = False
                        logger.error(
                            "Store rollback could not hard-delete episode row for %s",
                            episode_id,
                        )
                except Exception:
                    db_rollback_ok = False
                    logger.exception(
                        "Store rollback failed while hard-deleting episode row for %s",
                        episode_id,
                    )
            else:
                db_rollback_ok = False
                logger.error(
                    "Skipped hard-delete rollback for %s because vector rollback failed",
                    episode_id,
                )

            logger.error("Store failed for %s and triggered rollback: %s", episode_id, e)
            if not vector_rollback_ok or not db_rollback_ok:
                raise RuntimeError(
                    f"Store rollback incomplete for {episode_id}; manual reconciliation may be required."
                ) from e
            raise

        self._persist_episode_anchors(episode_id=episode_id, content=content)

        # Tag co-occurrence is non-critical — log and continue on failure
        if tags and len(tags) >= 2:
            from consolidation_memory.database import update_tag_cooccurrence
            try:
                update_tag_cooccurrence(tags)
            except Exception as e:
                logger.warning("Failed to update tag co-occurrence: %s", e)

        logger.info(
            "Stored episode %s (type=%s, surprise=%.2f, tags=%s, namespace=%s, project=%s)",
            episode_id,
            content_type,
            surprise,
            tags,
            scope_row["namespace_slug"],
            scope_row["project_slug"],
        )

        from consolidation_memory.plugins import get_plugin_manager
        get_plugin_manager().fire(
            "on_store",
            episode_id=episode_id,
            content=content,
            content_type=content_type,
            tags=tags or [],
            surprise=surprise,
        )

        result = StoreResult(
            status="stored",
            id=episode_id,
            content_type=content_type,
            tags=tags or [],
        )
        self._maybe_auto_consolidate(trigger_source="store")
        return result

    def store_batch(
        self,
        episodes: list[dict],
    ) -> BatchStoreResult:
        """Store multiple episodes using backward-compatible default scope."""
        self._ensure_open()
        return self._store_batch_internal(
            episodes=episodes,
            resolved_scope=self.resolve_scope(),
        )

    def _store_batch_internal(
        self,
        *,
        episodes: list[dict],
        resolved_scope: ResolvedScopeEnvelope,
    ) -> BatchStoreResult:
        """Store multiple memory episodes in a single operation.

        More efficient than calling store() N times: single embedding call,
        single FAISS batch add, single DB transaction for dedup checks.

        Args:
            episodes: List of dicts, each with keys:
                - content (str, required)
                - content_type (str, default 'exchange')
                - tags (list[str], optional)
                - surprise (float, default 0.5)

        Returns:
            BatchStoreResult with per-episode status.
        """
        from consolidation_memory.database import (
            get_episode,
            hard_delete_episode,
            insert_episode,
            mark_episode_indexed,
        )
        from consolidation_memory.backends import encode_documents
        from consolidation_memory.config import get_config
        import numpy as np

        cfg = get_config()
        denied_message = _write_denied_message(resolved_scope)
        if denied_message is not None:
            logger.warning(
                "Denied batch store for namespace=%s project=%s due to scope policy",
                resolved_scope.namespace.slug,
                resolved_scope.project.slug,
            )
            return BatchStoreResult(
                status="write_denied",
                stored=0,
                duplicates=0,
                results=[{"status": "write_denied", "message": denied_message}],
            )

        if not episodes:
            return BatchStoreResult(status="stored", stored=0, duplicates=0)

        self._vector_store.reload_if_stale()
        scope_row = _resolved_scope_to_db_row(resolved_scope)
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)

        # Validate and normalize
        items = []
        for i, ep in enumerate(episodes):
            if not isinstance(ep, dict) or "content" not in ep:
                logger.warning("Skipping episode %d: missing 'content' key", i)
                continue
            ct = _normalize_content_type(ep.get("content_type", "exchange"))
            try:
                surprise = float(ep.get("surprise", 0.5))
            except (TypeError, ValueError):
                surprise = 0.5
            items.append({
                "content": ep["content"],
                "content_type": ct,
                "tags": ep.get("tags"),
                "surprise": max(0.0, min(1.0, surprise)),
            })

        if not items:
            logger.warning("Batch store received %d episodes but none were valid", len(episodes))
            return BatchStoreResult(status="stored", stored=0, duplicates=0)

        # Single embedding call for all texts
        try:
            embeddings = _validate_embedding_batch(
                encode_documents([it["content"] for it in items]),
                expected_count=len(items),
                operation="store_batch",
            )
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during batch store: %s", e)
            return BatchStoreResult(
                status="backend_unavailable",
                results=[{"status": "error", "message": str(e)}],
            )

        results = []
        stored = 0
        duplicates = 0
        # Collect non-duplicate items for a single FAISS batch add
        pending_ids: list[str] = []
        pending_embs: list[np.ndarray] = []
        # Track accepted embeddings within this batch for intra-batch dedup
        accepted_embs: list[np.ndarray] = []
        # Map episode_id -> item for plugin hooks
        stored_items_by_id: dict[str, dict] = {}

        for i, item in enumerate(items):
            emb = embeddings[i]

            # Dedup check is scope-aware to avoid false positives across isolated contexts.
            if cfg.DEDUP_ENABLED and self._vector_store.size > 0:
                matches = self._vector_store.search(emb, k=10)
                is_dup = False
                for match_id, match_sim in matches:
                    if match_sim >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                        existing = get_episode(match_id)
                        if existing is not None and _row_matches_scope_filter(existing, scope_filter):
                            duplicates += 1
                            results.append({
                                "status": "duplicate_detected",
                                "existing_id": match_id,
                                "similarity": round(float(match_sim), 4),
                            })
                            is_dup = True
                            break
                    else:
                        break
                if is_dup:
                    continue

            # Intra-batch dedup: check against already-accepted items in this batch
            if cfg.DEDUP_ENABLED and accepted_embs:
                emb_norm = emb.reshape(1, -1).astype(np.float32)
                batch_matrix = np.stack(accepted_embs)
                sims = (emb_norm @ batch_matrix.T).flatten()
                if float(sims.max()) >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                    duplicates += 1
                    results.append({
                        "status": "duplicate_detected",
                        "existing_id": pending_ids[int(sims.argmax())],
                        "similarity": round(float(sims.max()), 4),
                    })
                    continue

            episode_id = insert_episode(
                content=item["content"],
                content_type=item["content_type"],
                tags=item["tags"],
                surprise_score=item["surprise"],
                scope=scope_row,
                indexed=False,
            )

            pending_ids.append(episode_id)
            pending_embs.append(emb)
            accepted_embs.append(emb.reshape(-1).astype(np.float32))
            stored_items_by_id[episode_id] = item
            stored += 1
            results.append({
                "status": "stored",
                "id": episode_id,
                "content_type": item["content_type"],
            })

        # Single FAISS batch add instead of per-item add()
        batch_add_succeeded = False
        if pending_ids:
            vectors_added = False
            try:
                emb_matrix = np.stack(pending_embs)
                self._vector_store.add_batch(pending_ids, emb_matrix)
                vectors_added = True
                mark_episode_indexed(pending_ids)
                batch_add_succeeded = True
            except Exception as e:
                db_rollback_safe = True
                if vectors_added:
                    try:
                        removed_count = self._vector_store.remove_batch(pending_ids)
                        if removed_count != len(pending_ids):
                            db_rollback_safe = False
                            logger.error(
                                "Batch store rollback tombstoned %d/%d vectors",
                                removed_count,
                                len(pending_ids),
                            )
                    except Exception:
                        db_rollback_safe = False
                        logger.exception(
                            "Batch store rollback failed while tombstoning vectors",
                        )

                rollback_failures: list[str] = []
                if db_rollback_safe:
                    for eid in pending_ids:
                        try:
                            if not hard_delete_episode(eid):
                                rollback_failures.append(eid)
                        except Exception:
                            rollback_failures.append(eid)
                    if rollback_failures:
                        logger.error(
                            "Batch store rollback failed to hard-delete %d/%d rows",
                            len(rollback_failures),
                            len(pending_ids),
                        )
                else:
                    logger.error(
                        "Skipped DB hard-delete rollback for batch because vector rollback failed"
                    )

                logger.error(
                    "Batch store failed and triggered rollback for %d episodes: %s",
                    len(pending_ids),
                    e,
                )
                stored = 0
                results = [r for r in results if r.get("status") != "stored"]
                results.append({"status": "error", "message": str(e)})

        if batch_add_succeeded:
            for eid in pending_ids:
                if eid not in stored_items_by_id:
                    continue
                stored_item = stored_items_by_id[eid]
                self._persist_episode_anchors(
                    episode_id=eid,
                    content=str(stored_item.get("content", "")),
                )

        # Fire on_store plugin hooks for each successfully stored episode
        if pending_ids:
            from consolidation_memory.plugins import get_plugin_manager
            pm = get_plugin_manager()
            for res in results:
                if res.get("status") == "stored" and "id" in res:
                    eid = str(res["id"])
                    if eid in stored_items_by_id:
                        item = stored_items_by_id[eid]
                        pm.fire(
                            "on_store",
                            episode_id=eid,
                            content=item["content"],
                            content_type=item["content_type"],
                            tags=item.get("tags") or [],
                            surprise=item["surprise"],
                        )

        logger.info("Batch store: %d stored, %d duplicates out of %d", stored, duplicates, len(items))
        batch_result = BatchStoreResult(
            status="stored",
            stored=stored,
            duplicates=duplicates,
            results=results,
        )
        if stored > 0:
            self._maybe_auto_consolidate(trigger_source="store_batch")
        return batch_result

    def recall(
        self,
        query: str,
        n_results: int = 10,
        include_knowledge: bool = True,
        *,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        include_expired: bool = False,
        as_of: str | None = None,
    ) -> RecallResult:
        """Retrieve relevant memories using backward-compatible default scope."""
        return self.query_recall(
            query=query,
            n_results=n_results,
            include_knowledge=include_knowledge,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            include_expired=include_expired,
            as_of=as_of,
        )

    def _recall_internal(
        self,
        *,
        query: str,
        n_results: int,
        include_knowledge: bool,
        content_types: list[str] | None,
        tags: list[str] | None,
        after: str | None,
        before: str | None,
        include_expired: bool,
        as_of: str | None,
        resolved_scope: ResolvedScopeEnvelope,
    ) -> RecallResult:
        """Retrieve relevant memories with canonical scope filtering."""
        self._vector_store.reload_if_stale()
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)

        try:
            recall_result = self._query_service.recall(
                RecallQuery(
                    query=query,
                    n_results=n_results,
                    include_knowledge=include_knowledge,
                    content_types=content_types,
                    tags=tags,
                    after=after,
                    before=before,
                    include_expired=include_expired,
                    as_of=as_of,
                ),
                scope_filter=scope_filter,
            )
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during recall: %s", e)
            self._record_recall_signal(fallback=True)
            return RecallResult(
                message=f"Embedding backend unreachable: {e}",
            )

        records = recall_result.records
        claims = recall_result.claims
        logger.info(
            "Recall query='%s' returned %d episodes, %d knowledge entries, %d records, %d claims",
            query[:80],
            len(recall_result.episodes),
            len(recall_result.knowledge),
            len(records),
            len(claims),
        )

        if (
            len(recall_result.episodes) == 0
            and len(recall_result.knowledge) == 0
            and len(recall_result.records) == 0
            and len(recall_result.claims) == 0
        ):
            self._record_recall_signal(miss=True)

        from consolidation_memory.plugins import get_plugin_manager
        get_plugin_manager().fire("on_recall", query=query, result=recall_result)

        self._maybe_auto_consolidate(trigger_source="recall")
        return recall_result

    def search(
        self,
        query: str | None = None,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 20,
    ) -> SearchResult:
        """Keyword/metadata search using backward-compatible default scope."""
        return self.query_search(
            query=query,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            limit=limit,
        )

    def _search_internal(
        self,
        *,
        query: str | None,
        content_types: list[str] | None,
        tags: list[str] | None,
        after: str | None,
        before: str | None,
        limit: int,
        resolved_scope: ResolvedScopeEnvelope,
    ) -> SearchResult:
        """Keyword/metadata search over episodes. No embedding backend required.

        Unlike recall(), this does plain text LIKE matching in SQLite — useful
        when the embedding backend is down, or for exact substring matching.

        Args:
            query: Text substring to search for (case-insensitive).
            content_types: Filter to specific content types.
            tags: Filter to episodes with at least one matching tag.
            after: Only episodes created after this ISO date.
            before: Only episodes created before this ISO date.
            limit: Max results (default 20).

        Returns:
            SearchResult with matching episodes.
        """
        self._vector_store.reload_if_stale()
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)

        result = self._query_service.search(
            EpisodeSearchQuery(
                query=query,
                content_types=content_types,
                tags=tags,
                after=after,
                before=before,
                limit=limit,
            ),
            scope_filter=scope_filter,
        )
        logger.info(
            "Search query=%r returned %d results",
            query,
            len(result.episodes),
        )
        return result

    def query_browse_claims(
        self,
        claim_type: str | None = None,
        as_of: str | None = None,
        limit: int = 50,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> ClaimBrowseResult:
        """Canonical claim-browse entrypoint for all adapters."""
        operation_context = self.build_operation_context(scope)
        scope_filter = _resolved_scope_to_query_filter(operation_context.scope)
        result = self._query_service.browse_claims(
            ClaimBrowseQuery(
                claim_type=claim_type,
                as_of=as_of,
                limit=limit,
            ),
            scope_filter=scope_filter,
        )
        logger.info(
            "Browse claims claim_type=%r as_of=%r returned %d results",
            claim_type,
            as_of,
            len(result.claims),
        )
        return result

    def browse_claims(
        self,
        claim_type: str | None = None,
        as_of: str | None = None,
        limit: int = 50,
    ) -> ClaimBrowseResult:
        """Backward-compatible wrapper for canonical claim browse."""
        return self.query_browse_claims(
            claim_type=claim_type,
            as_of=as_of,
            limit=limit,
        )

    def query_search_claims(
        self,
        query: str,
        claim_type: str | None = None,
        as_of: str | None = None,
        limit: int = 50,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> ClaimSearchResult:
        """Canonical claim-search entrypoint for all adapters."""
        operation_context = self.build_operation_context(scope)
        scope_filter = _resolved_scope_to_query_filter(operation_context.scope)
        result = self._query_service.search_claims(
            ClaimSearchQuery(
                query=query,
                claim_type=claim_type,
                as_of=as_of,
                limit=limit,
            ),
            scope_filter=scope_filter,
        )
        logger.info(
            "Search claims query=%r claim_type=%r as_of=%r returned %d matches",
            query.strip(),
            claim_type,
            as_of,
            len(result.claims),
        )
        return result

    def search_claims(
        self,
        query: str,
        claim_type: str | None = None,
        as_of: str | None = None,
        limit: int = 50,
    ) -> ClaimSearchResult:
        """Backward-compatible wrapper for canonical claim search."""
        return self.query_search_claims(
            query=query,
            claim_type=claim_type,
            as_of=as_of,
            limit=limit,
        )

    def query_detect_drift(
        self,
        base_ref: str | None = None,
        repo_path: str | None = None,
    ) -> DriftOutput:
        """Canonical drift detection/challenge entrypoint for all adapters."""
        self._ensure_open()
        resolved_scope = self.resolve_scope()
        result = self._query_service.detect_drift(
            DriftQuery(
                base_ref=base_ref,
                repo_path=repo_path,
                scope={
                    "namespace_slug": resolved_scope.namespace.slug,
                    "project_slug": resolved_scope.project.slug,
                },
            )
        )
        logger.info(
            "Detect drift base_ref=%r repo_path=%r impacted=%d challenged=%d",
            base_ref,
            repo_path,
            len(result["impacted_claim_ids"]),
            len(result["challenged_claim_ids"]),
        )
        return result

    def detect_drift(
        self,
        base_ref: str | None = None,
        repo_path: str | None = None,
    ) -> DriftOutput:
        """Backward-compatible wrapper for canonical drift detection."""
        return self.query_detect_drift(base_ref=base_ref, repo_path=repo_path)

    def status(self) -> StatusResult:
        """Get memory system statistics.

        Returns:
            StatusResult with counts, backend info, health, and last consolidation.
        """
        self._ensure_open()
        from consolidation_memory.database import (
            count_active_challenged_claims,
            get_claim_trust_stats,
            get_consolidation_scheduler_state,
            get_recently_contradicted_topic_ids,
            get_stats,
            get_last_consolidation_run,
            get_recent_consolidation_runs,
        )
        from consolidation_memory.config import get_config
        from consolidation_memory.knowledge_consistency import build_knowledge_consistency_report

        cfg = get_config()
        stats = get_stats()
        last_run = get_last_consolidation_run()

        db_size_mb = 0.0
        if cfg.DB_PATH.exists():
            db_size_mb = round(cfg.DB_PATH.stat().st_size / (1024 * 1024), 2)

        knowledge_consistency = build_knowledge_consistency_report()
        health = self._compute_health(
            last_run,
            cfg.CONSOLIDATION_INTERVAL_HOURS,
            cfg.FAISS_COMPACTION_THRESHOLD,
            knowledge_consistency,
        )

        from consolidation_memory.database import get_consolidation_metrics
        metrics = get_consolidation_metrics(limit=10)

        # Compute aggregate quality stats from recent metrics
        quality_summary = {}
        if metrics:
            total_succeeded = sum(m.get("clusters_succeeded", 0) for m in metrics)
            total_failed = sum(m.get("clusters_failed", 0) for m in metrics)
            total_clusters = total_succeeded + total_failed
            confidences = [m["avg_confidence"] for m in metrics if m.get("avg_confidence", 0) > 0]
            quality_summary = {
                "runs_analyzed": len(metrics),
                "total_clusters_processed": total_clusters,
                "success_rate": round(total_succeeded / total_clusters, 3) if total_clusters else 0,
                "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
                "total_api_calls": sum(m.get("api_calls", 0) for m in metrics),
                "total_episodes_processed": sum(m.get("episodes_processed", 0) for m in metrics),
            }

        recent_runs = get_recent_consolidation_runs(limit=5)
        recent_activity = []
        for run in recent_runs:
            summary = {
                "timestamp": run.get("completed_at") or run.get("started_at"),
                "status": run.get("status", "unknown"),
                "episodes_processed": run.get("episodes_processed", 0),
                "clusters_formed": run.get("clusters_formed", 0),
                "topics_created": run.get("topics_created", 0),
                "topics_updated": run.get("topics_updated", 0),
                "episodes_pruned": run.get("episodes_pruned", 0),
            }
            recent_activity.append(summary)

        utility_state = self._compute_consolidation_utility()
        scheduler_state = get_consolidation_scheduler_state()
        now_utc = datetime.now(timezone.utc)
        next_due_at = scheduler_state.get("next_due_at")
        is_due = True
        seconds_until_due: float | None = None
        if isinstance(next_due_at, str) and next_due_at.strip():
            try:
                due_dt = parse_datetime(next_due_at)
                seconds_until_due = round((due_dt - now_utc).total_seconds(), 3)
                is_due = seconds_until_due <= 0
            except (ValueError, TypeError):
                is_due = True
                seconds_until_due = None

        lease_owner = scheduler_state.get("lease_owner")
        lease_expires_at = scheduler_state.get("lease_expires_at")
        lease_stale = False
        if isinstance(lease_owner, str) and lease_owner and isinstance(lease_expires_at, str):
            try:
                lease_stale = parse_datetime(lease_expires_at) < now_utc
            except (ValueError, TypeError):
                lease_stale = False

        score_value = utility_state.get("score")
        utility_score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
        raw_signals_obj = utility_state.get("raw_signals")
        raw_signals = raw_signals_obj if isinstance(raw_signals_obj, dict) else None
        should_run_now, trigger_reason = self._should_trigger_scheduler_run(
            scheduler_state=scheduler_state,
            utility_score=utility_score,
            raw_signals=raw_signals,
            now_utc=now_utc,
        )

        backlog_force_threshold, challenged_force_threshold = _compute_force_thresholds_runtime(
            max_episodes_per_run=cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN
        )
        utility_scheduler = {
            "enabled": bool(cfg.CONSOLIDATION_AUTO_RUN),
            "threshold": cfg.CONSOLIDATION_UTILITY_THRESHOLD,
            "weights": dict(cfg.CONSOLIDATION_UTILITY_WEIGHTS),
            "score": utility_state["score"],
            "normalized_signals": utility_state["normalized_signals"],
            "weighted_components": utility_state["weighted_components"],
            "raw_signals": utility_state["raw_signals"],
            "next_due_at": scheduler_state.get("next_due_at"),
            "last_status": scheduler_state.get("last_status"),
            "last_trigger": scheduler_state.get("last_trigger"),
            "last_error": scheduler_state.get("last_error"),
            "last_run_started_at": scheduler_state.get("last_run_started_at"),
            "last_run_completed_at": scheduler_state.get("last_run_completed_at"),
            "updated_at": scheduler_state.get("updated_at"),
            "lease_owner": lease_owner,
            "lease_expires_at": lease_expires_at,
            "lease_stale": lease_stale,
            "is_due": is_due,
            "seconds_until_due": seconds_until_due,
            "run_decision": {
                "should_run": should_run_now,
                "reason": trigger_reason,
            },
            "force_thresholds": {
                "unconsolidated_backlog": backlog_force_threshold,
                "challenged_claim_backlog": challenged_force_threshold,
            },
        }
        scaling = {
            "index_type": self._vector_store.index_type,
            "vector_count": self._vector_store.size,
            "ivf_upgrade_threshold": cfg.FAISS_IVF_UPGRADE_THRESHOLD,
            "platform_review_threshold": cfg.FAISS_PLATFORM_REVIEW_THRESHOLD,
            "needs_platform_scaling_pass": (
                self._vector_store.size >= cfg.FAISS_PLATFORM_REVIEW_THRESHOLD
            ),
        }
        trust_counts = get_claim_trust_stats()
        evolving_topic_ids = get_recently_contradicted_topic_ids(
            days=cfg.EVOLVING_TOPIC_LOOKBACK_DAYS
        )
        challenged_backlog = count_active_challenged_claims(as_of=now_utc.isoformat())
        source_coverage = float(trust_counts["source_coverage_ratio"])
        if int(trust_counts["currently_valid_claims"]) == 0:
            trust_posture = "bootstrapping"
        elif challenged_backlog > 0 or evolving_topic_ids:
            trust_posture = "drift_watch"
        elif source_coverage < 0.7:
            trust_posture = "needs_provenance"
        else:
            trust_posture = "trusted_reuse_ready"
        trust_profile = {
            "role": "trust_layer_for_coding_agents",
            "primary_unit": "claims",
            "evidence_model": {
                "reusable_unit": "claims",
                "evidence_units": [
                    "episodes",
                    "knowledge_records",
                    "claim_sources",
                    "episode_anchors",
                    "claim_events",
                ],
                "retrieval_preference": (
                    "Prefer reusable claims with provenance and uncertainty "
                    "signals; use episodes as raw evidence."
                ),
            },
            "design_priorities": [
                "reliability_over_breadth",
                "claim_first_memory",
                "provenance_required_for_reuse",
                "drift_aware_retrieval",
                "scope_safe_sharing",
            ],
            "posture": trust_posture,
            "current_state": {
                **trust_counts,
                "recently_contradicted_topics": len(evolving_topic_ids),
                "challenged_claim_backlog": challenged_backlog,
                "evolving_topic_window_days": cfg.EVOLVING_TOPIC_LOOKBACK_DAYS,
            },
        }

        return StatusResult(
            episodic_buffer=stats["episodic_buffer"],
            knowledge_base=stats["knowledge_base"],
            last_consolidation=last_run,
            embedding_backend=cfg.EMBEDDING_BACKEND,
            embedding_model=cfg.EMBEDDING_MODEL_NAME,
            faiss_index_size=self._vector_store.size,
            faiss_tombstones=self._vector_store.tombstone_count,
            db_size_mb=db_size_mb,
            version=__version__,
            health=health,
            consolidation_metrics=metrics,
            consolidation_quality=quality_summary,
            recent_activity=recent_activity,
            utility_scheduler=utility_scheduler,
            knowledge_consistency=knowledge_consistency,
            scaling=scaling,
            trust_profile=trust_profile,
        )

    def forget(
        self,
        episode_id: str,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> ForgetResult:
        """Soft-delete an episode.

        Args:
            episode_id: UUID of the episode to forget.

        Returns:
            ForgetResult with status 'forgotten' or 'not_found'.
        """
        from consolidation_memory.database import restore_soft_deleted_episode, soft_delete_episode

        self._ensure_open()
        resolved_scope = self.resolve_scope(scope)
        denied_message = _write_denied_message(resolved_scope)
        if denied_message is not None:
            return ForgetResult(status="write_denied", id=episode_id, message=denied_message)
        self._vector_store.reload_if_stale()
        mutation_filter = _resolved_scope_to_mutation_filter(resolved_scope)
        deleted = soft_delete_episode(episode_id, scope=mutation_filter)
        if deleted:
            try:
                self._vector_store.remove(episode_id)
            except Exception:
                restored = False
                try:
                    restored = restore_soft_deleted_episode(episode_id, scope=mutation_filter)
                except Exception:
                    logger.exception(
                        "Forget rollback failed while restoring DB row for %s",
                        episode_id,
                    )
                    raise RuntimeError(
                        f"Forget failed for {episode_id} and DB restore also failed."
                    )
                if not restored:
                    logger.error(
                        "Forget failed for %s and DB restore did not find row",
                        episode_id,
                    )
                    raise RuntimeError(
                        f"Forget failed for {episode_id}; unable to confirm DB rollback."
                    )
                logger.error(
                    "Forget vector tombstone failed for %s; restored SQLite deletion",
                    episode_id,
                )
                raise
            logger.info("Forgot episode %s", episode_id)

            from consolidation_memory.plugins import get_plugin_manager
            get_plugin_manager().fire("on_forget", episode_id=episode_id)

            return ForgetResult(status="forgotten", id=episode_id)
        else:
            logger.warning("Episode %s not found for deletion", episode_id)
            return ForgetResult(status="not_found", id=episode_id)

    def export(
        self,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> ExportResult:
        """Export all episodes and knowledge to a JSON snapshot.

        Returns:
            ExportResult with status, file path, and counts.
        """
        from consolidation_memory.config import get_config
        from consolidation_memory.database import (
            get_all_active_records,
            get_all_claim_edges,
            get_all_claim_events,
            get_all_claim_sources,
            get_all_claims,
            get_all_episode_anchors,
            get_all_episodes,
            get_all_knowledge_topics,
        )
        from consolidation_memory.knowledge_paths import resolve_topic_path
        from consolidation_memory.query_semantics import filter_claims_for_scope

        self._ensure_open()
        cfg = get_config()
        cfg.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        resolved_scope = self.resolve_scope(scope)
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)

        episodes = get_all_episodes(include_deleted=False, scope=scope_filter)

        topics = get_all_knowledge_topics(scope=scope_filter)
        knowledge = []
        for topic in topics:
            content = ""
            try:
                filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, topic, prefer_existing=True)
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
            except ValueError:
                logger.warning(
                    "Skipped invalid knowledge export path for topic %s (%s)",
                    topic.get("id"),
                    topic.get("filename"),
                )
            knowledge.append({**topic, "file_content": content})

        records = get_all_active_records(include_expired=True, scope=scope_filter)
        claims = filter_claims_for_scope(get_all_claims(), scope_filter)
        claim_ids = [str(claim["id"]) for claim in claims if claim.get("id")]
        claim_edges = get_all_claim_edges(claim_ids=claim_ids)
        claim_sources = get_all_claim_sources(claim_ids=claim_ids)
        claim_events = get_all_claim_events(claim_ids=claim_ids)
        episode_ids = [str(ep["id"]) for ep in episodes if ep.get("id")]
        episode_anchors = get_all_episode_anchors(episode_ids=episode_ids)

        snapshot = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.2",
            "episodes": episodes,
            "knowledge_topics": knowledge,
            "knowledge_records": records,
            "claims": claims,
            "claim_edges": claim_edges,
            "claim_sources": claim_sources,
            "claim_events": claim_events,
            "episode_anchors": episode_anchors,
            "stats": {
                "episode_count": len(episodes),
                "knowledge_count": len(knowledge),
                "record_count": len(records),
                "claim_count": len(claims),
                "claim_edge_count": len(claim_edges),
                "claim_source_count": len(claim_sources),
                "claim_event_count": len(claim_events),
                "episode_anchor_count": len(episode_anchors),
            },
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_path = cfg.BACKUP_DIR / f"memory_export_{timestamp}.json"
        export_path.write_text(
            json.dumps(snapshot, indent=2, default=str),
            encoding="utf-8",
        )

        # Prune old exports
        existing = sorted(cfg.BACKUP_DIR.glob("memory_export_*.json"), reverse=True)
        for old in existing[cfg.MAX_BACKUPS:]:
            old.unlink()

        logger.info(
            "Exported %d episodes + %d topics + %d claims to %s",
            len(episodes),
            len(knowledge),
            len(claims),
            export_path,
        )

        return ExportResult(
            status="exported",
            path=str(export_path),
            episodes=len(episodes),
            knowledge_topics=len(knowledge),
            claims=len(claims),
            claim_edges=len(claim_edges),
            claim_sources=len(claim_sources),
            claim_events=len(claim_events),
            episode_anchors=len(episode_anchors),
        )

    def correct(
        self,
        topic_filename: str,
        correction: str,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> CorrectResult:
        """Correct a knowledge document with new information.

        Args:
            topic_filename: Filename of the knowledge topic (e.g. 'vr_setup.md').
            correction: Description of what needs correcting and the correct info.

        Returns:
            CorrectResult with status and updated metadata.
        """
        from consolidation_memory.config import get_config
        from consolidation_memory.database import (
            expire_record,
            get_connection,
            get_knowledge_topic,
            get_records_by_topic,
            insert_knowledge_records,
            upsert_knowledge_topic,
        )
        from consolidation_memory.consolidation.engine import _version_knowledge_file
        from consolidation_memory.consolidation.prompting import (
            _embedding_text_for_record,
            _normalize_output,
            _parse_frontmatter,
            _sanitize_for_prompt,
        )
        from consolidation_memory.backends import get_llm_backend
        from consolidation_memory.knowledge_paths import resolve_topic_path

        self._ensure_open()
        cfg = get_config()
        resolved_scope = self.resolve_scope(scope)
        denied_message = _write_denied_message(resolved_scope)
        if denied_message is not None:
            return CorrectResult(status="write_denied", filename=topic_filename, message=denied_message)
        mutation_filter = _resolved_scope_to_mutation_filter(resolved_scope)

        # Validate filename doesn't escape KNOWLEDGE_DIR (path traversal)
        existing_topic = get_knowledge_topic(topic_filename, scope=mutation_filter)
        if existing_topic is None:
            return CorrectResult(status="not_found", filename=topic_filename)
        try:
            # Prefer an existing legacy logical filename when present.
            filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, existing_topic, prefer_existing=True)
        except ValueError:
            return CorrectResult(
                status="error",
                filename=topic_filename,
                message="Invalid filename: path traversal detected.",
            )
        if not filepath.exists():
            return CorrectResult(status="not_found", filename=topic_filename)

        llm = get_llm_backend()
        if llm is None:
            return CorrectResult(status="error", message="LLM backend is disabled")

        existing_content = filepath.read_text(encoding="utf-8")
        system_prompt = (
            "You are a precise knowledge editor. Apply corrections to existing documents "
            "while preserving all other information. Output the complete corrected document."
        )
        user_prompt = (
            f"Apply this correction to the existing knowledge document:\n\n"
            f"CORRECTION:\n{_sanitize_for_prompt(correction)}\n\n"
            f"EXISTING DOCUMENT:\n{_sanitize_for_prompt(existing_content)}\n\n"
            f"Output the complete corrected document with updated frontmatter "
            f"(title, summary, tags, confidence).\n"
            f"Do NOT wrap in code fences. Output raw markdown starting with --- frontmatter."
        )

        try:
            if self._llm_executor is None:
                return CorrectResult(status="error", message="Client is closed.")
            future = self._llm_executor.submit(llm.generate, system_prompt, user_prompt)
            raw = future.result(timeout=cfg.LLM_CORRECTION_TIMEOUT)
            corrected = _normalize_output(raw)
        except FuturesTimeoutError:
            return CorrectResult(
                status="error",
                message=f"LLM generation timed out after {cfg.LLM_CORRECTION_TIMEOUT}s. "
                        "Try again or increase llm.correction_timeout in config.",
            )
        except Exception as e:
            return CorrectResult(status="error", message=str(e))

        # Validate LLM output before writing — reject empty or structureless output
        if not corrected or not corrected.strip():
            return CorrectResult(
                status="error",
                message="LLM returned empty output; original document preserved.",
            )
        parsed_check = _parse_frontmatter(corrected)
        if not parsed_check["meta"].get("title"):
            return CorrectResult(
                status="error",
                message="LLM output missing frontmatter structure; original document preserved.",
            )

        parsed = _parse_frontmatter(corrected)
        meta = parsed["meta"]
        body = parsed.get("body", "")

        source_eps = parse_json_list(existing_topic.get("source_episodes"))
        existing_confidence = float(existing_topic["confidence"])
        topic_scope = {
            "namespace_slug": existing_topic.get("namespace_slug"),
            "namespace_sharing_mode": existing_topic.get("namespace_sharing_mode"),
            "app_client_name": existing_topic.get("app_client_name"),
            "app_client_type": existing_topic.get("app_client_type"),
            "app_client_provider": existing_topic.get("app_client_provider"),
            "app_client_external_key": existing_topic.get("app_client_external_key"),
            "agent_name": existing_topic.get("agent_name"),
            "agent_external_key": existing_topic.get("agent_external_key"),
            "session_external_key": existing_topic.get("session_external_key"),
            "session_kind": existing_topic.get("session_kind"),
            "project_slug": existing_topic.get("project_slug"),
            "project_display_name": existing_topic.get("project_display_name"),
            "project_root_uri": existing_topic.get("project_root_uri"),
            "project_repo_remote": existing_topic.get("project_repo_remote"),
            "project_default_branch": existing_topic.get("project_default_branch"),
        }

        try:
            confidence = float(meta.get("confidence", existing_confidence))
        except (TypeError, ValueError):
            confidence = existing_confidence
        confidence = max(0.0, min(1.0, confidence))

        parsed_records = parse_markdown_records(body)
        if not parsed_records:
            fallback_info = (
                str(meta.get("summary", "")).strip()
                or body.strip()
                or correction.strip()
                or "Knowledge document corrected."
            )
            fallback_subject = str(meta.get("title", topic_filename)).strip() or topic_filename
            parsed_records = [{
                "type": "fact",
                "subject": fallback_subject,
                "info": fallback_info,
            }]

        record_rows = []
        now_ts = datetime.now(timezone.utc).isoformat()
        for rec in parsed_records:
            record_rows.append({
                "record_type": rec.get("type", "fact"),
                "content": rec,
                "embedding_text": _embedding_text_for_record(rec),
                "confidence": confidence,
                "valid_from": now_ts,
            })

        prepared_path: str | None = None
        replaced_file = False
        try:
            prepared_path = _write_temp_text(filepath, corrected)
            with get_connection():
                topic_id = upsert_knowledge_topic(
                    filename=topic_filename,
                    title=str(meta.get("title", existing_topic["title"])),
                    summary=str(meta.get("summary", existing_topic["summary"])),
                    source_episodes=source_eps,
                    fact_count=len(record_rows),
                    confidence=confidence,
                    scope=topic_scope,
                )

                for old in get_records_by_topic(topic_id, include_expired=False):
                    expire_record(old["id"], valid_until=now_ts)

                if record_rows:
                    insert_knowledge_records(
                        topic_id,
                        record_rows,
                        source_episodes=source_eps,
                        scope=topic_scope,
                    )

                _version_knowledge_file(filepath)
                os.replace(prepared_path, filepath)
                replaced_file = True
        except Exception as e:
            restore_error: Exception | None = None
            if replaced_file:
                try:
                    restored_path = _write_temp_text(filepath, existing_content)
                    try:
                        os.replace(restored_path, filepath)
                    finally:
                        if os.path.exists(restored_path):
                            os.unlink(restored_path)
                except Exception as restore_exc:  # pragma: no cover - rare fs failure
                    restore_error = restore_exc
                    logger.exception(
                        "Failed to restore knowledge file after correction rollback: %s",
                        filepath,
                    )
            message = str(e)
            if restore_error is not None:
                message = (
                    f"{message}. Database changes were rolled back, but restoring the original "
                    f"knowledge file failed: {restore_error}"
                )
            return CorrectResult(status="error", filename=topic_filename, message=message)
        finally:
            if prepared_path is not None and os.path.exists(prepared_path):
                os.unlink(prepared_path)
        from consolidation_memory import claim_cache as _cc, topic_cache as _tc, record_cache as _rc
        _cc.invalidate()
        _tc.invalidate()
        _rc.invalidate()

        logger.info(
            "Corrected knowledge topic: %s (%d records)",
            topic_filename,
            len(record_rows),
        )

        return CorrectResult(
            status="corrected",
            filename=topic_filename,
            title=meta.get("title", ""),
        )

    def browse(
        self,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> BrowseResult:
        """List all knowledge topics with summaries and metadata.

        Returns:
            BrowseResult with topic list and count.
        """
        from consolidation_memory.database import (
            get_all_active_records,
            get_all_knowledge_topics,
        )
        from consolidation_memory.config import get_config
        from consolidation_memory.knowledge_paths import resolve_topic_path

        self._ensure_open()
        cfg = get_config()
        resolved_scope = self.resolve_scope(scope)
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)
        topics = get_all_knowledge_topics(scope=scope_filter)
        records = get_all_active_records(include_expired=False, scope=scope_filter)

        # Group record counts by topic_id
        records_by_topic: dict[str, dict[str, int]] = {}
        for rec in records:
            tid = rec["topic_id"]
            if tid not in records_by_topic:
                records_by_topic[tid] = {"fact": 0, "solution": 0, "preference": 0, "procedure": 0}
            rt = rec.get("record_type", "fact")
            if rt in records_by_topic[tid]:
                records_by_topic[tid][rt] += 1

        result_topics = []
        for topic in topics:
            file_exists = False
            file_path = ""
            try:
                filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, topic, prefer_existing=True)
                file_exists = filepath.exists()
                file_path = str(filepath)
            except ValueError:
                logger.warning(
                    "Skipped invalid knowledge path for topic %s (%s)",
                    topic.get("id"),
                    topic.get("filename"),
                )
            rec_counts = records_by_topic.get(topic["id"], {})
            total_records = sum(rec_counts.values())
            result_topics.append({
                "filename": topic["filename"],
                "title": topic["title"],
                "summary": topic["summary"],
                "confidence": topic.get("confidence", 0),
                "updated_at": topic.get("updated_at", ""),
                "record_counts": rec_counts,
                "total_records": total_records,
                "file_exists": file_exists,
                "file_path": file_path,
            })

        return BrowseResult(topics=result_topics, total=len(result_topics))

    def read_topic(
        self,
        filename: str,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> TopicDetailResult:
        """Read the full rendered markdown content of a knowledge file.

        Args:
            filename: The filename of the knowledge topic (e.g. 'python_setup.md').

        Returns:
            TopicDetailResult with the markdown content.
        """
        from consolidation_memory.config import get_config
        from consolidation_memory.database import get_knowledge_topic
        from consolidation_memory.knowledge_paths import resolve_topic_path

        self._ensure_open()
        cfg = get_config()
        resolved_scope = self.resolve_scope(scope)
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)
        topic_row = get_knowledge_topic(filename, scope=scope_filter)
        if topic_row is None:
            return TopicDetailResult(status="not_found", filename=filename)

        try:
            # Prefer an existing legacy logical filename when present.
            filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, topic_row, prefer_existing=True)
        except ValueError:
            return TopicDetailResult(
                status="error",
                filename=filename,
                message="Invalid filename: path traversal detected.",
            )
        if not filepath.exists():
            return TopicDetailResult(status="not_found", filename=filename)

        content = filepath.read_text(encoding="utf-8")
        return TopicDetailResult(status="ok", filename=filename, content=content)

    def timeline(
        self,
        topic: str,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> TimelineResult:
        """Show how understanding of a topic has changed over time.

        Retrieves all records (including expired) matching the topic,
        sorted chronologically, with supersession links where a record
        was replaced by a newer one.

        Args:
            topic: Natural language topic to query (e.g., 'frontend framework preference').

        Returns:
            TimelineResult with chronologically sorted entries.
        """
        from consolidation_memory.database import get_all_active_records
        from consolidation_memory.backends import encode_query
        import numpy as np

        self._ensure_open()
        self._vector_store.reload_if_stale()

        resolved_scope = self.resolve_scope(scope)
        scope_filter = _resolved_scope_to_query_filter(resolved_scope)

        # Get all records including expired
        all_records = get_all_active_records(include_expired=True, scope=scope_filter)
        if not all_records:
            return TimelineResult(
                query=topic, message="No knowledge records found."
            )

        # Embed the query and find relevant records by semantic similarity
        try:
            query_vec = encode_query(topic)
        except ConnectionError as e:
            return TimelineResult(
                query=topic, message=f"Embedding backend unreachable: {e}"
            )

        # Embed record texts and compute similarity
        from consolidation_memory.backends import encode_documents

        embedding_texts = [r.get("embedding_text", "") or "" for r in all_records]
        # Filter out records with no embedding text
        valid_indices = [i for i, t in enumerate(embedding_texts) if t.strip()]
        if not valid_indices:
            return TimelineResult(
                query=topic, message="No records with embedding text found."
            )

        valid_texts = [embedding_texts[i] for i in valid_indices]
        try:
            record_vecs = encode_documents(valid_texts)
        except ConnectionError as e:
            return TimelineResult(
                query=topic, message=f"Embedding backend unreachable: {e}"
            )

        # Compute cosine similarities
        query_norm = query_vec.reshape(1, -1).astype(np.float32)
        sims = (query_norm @ record_vecs.T).flatten()

        # Filter to relevant records (similarity >= 0.3)
        threshold = 0.3
        matching = []
        for idx, sim in zip(valid_indices, sims):
            if sim >= threshold:
                matching.append((idx, float(sim)))

        if not matching:
            return TimelineResult(
                query=topic, message="No records found matching that topic."
            )

        # Sort by matching record's creation date
        matching.sort(key=lambda x: all_records[x[0]].get("created_at", ""))

        # Build reverse lookup: valid_indices position -> vec position (O(1) vs O(n))
        valid_idx_to_vec_pos = {idx: pos for pos, idx in enumerate(valid_indices)}

        # Build timeline entries, detect supersession
        entries = []
        for rec_idx, sim in matching:
            rec = all_records[rec_idx]
            is_expired = rec.get("valid_until") is not None and rec["valid_until"] != ""

            entry = {
                "date": rec.get("valid_from") or rec.get("created_at", ""),
                "record_type": rec.get("record_type", ""),
                "content": rec.get("content", {}),
                "embedding_text": rec.get("embedding_text", ""),
                "confidence": rec.get("confidence", 0),
                "similarity": round(sim, 4),
                "status": "superseded" if is_expired else "active",
                "valid_from": rec.get("valid_from"),
                "valid_until": rec.get("valid_until"),
                "topic_title": rec.get("topic_title", ""),
                "topic_filename": rec.get("topic_filename", ""),
            }

            # Try to find the superseding record
            if is_expired:
                expired_at = rec["valid_until"]
                expired_text = rec.get("embedding_text", "")
                if expired_text:
                    vec_pos = valid_idx_to_vec_pos.get(rec_idx)
                    expired_vec = record_vecs[vec_pos] if vec_pos is not None else None
                    if expired_vec is not None:
                        # Find the most similar active record created around the same time
                        best_successor = None
                        best_sim = 0.0
                        for other_idx, _ in matching:
                            other = all_records[other_idx]
                            if other.get("valid_until"):
                                continue  # skip other expired records
                            other_created = other.get("created_at", "")
                            if other_created >= (expired_at or ""):
                                other_vec_pos = valid_idx_to_vec_pos.get(other_idx)
                                if other_vec_pos is not None:
                                    pair_sim = float(
                                        expired_vec.reshape(1, -1) @ record_vecs[other_vec_pos].reshape(-1, 1)
                                    )
                                    if pair_sim > best_sim and pair_sim >= 0.5:
                                        best_sim = pair_sim
                                        best_successor = other.get("embedding_text", "")
                        if best_successor:
                            entry["superseded_by"] = best_successor

            entries.append(entry)

        return TimelineResult(
            query=topic,
            entries=entries,
            total=len(entries),
        )

    def contradictions(self, topic: str | None = None) -> ContradictionResult:
        """List contradictions from the audit log, optionally filtered by topic.

        Args:
            topic: Optional topic filename or title to filter by.

        Returns:
            ContradictionResult with logged contradictions.
        """
        from consolidation_memory.database import (
            get_contradictions as db_get_contradictions,
            get_all_knowledge_topics,
            topic_storage_filename,
        )

        self._ensure_open()
        topic_id = None
        if topic:
            topics = get_all_knowledge_topics()
            for t in topics:
                logical_filename = str(t.get("filename", ""))
                storage_name = topic_storage_filename(t)
                if topic in (logical_filename, storage_name, t["title"]):
                    topic_id = t["id"]
                    break
            else:
                # Topic specified but not found — return empty
                return ContradictionResult(
                    contradictions=[], total=0, topic=topic,
                )

        rows = db_get_contradictions(topic_id=topic_id)
        return ContradictionResult(
            contradictions=rows,
            total=len(rows),
            topic=topic,
        )

    def consolidation_log(self, last_n: int = 5) -> ConsolidationLogResult:
        """Show recent consolidation activity as a human-readable changelog.

        Returns a summary of recent consolidation runs including topics
        created/updated, contradictions detected, and episodes pruned.

        Args:
            last_n: Number of recent runs to include (default 5, max 20).

        Returns:
            ConsolidationLogResult with formatted changelog entries.
        """
        from consolidation_memory.database import (
            get_recent_consolidation_runs,
            get_contradictions as db_get_contradictions,
        )

        self._ensure_open()
        last_n = min(max(1, last_n), 20)
        runs = get_recent_consolidation_runs(limit=last_n)

        if not runs:
            return ConsolidationLogResult(
                entries=[], total=0,
                message="No consolidation runs yet.",
            )

        # Heuristic: each consolidation run typically produces up to ~50
        # contradictions, so we over-fetch to avoid missing any.
        _CONTRADICTIONS_PER_RUN_ESTIMATE = 50
        contradictions = db_get_contradictions(limit=last_n * _CONTRADICTIONS_PER_RUN_ESTIMATE)

        entries = []
        for run in runs:
            started = run.get("started_at", "")
            status = run.get("status", "unknown")

            # Count contradictions detected during this run's time window
            run_contradictions = 0
            completed = run.get("completed_at", "")
            if started and completed:
                run_contradictions = sum(
                    1 for c in contradictions
                    if started <= c.get("detected_at", "") <= completed
                )

            # Build human-readable summary
            parts = []
            tc = run.get("topics_created", 0)
            tu = run.get("topics_updated", 0)
            ep = run.get("episodes_processed", 0)
            pr = run.get("episodes_pruned", 0)

            if tc:
                parts.append(f"created {tc} topic{'s' if tc != 1 else ''}")
            if tu:
                parts.append(f"updated {tu} topic{'s' if tu != 1 else ''}")
            if run_contradictions:
                parts.append(
                    f"detected {run_contradictions} "
                    f"contradiction{'s' if run_contradictions != 1 else ''}"
                )
            if pr:
                parts.append(f"pruned {pr} episode{'s' if pr != 1 else ''}")
            if ep and not parts:
                parts.append(f"processed {ep} episode{'s' if ep != 1 else ''}")

            summary = ", ".join(parts) if parts else "no changes"
            if status == RUN_STATUS_FAILED:
                summary = f"FAILED — {run.get('error_message') or 'unknown error'}"
            elif status == RUN_STATUS_RUNNING:
                summary = "In progress"
            else:
                summary = summary[0].upper() + summary[1:] if summary else summary

            entries.append({
                "run_id": run.get("id", ""),
                "started_at": started,
                "completed_at": completed,
                "status": status,
                "summary": summary,
                "topics_created": tc,
                "topics_updated": tu,
                "episodes_processed": ep,
                "episodes_pruned": pr,
                "contradictions_detected": run_contradictions,
            })

        return ConsolidationLogResult(
            entries=entries,
            total=len(entries),
        )

    def decay_report(self) -> DecayReportResult:
        """Show what would be forgotten if pruning ran right now.

        Reports prunable episodes, low-confidence records, and protected
        episode counts. Does NOT actually delete anything.

        Returns:
            DecayReportResult with counts and details.
        """
        from consolidation_memory.database import (
            get_prunable_episodes, get_low_confidence_records,
        )
        from consolidation_memory.config import get_config

        self._ensure_open()
        cfg = get_config()

        prunable = get_prunable_episodes(days=cfg.CONSOLIDATION_PRUNE_AFTER_DAYS)
        low_conf = get_low_confidence_records(threshold=0.5)

        # Count protected episodes
        from consolidation_memory.database import get_connection
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM episodes WHERE protected = 1 AND deleted = 0"
            ).fetchone()
        protected_count = row["c"] if row else 0

        details = {
            "prune_after_days": cfg.CONSOLIDATION_PRUNE_AFTER_DAYS,
            "prune_enabled": cfg.CONSOLIDATION_PRUNE_ENABLED,
            "decay_policies": cfg.DECAY_POLICIES,
            "prunable_episodes": [
                {
                    "id": ep["id"],
                    "content": ep["content"][:100],
                    "consolidated_at": ep.get("consolidated_at", ""),
                    "tags": ep.get("tags", "[]"),
                }
                for ep in prunable[:20]  # Cap at 20 for readability
            ],
            "low_confidence_records": [
                {
                    "id": rec["id"],
                    "record_type": rec.get("record_type", ""),
                    "confidence": rec.get("confidence", 0),
                    "topic_title": rec.get("topic_title", ""),
                    "embedding_text": rec.get("embedding_text", "")[:100],
                }
                for rec in low_conf[:20]
            ],
        }

        return DecayReportResult(
            prunable_episodes=len(prunable),
            low_confidence_records=len(low_conf),
            protected_episodes=protected_count,
            details=details,
        )

    def protect(
        self,
        episode_id: str | None = None,
        tag: str | None = None,
        *,
        scope: ScopeEnvelope | dict[str, object] | None = None,
    ) -> ProtectResult:
        """Mark episodes as immune to pruning.

        Args:
            episode_id: Protect a specific episode by UUID.
            tag: Protect all episodes with this tag.

        Returns:
            ProtectResult with status and count.
        """
        from consolidation_memory.database import protect_episode, protect_by_tag

        self._ensure_open()
        if not episode_id and not tag:
            return ProtectResult(
                status="error", message="Provide either episode_id or tag."
            )

        resolved_scope = self.resolve_scope(scope)
        denied_message = _write_denied_message(resolved_scope)
        if denied_message is not None:
            return ProtectResult(status="write_denied", message=denied_message)

        mutation_filter = _resolved_scope_to_mutation_filter(resolved_scope)
        total = 0
        episode_missing = False
        if episode_id:
            found = protect_episode(episode_id, scope=mutation_filter)
            if not found:
                episode_missing = True
            else:
                total += 1

        if tag:
            count = protect_by_tag(tag, scope=mutation_filter)
            total += count

        if total == 0 and episode_missing and tag is None:
            return ProtectResult(status="not_found", message=f"Episode {episode_id} not found.")

        message = f"Protected {total} episode(s)."
        if episode_missing and tag is not None:
            message = (
                f"Protected {total} episode(s). "
                f"Episode {episode_id} was not found or is outside the writable scope."
            )

        return ProtectResult(
            status="protected",
            protected_count=total,
            message=message,
        )

    def consolidate(self) -> ConsolidationReport:
        """Run consolidation manually. Thread-safe with background thread.

        Returns:
            Dict with consolidation results or ``{"status": "already_running"}``.
        """
        from consolidation_memory.config import get_config
        from consolidation_memory.consolidation import run_consolidation
        from consolidation_memory.database import (
            mark_consolidation_scheduler_finished,
            mark_consolidation_scheduler_started,
            release_consolidation_lease,
            try_acquire_consolidation_lease,
        )

        self._ensure_open()
        if not self._consolidation_lock.acquire(blocking=False):
            return {"status": "already_running"}
        cfg = get_config()
        lease_seconds = cfg.CONSOLIDATION_MAX_DURATION + 60
        lease_acquired = False
        scheduler_finished = False
        try:
            lease_acquired = try_acquire_consolidation_lease(
                owner=self._scheduler_owner,
                lease_seconds=lease_seconds,
            )
            if not lease_acquired:
                return {
                    "status": "already_running",
                    "message": "A consolidation run is already in progress.",
                }

            mark_consolidation_scheduler_started(
                owner=self._scheduler_owner,
                trigger_reason="manual",
                utility_score=None,
            )

            report = run_consolidation(vector_store=self._vector_store)
            run_status = report.get("status") if isinstance(report, dict) else None
            scheduler_status = (
                RUN_STATUS_FAILED if run_status == RUN_STATUS_FAILED else RUN_STATUS_COMPLETED
            )
            error_message = None
            if scheduler_status == RUN_STATUS_FAILED and isinstance(report, dict):
                error_message = str(
                    report.get("error_message")
                    or report.get("message")
                    or "consolidation failed"
                )

            mark_consolidation_scheduler_finished(
                owner=self._scheduler_owner,
                status=scheduler_status,
                interval_hours=cfg.CONSOLIDATION_INTERVAL_HOURS,
                error_message=error_message,
            )
            scheduler_finished = True
            return report
        except Exception as exc:
            if lease_acquired:
                try:
                    mark_consolidation_scheduler_finished(
                        owner=self._scheduler_owner,
                        status=RUN_STATUS_FAILED,
                        interval_hours=cfg.CONSOLIDATION_INTERVAL_HOURS,
                        error_message=str(exc),
                    )
                    scheduler_finished = True
                except Exception:
                    logger.exception("Failed to persist manual consolidation failure state")
            raise
        finally:
            if lease_acquired and not scheduler_finished:
                try:
                    release_consolidation_lease(self._scheduler_owner)
                except Exception:
                    logger.exception("Failed to release scheduler lease after manual consolidation")
            self._consolidation_lock.release()

    def compact(self) -> CompactResult:
        """Compact the FAISS index by removing tombstoned vectors.

        Returns:
            CompactResult with status, tombstones removed, and new index size.
        """
        self._ensure_open()
        removed = self._vector_store.compact()
        if removed == 0:
            return CompactResult(
                status="no_tombstones",
                tombstones_removed=0,
                index_size=self._vector_store.size,
            )
        logger.info("Manual compaction removed %d tombstones", removed)
        return CompactResult(
            status="compacted",
            tombstones_removed=removed,
            index_size=self._vector_store.size,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _persist_episode_anchors(self, episode_id: str, content: str) -> None:
        """Best-effort anchor extraction and persistence for stored episodes."""
        from consolidation_memory.anchors import extract_anchors
        from consolidation_memory.database import insert_episode_anchors

        try:
            anchors = extract_anchors(content)
        except Exception as e:
            logger.warning("Anchor extraction failed for episode %s: %s", episode_id, e)
            return

        if not anchors:
            return

        try:
            insert_episode_anchors(episode_id, anchors)
        except Exception as e:
            logger.warning("Failed to persist anchors for episode %s: %s", episode_id, e)

    def _compute_health(
        self,
        last_run: dict[str, object] | None,
        interval_hours: float,
        compaction_threshold: float,
        knowledge_consistency: dict[str, object] | None = None,
    ) -> HealthStatus:
        return _compute_health_runtime(
            self,
            last_run,
            interval_hours,
            compaction_threshold,
            knowledge_consistency,
        )

    def _probe_backend(self) -> bool:
        return _probe_backend_runtime(self)

    def _check_embedding_backend(self) -> None:
        _check_embedding_backend_runtime(self)

    def _record_recall_signal(
        self,
        *,
        miss: bool = False,
        fallback: bool = False,
        timestamp_monotonic: float | None = None,
    ) -> None:
        _record_recall_signal_runtime(
            self,
            miss=miss,
            fallback=fallback,
            timestamp_monotonic=timestamp_monotonic,
        )

    def _recent_recall_signal_counts(
        self,
        lookback_seconds: float,
        now_monotonic: float | None = None,
    ) -> tuple[int, int]:
        return _recent_recall_signal_counts_runtime(
            self,
            lookback_seconds,
            now_monotonic,
        )

    def _compute_consolidation_utility(
        self,
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, object]:
        return _compute_consolidation_utility_runtime(self, now_monotonic=now_monotonic)

    def _should_trigger_consolidation(
        self,
        *,
        now_monotonic: float,
        last_run_monotonic: float,
        interval_seconds: float,
        utility_score: float,
        raw_signals: Mapping[str, object] | None = None,
    ) -> tuple[bool, str]:
        return _should_trigger_consolidation_runtime(
            now_monotonic=now_monotonic,
            last_run_monotonic=last_run_monotonic,
            interval_seconds=interval_seconds,
            utility_score=utility_score,
            raw_signals=raw_signals,
        )

    def _should_trigger_scheduler_run(
        self,
        *,
        scheduler_state: dict[str, object],
        utility_score: float,
        raw_signals: Mapping[str, object] | None = None,
        now_utc: datetime | None = None,
    ) -> tuple[bool, str]:
        return _should_trigger_scheduler_run_runtime(
            scheduler_state=scheduler_state,
            utility_score=utility_score,
            raw_signals=raw_signals,
            now_utc=now_utc,
        )

    def _maybe_auto_consolidate(self, *, trigger_source: str) -> bool:
        return _maybe_auto_consolidate_runtime(self, trigger_source=trigger_source)

    def _submit_auto_consolidation(
        self,
        *,
        trigger_source: str,
        trigger_reason: str,
        utility_state: dict[str, object],
        utility_score: float,
    ) -> bool:
        return _submit_auto_consolidation_runtime(
            self,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            utility_state=utility_state,
            utility_score=utility_score,
        )

    def _finalize_auto_consolidation(self, future: Future[ConsolidationReport]) -> None:
        _finalize_auto_consolidation_runtime(self, future)

    def _start_consolidation_thread(self) -> None:
        _start_consolidation_thread_runtime(self)

    def _consolidation_loop(self) -> None:
        _consolidation_loop_runtime(self, monotonic_fn=time.monotonic)
