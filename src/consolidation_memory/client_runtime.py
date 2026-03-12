"""Runtime helpers extracted from MemoryClient.

Keeps scheduling/health orchestration isolated from CRUD-facing client logic.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from consolidation_memory.types import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    ConsolidationReport,
    HealthStatus,
)
from consolidation_memory.utils import parse_datetime

logger = logging.getLogger("consolidation_memory")

_FORCE_CHALLENGED_BACKLOG_FLOOR = 10
_FORCE_BACKLOG_FLOOR = 25
_FORCE_BACKLOG_RATIO = 0.5
_FORCE_CHALLENGED_BACKLOG_RATIO = 0.25
_STALE_CHALLENGED_CLAIM_TTL_HOURS = 24.0 * 7.0
_STALE_CHALLENGED_CLAIM_TRIAGE_MAX = 200


class RuntimeClient(Protocol):
    _vector_store: Any
    _probe_cache: tuple[bool, float] | None
    _probe_cache_ttl: float
    _scheduler_signal_lock: threading.Lock
    _recall_miss_events: deque[float]
    _recall_fallback_events: deque[float]
    _auto_consolidate_enabled: bool
    _consolidation_stop: threading.Event
    _consolidation_future: Future[ConsolidationReport] | None
    _consolidation_pool: ThreadPoolExecutor | None
    _consolidation_lock: threading.Lock
    _scheduler_owner: str
    _consolidation_thread: threading.Thread | None
    def _probe_backend(self) -> bool: ...
    def _compute_consolidation_utility(self, *, now_monotonic: float | None = None) -> dict[str, object]: ...
    def _should_trigger_consolidation(
        self,
        *,
        now_monotonic: float,
        last_run_monotonic: float,
        interval_seconds: float,
        utility_score: float,
        raw_signals: Mapping[str, object] | None = None,
    ) -> tuple[bool, str]: ...
    def _should_trigger_scheduler_run(
        self,
        *,
        scheduler_state: dict[str, object],
        utility_score: float,
        raw_signals: Mapping[str, object] | None = None,
        now_utc: datetime | None = None,
    ) -> tuple[bool, str]: ...
    def _submit_auto_consolidation(
        self,
        *,
        trigger_source: str,
        trigger_reason: str,
        utility_state: dict[str, object],
        utility_score: float,
    ) -> bool: ...
    def _finalize_auto_consolidation(self, future: Future[ConsolidationReport]) -> None: ...
    def _consolidation_loop(self) -> None: ...


def compute_health(
    client: RuntimeClient,
    last_run: dict[str, object] | None,
    interval_hours: float,
    compaction_threshold: float,
    knowledge_consistency: dict[str, Any] | None = None,
) -> HealthStatus:
    """Build health assessment dict."""
    from consolidation_memory.config import get_config

    cfg = get_config()
    issues: list[str] = []

    backend_reachable = client._probe_backend()
    if not backend_reachable:
        issues.append("Embedding backend unreachable")

    tombstone_ratio = client._vector_store.tombstone_ratio
    if tombstone_ratio > compaction_threshold * 0.75:
        issues.append(
            f"FAISS tombstone ratio {tombstone_ratio:.1%} approaching "
            f"compaction threshold {compaction_threshold:.0%}"
        )

    if client._vector_store.size >= cfg.FAISS_PLATFORM_REVIEW_THRESHOLD:
        issues.append(
            "Vector corpus exceeds platform review threshold "
            f"({client._vector_store.size} >= {cfg.FAISS_PLATFORM_REVIEW_THRESHOLD}); "
            "plan a scaling pass for higher concurrency/corpus size."
        )

    if knowledge_consistency:
        ratio_val = knowledge_consistency.get("consistency_ratio")
        threshold_val = knowledge_consistency.get("threshold")
        if isinstance(ratio_val, (int, float)) and isinstance(threshold_val, (int, float)):
            if float(ratio_val) < float(threshold_val):
                issues.append(
                    "Knowledge markdown/record consistency below target "
                    f"({float(ratio_val):.1%} < {float(threshold_val):.1%})"
                )

    if last_run:
        if last_run.get("status") == RUN_STATUS_FAILED:
            issues.append(
                f"Last consolidation failed: {last_run.get('error_message', 'unknown')}"
            )
        completed_at = last_run.get("completed_at") or last_run.get("started_at")
        if completed_at and isinstance(completed_at, str):
            try:
                last_time = parse_datetime(completed_at)
                age_hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
                if age_hours > interval_hours * 2:
                    issues.append(
                        f"Last consolidation was {age_hours:.0f}h ago "
                        f"(expected every {interval_hours:.0f}h)"
                    )
            except (ValueError, TypeError):
                pass

    if issues:
        has_critical = not backend_reachable
        status = "error" if has_critical else "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "issues": issues,
        "backend_reachable": backend_reachable,
    }


def probe_backend(client: RuntimeClient) -> bool:
    """Quick check if embedding backend is reachable. Cached for 30s."""
    from consolidation_memory.config import get_config
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    cfg = get_config()
    if cfg.EMBEDDING_BACKEND == "fastembed":
        return True

    if client._probe_cache is not None:
        cached_result, cached_at = client._probe_cache
        if time.monotonic() - cached_at < client._probe_cache_ttl:
            return cached_result

    try:
        req = Request(
            f"{cfg.EMBEDDING_API_BASE}/models",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=3) as resp:  # nosec B310
            resp.read()
        client._probe_cache = (True, time.monotonic())
        return True
    except (URLError, ConnectionError, TimeoutError, OSError):
        client._probe_cache = (False, time.monotonic())
        return False


def check_embedding_backend(client: RuntimeClient) -> None:
    """Verify the embedding backend is reachable."""
    from consolidation_memory.config import get_config
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    cfg = get_config()
    if cfg.EMBEDDING_BACKEND == "fastembed":
        logger.info("Embedding backend: fastembed (local, no server check needed)")
        return

    try:
        req = Request(
            f"{cfg.EMBEDDING_API_BASE}/models",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:  # nosec B310
            body_raw = resp.read()
        body = json.loads(body_raw)
        if not isinstance(body, Mapping):
            raise ValueError("response body must be a JSON object")
        models = body.get("data", [])
        if not isinstance(models, list):
            raise ValueError("response field 'data' must be a list")
        model_ids = [
            str(model.get("id", ""))
            for model in models
            if isinstance(model, Mapping)
        ]
        if cfg.EMBEDDING_MODEL_NAME not in model_ids:
            logger.warning(
                "Embedding model '%s' not found. Loaded: %s",
                cfg.EMBEDDING_MODEL_NAME, model_ids,
            )
        else:
            logger.info("Embedding backend health check passed (%s).", cfg.EMBEDDING_BACKEND)
    except (URLError, ConnectionError, TimeoutError) as exc:
        logger.warning(
            "%s not reachable at %s: %s. Store/recall will fail until available.",
            cfg.EMBEDDING_BACKEND, cfg.EMBEDDING_API_BASE, exc,
        )
    except Exception as exc:
        logger.warning(
            "%s health check returned malformed payload from %s: %s",
            cfg.EMBEDDING_BACKEND,
            cfg.EMBEDDING_API_BASE,
            exc,
        )


def record_recall_signal(
    client: RuntimeClient,
    *,
    miss: bool = False,
    fallback: bool = False,
    timestamp_monotonic: float | None = None,
) -> None:
    """Record recall miss/fallback events for utility scheduling."""
    if not miss and not fallback:
        return
    ts = timestamp_monotonic if timestamp_monotonic is not None else time.monotonic()
    with client._scheduler_signal_lock:
        if miss:
            client._recall_miss_events.append(ts)
        if fallback:
            client._recall_fallback_events.append(ts)


def recent_recall_signal_counts(
    client: RuntimeClient,
    lookback_seconds: float,
    now_monotonic: float | None = None,
) -> tuple[int, int]:
    """Return miss/fallback counts within lookback window."""
    now = now_monotonic if now_monotonic is not None else time.monotonic()
    cutoff = now - max(1.0, lookback_seconds)
    with client._scheduler_signal_lock:
        while client._recall_miss_events and client._recall_miss_events[0] < cutoff:
            client._recall_miss_events.popleft()
        while client._recall_fallback_events and client._recall_fallback_events[0] < cutoff:
            client._recall_fallback_events.popleft()
        return len(client._recall_miss_events), len(client._recall_fallback_events)


def compute_consolidation_utility(
    client: RuntimeClient,
    *,
    now_monotonic: float | None = None,
) -> dict[str, object]:
    """Compute current utility score and signal breakdown."""
    from consolidation_memory.config import get_config
    from consolidation_memory.consolidation.utility_scheduler import compute_utility_score
    from consolidation_memory.database import (
        count_active_challenged_claims,
        count_contradictions_since,
        get_stats,
    )

    cfg = get_config()
    lookback_seconds = max(300.0, cfg.CONSOLIDATION_INTERVAL_HOURS * 3600.0)
    miss_count, fallback_count = recent_recall_signal_counts(
        client,
        lookback_seconds=lookback_seconds,
        now_monotonic=now_monotonic,
    )
    now_wallclock = datetime.now(timezone.utc)
    contradictions_since = (now_wallclock - timedelta(seconds=lookback_seconds)).isoformat()

    stats = get_stats()
    pending_backlog = int(stats["episodic_buffer"]["pending_consolidation"])
    contradiction_count = count_contradictions_since(contradictions_since)
    challenged_backlog = count_active_challenged_claims(as_of=now_wallclock.isoformat())

    score_breakdown = compute_utility_score(
        unconsolidated_backlog=pending_backlog,
        recall_miss_count=miss_count,
        recall_fallback_count=fallback_count,
        contradiction_count=contradiction_count,
        challenged_claim_backlog=challenged_backlog,
        weights=cfg.CONSOLIDATION_UTILITY_WEIGHTS,
        backlog_target=max(1, cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN),
        recall_signal_target=3,
        contradiction_target=3,
        challenged_claim_target=max(1, cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN // 4),
    )

    return {
        "score": score_breakdown["score"],
        "normalized_signals": score_breakdown["normalized_signals"],
        "weighted_components": score_breakdown["weighted_components"],
        "raw_signals": {
            "unconsolidated_backlog": pending_backlog,
            "recall_miss_count": miss_count,
            "recall_fallback_count": fallback_count,
            "contradiction_count": contradiction_count,
            "challenged_claim_backlog": challenged_backlog,
            "lookback_seconds": lookback_seconds,
        },
    }


def _compute_force_thresholds(*, max_episodes_per_run: int) -> tuple[int, int]:
    max_episodes = max(1, int(max_episodes_per_run))

    backlog_threshold = int(max_episodes * _FORCE_BACKLOG_RATIO)
    backlog_threshold = max(_FORCE_BACKLOG_FLOOR, backlog_threshold)
    backlog_threshold = min(max_episodes, backlog_threshold)

    challenged_threshold = int(max_episodes * _FORCE_CHALLENGED_BACKLOG_RATIO)
    challenged_threshold = max(_FORCE_CHALLENGED_BACKLOG_FLOOR, challenged_threshold)
    challenged_threshold = min(backlog_threshold, challenged_threshold)

    return backlog_threshold, challenged_threshold


def _coerce_signal_int(
    raw_signals: Mapping[str, object] | None,
    key: str,
) -> int:
    if raw_signals is None:
        return 0
    value = raw_signals.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _force_trigger_reason(
    *,
    raw_signals: Mapping[str, object] | None,
    max_episodes_per_run: int,
) -> str | None:
    if raw_signals is None:
        return None
    backlog_threshold, challenged_threshold = _compute_force_thresholds(
        max_episodes_per_run=max_episodes_per_run,
    )
    pending_backlog = _coerce_signal_int(raw_signals, "unconsolidated_backlog")
    challenged_backlog = _coerce_signal_int(raw_signals, "challenged_claim_backlog")

    if pending_backlog >= backlog_threshold:
        return "backlog_pressure"
    if challenged_backlog >= challenged_threshold:
        return "challenged_backlog_pressure"
    return None


def _triage_stale_challenged_claims(now_utc: datetime | None = None) -> dict[str, object]:
    from consolidation_memory.database import auto_expire_stale_challenged_claims

    as_of = (now_utc or datetime.now(timezone.utc)).isoformat()
    report = auto_expire_stale_challenged_claims(
        as_of=as_of,
        max_age_hours=_STALE_CHALLENGED_CLAIM_TTL_HOURS,
        max_claims=_STALE_CHALLENGED_CLAIM_TRIAGE_MAX,
    )
    expired_count = int(report.get("expired_count", 0))
    if expired_count > 0:
        logger.info(
            "Auto-triaged %d stale challenged claims (ttl_hours=%.0f, cutoff=%s)",
            expired_count,
            _STALE_CHALLENGED_CLAIM_TTL_HOURS,
            report.get("cutoff"),
        )
    return report


def should_trigger_consolidation(
    *,
    now_monotonic: float,
    last_run_monotonic: float,
    interval_seconds: float,
    utility_score: float,
    raw_signals: Mapping[str, object] | None = None,
) -> tuple[bool, str]:
    """Decide whether to trigger consolidation this cycle."""
    from consolidation_memory.config import get_config

    if now_monotonic - last_run_monotonic >= interval_seconds:
        return True, "interval"

    cfg = get_config()
    pressure_reason = _force_trigger_reason(
        raw_signals=raw_signals,
        max_episodes_per_run=cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN,
    )
    if pressure_reason is not None:
        return True, pressure_reason
    if utility_score >= cfg.CONSOLIDATION_UTILITY_THRESHOLD:
        return True, "utility"
    return False, "none"


def should_trigger_scheduler_run(
    *,
    scheduler_state: dict[str, object],
    utility_score: float,
    raw_signals: Mapping[str, object] | None = None,
    now_utc: datetime | None = None,
) -> tuple[bool, str]:
    """Decide if scheduler state warrants launching a run."""
    from consolidation_memory.config import get_config

    cfg = get_config()
    now = now_utc or datetime.now(timezone.utc)
    next_due_raw = scheduler_state.get("next_due_at")
    interval_due = True
    if isinstance(next_due_raw, str) and next_due_raw.strip():
        try:
            interval_due = now >= parse_datetime(next_due_raw)
        except (ValueError, TypeError):
            interval_due = True
    if interval_due:
        return True, "interval"
    pressure_reason = _force_trigger_reason(
        raw_signals=raw_signals,
        max_episodes_per_run=cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN,
    )
    if pressure_reason is not None:
        return True, pressure_reason
    if utility_score >= cfg.CONSOLIDATION_UTILITY_THRESHOLD:
        return True, "utility"
    return False, "none"


def maybe_auto_consolidate(client: RuntimeClient, *, trigger_source: str) -> bool:
    """Best-effort non-blocking auto-consolidation trigger for API operations."""
    from consolidation_memory.config import get_config
    from consolidation_memory.database import get_consolidation_scheduler_state

    cfg = get_config()
    if (
        not client._auto_consolidate_enabled
        or not cfg.CONSOLIDATION_AUTO_RUN
        or client._consolidation_stop.is_set()
    ):
        return False
    if client._consolidation_future is not None and not client._consolidation_future.done():
        return False

    try:
        _triage_stale_challenged_claims()
        utility_state = client._compute_consolidation_utility()
        score_value = utility_state.get("score")
        utility_score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
        raw_signals = utility_state.get("raw_signals")
        signal_map = raw_signals if isinstance(raw_signals, Mapping) else None
        scheduler_state = get_consolidation_scheduler_state()
        should_run, trigger_reason = client._should_trigger_scheduler_run(
            scheduler_state=scheduler_state,
            utility_score=utility_score,
            raw_signals=signal_map,
        )
        if not should_run:
            return False
        return client._submit_auto_consolidation(
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            utility_state=utility_state,
            utility_score=utility_score,
        )
    except Exception:
        logger.exception(
            "Automatic consolidation tick failed (source=%s)",
            trigger_source,
        )
        return False


def submit_auto_consolidation(
    client: RuntimeClient,
    *,
    trigger_source: str,
    trigger_reason: str,
    utility_state: dict[str, object],
    utility_score: float,
) -> bool:
    """Submit one non-blocking consolidation run guarded by DB lease + process lock."""
    from consolidation_memory.config import get_config
    from consolidation_memory.consolidation import run_consolidation
    from consolidation_memory.database import (
        mark_consolidation_scheduler_started,
        release_consolidation_lease,
        try_acquire_consolidation_lease,
    )

    cfg = get_config()
    lease_seconds = cfg.CONSOLIDATION_MAX_DURATION + 60
    if not client._consolidation_lock.acquire(blocking=False):
        return False

    lease_acquired = False
    scheduled = False
    try:
        lease_acquired = try_acquire_consolidation_lease(
            owner=client._scheduler_owner,
            lease_seconds=lease_seconds,
        )
        if not lease_acquired:
            return False

        mark_consolidation_scheduler_started(
            owner=client._scheduler_owner,
            trigger_reason=trigger_reason,
            utility_score=utility_score,
        )
        if client._consolidation_pool is None:
            client._consolidation_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="consolidation",
            )
        future = client._consolidation_pool.submit(
            run_consolidation,
            vector_store=client._vector_store,
        )
        client._consolidation_future = future
        logger.info(
            "Auto consolidation submitted source=%s trigger=%s score=%.3f components=%s raw=%s",
            trigger_source,
            trigger_reason,
            utility_score,
            utility_state["weighted_components"],
            utility_state["raw_signals"],
        )
        if hasattr(future, "add_done_callback"):
            future.add_done_callback(
                lambda done: client._finalize_auto_consolidation(done)
            )
        else:
            client._finalize_auto_consolidation(future)
        scheduled = True
        return True
    except Exception:
        logger.exception("Failed to submit automatic consolidation run")
        return False
    finally:
        if not scheduled:
            if lease_acquired:
                try:
                    release_consolidation_lease(client._scheduler_owner)
                except Exception:
                    logger.exception("Failed to release scheduler lease after submit failure")
            if client._consolidation_lock.locked():
                try:
                    client._consolidation_lock.release()
                except RuntimeError:
                    pass


def finalize_auto_consolidation(
    client: RuntimeClient,
    future: Future[ConsolidationReport],
) -> None:
    """Handle completion bookkeeping for async auto-consolidation runs."""
    from consolidation_memory.config import get_config
    from consolidation_memory.database import (
        mark_consolidation_scheduler_finished,
        release_consolidation_lease,
    )

    cfg = get_config()
    max_duration = cfg.CONSOLIDATION_MAX_DURATION + 60
    status = RUN_STATUS_COMPLETED
    error_message: str | None = None

    try:
        result = future.result(timeout=max_duration)
        run_status = result.get("status") if isinstance(result, dict) else None
        if run_status == RUN_STATUS_FAILED:
            status = RUN_STATUS_FAILED
            error_message = str(
                result.get("error_message")
                or result.get("message")
                or "consolidation failed"
            )
        logger.info("Automatic consolidation completed with status=%s", run_status or status)
    except FuturesTimeoutError:
        future.cancel()
        status = RUN_STATUS_FAILED
        error_message = f"Timed out after {max_duration}s"
        logger.error("Automatic consolidation timed out after %ds", max_duration)
    except Exception as exc:
        status = RUN_STATUS_FAILED
        error_message = str(exc)
        logger.exception("Automatic consolidation failed")
    finally:
        try:
            mark_consolidation_scheduler_finished(
                owner=client._scheduler_owner,
                status=status,
                interval_hours=cfg.CONSOLIDATION_INTERVAL_HOURS,
                error_message=error_message,
            )
        except Exception:
            logger.exception("Failed to persist scheduler completion state")
            try:
                release_consolidation_lease(client._scheduler_owner)
            except Exception:
                logger.exception("Failed to release scheduler lease after completion failure")
        client._consolidation_future = None
        if client._consolidation_lock.locked():
            try:
                client._consolidation_lock.release()
            except RuntimeError:
                pass


def start_consolidation_thread(client: RuntimeClient) -> None:
    """Start the background consolidation daemon thread."""
    client._consolidation_stop.clear()
    client._consolidation_thread = threading.Thread(
        target=client._consolidation_loop,
        daemon=True,
        name="consolidation-bg",
    )
    client._consolidation_thread.start()


def consolidation_loop(
    client: RuntimeClient,
    *,
    monotonic_fn: Callable[[], float] | None = None,
) -> None:
    """Background consolidation thread target."""
    from consolidation_memory.config import get_config
    from consolidation_memory.consolidation import run_consolidation
    from consolidation_memory.database import (
        mark_consolidation_scheduler_finished,
        mark_consolidation_scheduler_started,
        release_consolidation_lease,
        try_acquire_consolidation_lease,
    )

    cfg = get_config()
    now_monotonic = monotonic_fn or time.monotonic
    interval = cfg.CONSOLIDATION_INTERVAL_HOURS * 3600
    poll_interval = min(interval, 60.0)
    max_duration = cfg.CONSOLIDATION_MAX_DURATION + 60
    logger.info(
        "Background consolidation thread started "
        "(interval: %.1fh, poll: %.0fs, utility_threshold: %.2f, timeout: %ds)",
        cfg.CONSOLIDATION_INTERVAL_HOURS,
        poll_interval,
        cfg.CONSOLIDATION_UTILITY_THRESHOLD,
        max_duration,
    )

    if client._consolidation_pool is None:
        client._consolidation_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="consolidation"
        )

    last_run_monotonic = now_monotonic()

    while not client._consolidation_stop.wait(timeout=poll_interval):
        if client._consolidation_stop.is_set():
            break
        if client._consolidation_pool is None:
            break
        current_monotonic = now_monotonic()
        try:
            _triage_stale_challenged_claims()
            utility_state = client._compute_consolidation_utility(now_monotonic=current_monotonic)
            score_value = utility_state.get("score")
            utility_score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
            raw_signals = utility_state.get("raw_signals")
            signal_map = raw_signals if isinstance(raw_signals, Mapping) else None
            should_run, trigger_reason = client._should_trigger_consolidation(
                now_monotonic=current_monotonic,
                last_run_monotonic=last_run_monotonic,
                interval_seconds=interval,
                utility_score=utility_score,
                raw_signals=signal_map,
            )
        except Exception:
            logger.exception("Background consolidation preflight failed")
            continue
        if not should_run:
            continue
        if not client._consolidation_lock.acquire(blocking=False):
            logger.info("Consolidation already running, skipping")
            continue
        lease_acquired = False
        try:
            lease_acquired = try_acquire_consolidation_lease(
                owner=client._scheduler_owner,
                lease_seconds=max_duration,
            )
            if not lease_acquired:
                logger.info("Consolidation lease held by another process; skipping")
                continue

            mark_consolidation_scheduler_started(
                owner=client._scheduler_owner,
                trigger_reason=trigger_reason,
                utility_score=utility_score,
            )

            logger.info(
                "Consolidation trigger=%s utility_score=%.3f components=%s raw=%s",
                trigger_reason,
                utility_score,
                utility_state["weighted_components"],
                utility_state["raw_signals"],
            )
            future = client._consolidation_pool.submit(
                run_consolidation, vector_store=client._vector_store
            )
            try:
                result = future.result(timeout=max_duration)
                last_run_monotonic = now_monotonic()
                run_status = result.get("status") if isinstance(result, dict) else RUN_STATUS_COMPLETED
                scheduler_status = (
                    RUN_STATUS_FAILED if run_status == RUN_STATUS_FAILED else RUN_STATUS_COMPLETED
                )
                error_message = None
                if scheduler_status == RUN_STATUS_FAILED and isinstance(result, dict):
                    error_message = str(
                        result.get("error_message")
                        or result.get("message")
                        or "consolidation failed"
                    )
                mark_consolidation_scheduler_finished(
                    owner=client._scheduler_owner,
                    status=scheduler_status,
                    interval_hours=cfg.CONSOLIDATION_INTERVAL_HOURS,
                    error_message=error_message,
                )
                logger.info(
                    "Background consolidation completed: %s",
                    result.get("status", result),
                )
            except FuturesTimeoutError:
                future.cancel()
                mark_consolidation_scheduler_finished(
                    owner=client._scheduler_owner,
                    status=RUN_STATUS_FAILED,
                    interval_hours=cfg.CONSOLIDATION_INTERVAL_HOURS,
                    error_message=f"Timed out after {max_duration}s",
                )
                logger.error(
                    "Background consolidation timed out after %ds; "
                    "releasing lock. The worker thread will be abandoned.",
                    max_duration,
                )
        except Exception:
            if lease_acquired:
                try:
                    mark_consolidation_scheduler_finished(
                        owner=client._scheduler_owner,
                        status=RUN_STATUS_FAILED,
                        interval_hours=cfg.CONSOLIDATION_INTERVAL_HOURS,
                        error_message="Background consolidation failed",
                    )
                except Exception:
                    logger.exception("Failed to persist scheduler failure state")
            logger.exception("Background consolidation failed")
        finally:
            if lease_acquired:
                try:
                    release_consolidation_lease(client._scheduler_owner)
                except Exception:
                    logger.exception("Failed to release scheduler lease")
            client._consolidation_lock.release()

    logger.info("Background consolidation thread stopped.")
