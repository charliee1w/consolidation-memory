"""Shared knowledge record embedding cache for recall.

Caches embedded record texts for vector search. Invalidated when records change
(after consolidation or correction). Thread-safe.

Same race-condition prevention pattern as topic_cache: version counter guards
against stale writes when invalidation happens during a cache-miss fetch.

Cache slots:
- _cache_all: include_expired=True (unscoped superset)
- _cache_unexpired: include_expired=False (unscoped filtered subset)
- _scoped_cache: include_expired + scope-specific slices used by scoped recall

When the unexpired unscoped slot is requested, we first check if the all-records
slot is fresh and filter from it, avoiding a redundant embed call.
"""

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

from consolidation_memory.database import get_all_active_records
from consolidation_memory.backends import encode_documents
from consolidation_memory.utils import parse_datetime

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_version: int = 0

_EMPTY_SLOT: dict = {
    "version": -1,
    "texts": [],
    "vecs": None,
    "records": [],
    "refresh_after_epoch": None,
    "loaded": False,
}

_cache_all: dict = dict(_EMPTY_SLOT)
_cache_unexpired: dict = dict(_EMPTY_SLOT)
_SCOPED_CACHE_MAX = 16
_scoped_cache: "OrderedDict[tuple[bool, tuple[tuple[str, str], ...]], dict]" = OrderedDict()


def invalidate() -> None:
    """Force re-embedding on next get_record_vecs() call."""
    global _version
    with _lock:
        _version += 1
        _scoped_cache.clear()


def _scope_cache_key(scope: dict[str, str | None]) -> tuple[tuple[str, str], ...]:
    """Normalize a scope filter dict into a deterministic hashable cache key."""
    normalized = []
    for key, value in sorted(scope.items(), key=lambda item: item[0]):
        if value is None:
            continue
        normalized.append((str(key), str(value)))
    return tuple(normalized)


def _normalize_datetime(value: str | datetime | None) -> datetime | None:
    """Parse a datetime-like value into UTC, returning None on invalid input."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = parse_datetime(value)
        except (ValueError, TypeError):
            return None
    elif value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_record_current(record: dict, reference_time: datetime | None = None) -> bool:
    """Check whether a record's validity window includes the reference time."""
    now = reference_time or datetime.now(timezone.utc)
    valid_from = _normalize_datetime(record.get("valid_from"))
    if valid_from and valid_from > now:
        return False

    valid_until = _normalize_datetime(record.get("valid_until"))
    if valid_until and valid_until <= now:
        return False

    return True


def _filter_unexpired(
    records: list[dict],
    vecs: np.ndarray | None,
) -> tuple[list[dict], np.ndarray | None]:
    """Filter records whose validity window does not include the current time."""
    mask = [_is_record_current(r) for r in records]
    filtered_records = [r for r, keep in zip(records, mask) if keep]
    if not filtered_records:
        return [], None
    filtered_vecs = vecs[mask] if vecs is not None else None
    return filtered_records, filtered_vecs


def _next_refresh_after_epoch(records: list[dict], reference_time: datetime | None = None) -> float | None:
    """Return the next wall-clock time when unexpired membership can change."""
    now = reference_time or datetime.now(timezone.utc)
    candidates: list[float] = []
    for record in records:
        valid_from = _normalize_datetime(record.get("valid_from"))
        if valid_from is not None and valid_from > now:
            candidates.append(valid_from.timestamp())

        valid_until = _normalize_datetime(record.get("valid_until"))
        if valid_until is not None and valid_until > now:
            candidates.append(valid_until.timestamp())

    return min(candidates) if candidates else None


def _slot_is_fresh(slot: dict, *, include_expired: bool) -> bool:
    """Return True when a cache slot can be safely reused."""
    if slot.get("version") != _version or not slot.get("loaded", False):
        return False
    if include_expired:
        return True

    refresh_after = slot.get("refresh_after_epoch")
    if refresh_after is None:
        return True
    return time.time() < float(refresh_after)


def _populate_slot(
    slot: dict,
    *,
    version: int,
    texts: list[str],
    vecs: np.ndarray | None,
    records: list[dict],
    refresh_after_epoch: float | None = None,
) -> None:
    """Write a cache slot atomically under the module lock."""
    slot["version"] = version
    slot["texts"] = texts
    slot["vecs"] = vecs
    slot["records"] = records
    slot["refresh_after_epoch"] = refresh_after_epoch
    slot["loaded"] = True


def _derive_unexpired_slot(
    slot: dict,
    *,
    version: int,
    records: list[dict],
    vecs: np.ndarray | None,
) -> tuple[list[dict], np.ndarray | None]:
    """Populate an unexpired slot from an all-records snapshot."""
    filtered_records, filtered_vecs = _filter_unexpired(records, vecs)
    _populate_slot(
        slot,
        version=version,
        texts=[r["embedding_text"] for r in filtered_records],
        vecs=filtered_vecs,
        records=filtered_records,
        refresh_after_epoch=_next_refresh_after_epoch(records),
    )
    return filtered_records, filtered_vecs


def get_record_vecs(
    include_expired: bool = False,
    scope: dict[str, str | None] | None = None,
) -> tuple[list[dict], np.ndarray | None]:
    """Return (records, embedding_matrix) with caching.

    Cache is valid as long as its version matches the current version.
    Returns ([], None) if no records exist.

    Unscoped requests use two dedicated slots to avoid redundant embedding:
    - include_expired=True: caches all active records (superset)
    - include_expired=False: filters from the True cache if fresh, else embeds independently

    Scoped requests use a small LRU keyed by (include_expired, scope filter).

    Args:
        include_expired: If True, include records with valid_until in the past.
        scope: Optional canonical scope filter.
    """
    if scope:
        scoped_key = (include_expired, _scope_cache_key(scope))
        scoped_all_key = (True, _scope_cache_key(scope))
        with _lock:
            scoped_slot = _scoped_cache.get(scoped_key)
            if scoped_slot is not None and _slot_is_fresh(
                scoped_slot,
                include_expired=include_expired,
            ):
                _scoped_cache.move_to_end(scoped_key)
                return scoped_slot["records"], scoped_slot["vecs"]

            if not include_expired:
                scoped_all_slot = _scoped_cache.get(scoped_all_key)
                if scoped_all_slot is not None and _slot_is_fresh(
                    scoped_all_slot,
                    include_expired=True,
                ):
                    target_slot = scoped_slot or dict(_EMPTY_SLOT)
                    filtered_records, filtered_vecs = _derive_unexpired_slot(
                        target_slot,
                        version=_version,
                        records=scoped_all_slot["records"],
                        vecs=scoped_all_slot["vecs"],
                    )
                    _scoped_cache[scoped_key] = target_slot
                    _scoped_cache.move_to_end(scoped_key)
                    while len(_scoped_cache) > _SCOPED_CACHE_MAX:
                        _scoped_cache.popitem(last=False)
                    return filtered_records, filtered_vecs
            fetch_version = _version

        fetch_include_expired = True if not include_expired else include_expired
        records = get_all_active_records(include_expired=fetch_include_expired, scope=scope)
        if not records:
            with _lock:
                if fetch_version == _version:
                    target_slot = _scoped_cache.get(scoped_key) or dict(_EMPTY_SLOT)
                    _populate_slot(
                        target_slot,
                        version=fetch_version,
                        texts=[],
                        vecs=None,
                        records=[],
                        refresh_after_epoch=None,
                    )
                    _scoped_cache[scoped_key] = target_slot
                    _scoped_cache.move_to_end(scoped_key)
                    while len(_scoped_cache) > _SCOPED_CACHE_MAX:
                        _scoped_cache.popitem(last=False)
            return [], None

        texts = [r["embedding_text"] for r in records]
        try:
            vecs = encode_documents(texts)
        except Exception as e:
            logger.warning("Failed to embed scoped record texts: %s", e, exc_info=True)
            if include_expired:
                return records, None
            filtered_records, _filtered_vecs = _filter_unexpired(records, vecs=None)
            return filtered_records, None

        with _lock:
            if _version == fetch_version:
                if include_expired:
                    target_slot = _scoped_cache.get(scoped_key) or dict(_EMPTY_SLOT)
                    _populate_slot(
                        target_slot,
                        version=fetch_version,
                        texts=texts,
                        vecs=vecs,
                        records=records,
                    )
                    _scoped_cache[scoped_key] = target_slot
                    _scoped_cache.move_to_end(scoped_key)
                else:
                    all_slot = _scoped_cache.get(scoped_all_key) or dict(_EMPTY_SLOT)
                    _populate_slot(
                        all_slot,
                        version=fetch_version,
                        texts=texts,
                        vecs=vecs,
                        records=records,
                    )
                    _scoped_cache[scoped_all_key] = all_slot
                    _scoped_cache.move_to_end(scoped_all_key)

                    current_slot = _scoped_cache.get(scoped_key) or dict(_EMPTY_SLOT)
                    filtered_records, filtered_vecs = _derive_unexpired_slot(
                        current_slot,
                        version=fetch_version,
                        records=records,
                        vecs=vecs,
                    )
                    _scoped_cache[scoped_key] = current_slot
                    _scoped_cache.move_to_end(scoped_key)
                while len(_scoped_cache) > _SCOPED_CACHE_MAX:
                    _scoped_cache.popitem(last=False)
            else:
                logger.debug(
                    "Scoped record cache populate discarded: version changed %d -> %d during fetch",
                    fetch_version,
                    _version,
                )
                scoped_slot = _scoped_cache.get(scoped_key)
                if scoped_slot is not None and _slot_is_fresh(
                    scoped_slot,
                    include_expired=include_expired,
                ):
                    _scoped_cache.move_to_end(scoped_key)
                    return scoped_slot["records"], scoped_slot["vecs"]

        if include_expired:
            return records, vecs
        return filtered_records, filtered_vecs

    slot = _cache_all if include_expired else _cache_unexpired

    with _lock:
        if _slot_is_fresh(slot, include_expired=include_expired):
            return slot["records"], slot["vecs"]

        # If requesting unexpired, check if the all-records cache is fresh
        # and filter from it instead of re-embedding
        if not include_expired and _slot_is_fresh(_cache_all, include_expired=True):
            filtered_records, filtered_vecs = _derive_unexpired_slot(
                _cache_unexpired,
                version=_version,
                records=_cache_all["records"],
                vecs=_cache_all["vecs"],
            )
            return filtered_records, filtered_vecs

        fetch_version = _version

    fetch_include_expired = True if not include_expired else include_expired
    records = get_all_active_records(include_expired=fetch_include_expired)
    if not records:
        with _lock:
            if _version == fetch_version:
                _populate_slot(
                    slot,
                    version=fetch_version,
                    texts=[],
                    vecs=None,
                    records=[],
                    refresh_after_epoch=None,
                )
        return [], None

    texts = [r["embedding_text"] for r in records]
    try:
        vecs = encode_documents(texts)
    except Exception as e:
        logger.warning("Failed to embed record texts: %s", e, exc_info=True)
        if include_expired:
            return records, None
        filtered_records, _filtered_vecs = _filter_unexpired(records, vecs=None)
        return filtered_records, None

    with _lock:
        if _version == fetch_version:
            if include_expired:
                _populate_slot(
                    slot,
                    version=fetch_version,
                    texts=texts,
                    vecs=vecs,
                    records=records,
                )
            else:
                _populate_slot(
                    _cache_all,
                    version=fetch_version,
                    texts=texts,
                    vecs=vecs,
                    records=records,
                )
                filtered_records, filtered_vecs = _derive_unexpired_slot(
                    _cache_unexpired,
                    version=fetch_version,
                    records=records,
                    vecs=vecs,
                )
        else:
            logger.debug(
                "Record cache populate discarded: version changed %d -> %d during fetch",
                fetch_version, _version,
            )
            # Return cached data if available (consistent pair), else use our fetch
            if _slot_is_fresh(slot, include_expired=include_expired):
                return slot["records"], slot["vecs"]

    if include_expired:
        return records, vecs
    return filtered_records, filtered_vecs
