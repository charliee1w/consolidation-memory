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


def _filter_unexpired(records: list[dict], vecs: np.ndarray) -> tuple[list[dict], np.ndarray | None]:
    """Filter records whose validity window does not include the current time."""
    mask = [_is_record_current(r) for r in records]
    filtered_records = [r for r, keep in zip(records, mask) if keep]
    if not filtered_records:
        return [], None
    filtered_vecs = vecs[mask] if vecs is not None else None
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
        with _lock:
            scoped_slot = _scoped_cache.get(scoped_key)
            if (
                scoped_slot is not None
                and scoped_slot["version"] == _version
                and scoped_slot["vecs"] is not None
            ):
                _scoped_cache.move_to_end(scoped_key)
                return scoped_slot["records"], scoped_slot["vecs"]
            fetch_version = _version

        records = get_all_active_records(include_expired=include_expired, scope=scope)
        if not records:
            return [], None

        texts = [r["embedding_text"] for r in records]
        try:
            vecs = encode_documents(texts)
        except Exception as e:
            logger.warning("Failed to embed scoped record texts: %s", e, exc_info=True)
            return records, None

        with _lock:
            if _version == fetch_version:
                _scoped_cache[scoped_key] = {
                    "version": fetch_version,
                    "texts": texts,
                    "vecs": vecs,
                    "records": records,
                }
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
                if scoped_slot is not None and scoped_slot["vecs"] is not None:
                    _scoped_cache.move_to_end(scoped_key)
                    return scoped_slot["records"], scoped_slot["vecs"]

        return records, vecs

    slot = _cache_all if include_expired else _cache_unexpired

    with _lock:
        if slot["version"] == _version and slot["vecs"] is not None:
            return slot["records"], slot["vecs"]

        # If requesting unexpired, check if the all-records cache is fresh
        # and filter from it instead of re-embedding
        if not include_expired and _cache_all["version"] == _version and _cache_all["vecs"] is not None:
            filtered_records, filtered_vecs = _filter_unexpired(
                _cache_all["records"], _cache_all["vecs"],
            )
            _cache_unexpired["version"] = _version
            _cache_unexpired["records"] = filtered_records
            _cache_unexpired["vecs"] = filtered_vecs
            _cache_unexpired["texts"] = [r["embedding_text"] for r in filtered_records]
            return filtered_records, filtered_vecs

        fetch_version = _version

    records = get_all_active_records(include_expired=include_expired)
    if not records:
        return [], None

    texts = [r["embedding_text"] for r in records]
    try:
        vecs = encode_documents(texts)
    except Exception as e:
        logger.warning("Failed to embed record texts: %s", e, exc_info=True)
        return records, None

    with _lock:
        if _version == fetch_version:
            slot["version"] = fetch_version
            slot["texts"] = texts
            slot["vecs"] = vecs
            slot["records"] = records
        else:
            logger.debug(
                "Record cache populate discarded: version changed %d -> %d during fetch",
                fetch_version, _version,
            )
            # Return cached data if available (consistent pair), else use our fetch
            if slot["vecs"] is not None:
                return slot["records"], slot["vecs"]

    if not include_expired:
        return records, vecs

    # When we just populated the all-records cache, also populate unexpired
    # as a free side-effect
    with _lock:
        if _version == fetch_version:
            filtered_records, filtered_vecs = _filter_unexpired(records, vecs)
            _cache_unexpired["version"] = fetch_version
            _cache_unexpired["records"] = filtered_records
            _cache_unexpired["vecs"] = filtered_vecs
            _cache_unexpired["texts"] = [r["embedding_text"] for r in filtered_records]

    return records, vecs
