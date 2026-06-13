"""Disk-backed incremental embedding cache for recall hot paths.

Stores per-item vectors keyed by stable IDs and content hashes so process
restarts reuse prior embeddings instead of re-encoding entire corpora.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from consolidation_memory.process_write_lock import ProcessWriteLease

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_runtime_stores: dict[str, dict[str, tuple[str, np.ndarray]]] = {}


def _disk_write_lease() -> ProcessWriteLease:
    from consolidation_memory.config import get_config

    cfg = get_config()
    return ProcessWriteLease(
        cfg.EMBEDDING_CACHE_WRITE_LOCK_PATH,
        cfg.EMBEDDING_CACHE_WRITE_LOCK_TIMEOUT_SECONDS,
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint() -> dict[str, object]:
    from consolidation_memory.config import get_config

    cfg = get_config()
    return {
        "backend": cfg.EMBEDDING_BACKEND,
        "model": cfg.EMBEDDING_MODEL_NAME,
        "dimension": int(cfg.EMBEDDING_DIMENSION),
    }


def _namespace_dir(namespace: str) -> Path:
    from consolidation_memory.config import get_config

    return get_config().EMBEDDING_CACHE_DIR / namespace


def _disk_cache_enabled() -> bool:
    from consolidation_memory.config import get_config

    return bool(get_config().EMBEDDING_DISK_CACHE_ENABLED)


def clear_namespace(namespace: str) -> None:
    """Drop in-memory and on-disk vectors for one cache namespace."""
    with _lock:
        _runtime_stores.pop(namespace, None)
    if not _disk_cache_enabled():
        return
    path = _namespace_dir(namespace)
    if not path.exists():
        return
    with _disk_write_lease().acquire():
        shutil.rmtree(path, ignore_errors=True)


def clear_all() -> None:
    for namespace in ("records", "topics", "claims"):
        clear_namespace(namespace)


def _load_runtime_store(namespace: str) -> dict[str, tuple[str, np.ndarray]]:
    with _lock:
        cached = _runtime_stores.get(namespace)
        if cached is not None:
            return cached

    store: dict[str, tuple[str, np.ndarray]] = {}
    if _disk_cache_enabled():
        store = _load_disk_store(namespace)
    with _lock:
        _runtime_stores[namespace] = store
        return store


def _load_disk_store(namespace: str) -> dict[str, tuple[str, np.ndarray]]:
    base = _namespace_dir(namespace)
    meta_path = base / "meta.json"
    vectors_path = base / "vectors.npz"
    if not meta_path.exists() or not vectors_path.exists():
        return {}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != _fingerprint():
            logger.info(
                "Embedding disk cache fingerprint mismatch for %s; rebuilding",
                namespace,
            )
            return {}

        payload = np.load(vectors_path, allow_pickle=False)
        ids = payload["ids"]
        hashes = payload["hashes"]
        vecs = payload["vecs"]
        if len(ids) != len(hashes) or len(ids) != len(vecs):
            logger.warning("Embedding disk cache shape mismatch for %s; rebuilding", namespace)
            return {}

        store: dict[str, tuple[str, np.ndarray]] = {}
        for item_id, content_hash, vector in zip(ids, hashes, vecs, strict=True):
            store[str(item_id)] = (str(content_hash), np.asarray(vector, dtype=np.float32))
        logger.debug("Loaded %d cached embeddings for namespace=%s", len(store), namespace)
        return store
    except Exception as exc:
        logger.warning("Failed to load embedding disk cache for %s: %s", namespace, exc)
        return {}


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _save_disk_store(namespace: str, store: dict[str, tuple[str, np.ndarray]]) -> None:
    if not _disk_cache_enabled() or not store:
        return

    with _disk_write_lease().acquire():
        base = _namespace_dir(namespace)
        base.mkdir(parents=True, exist_ok=True)

        ids = np.array(sorted(store.keys()), dtype=str)
        hashes = np.array([store[item_id][0] for item_id in ids], dtype=str)
        vecs = np.stack([store[item_id][1] for item_id in ids]).astype(np.float32, copy=False)

        meta = {"fingerprint": _fingerprint(), "count": int(len(ids))}
        _atomic_write_bytes(
            base / "meta.json",
            json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"),
        )

        buffer = io.BytesIO()
        np.savez_compressed(buffer, ids=ids, hashes=hashes, vecs=vecs)
        _atomic_write_bytes(base / "vectors.npz", buffer.getvalue())


def embed_items_incremental(
    items: Iterable[tuple[str, str]],
    *,
    namespace: str,
    retain_ids: set[str] | None = None,
) -> np.ndarray | None:
    """Return an embedding matrix aligned with ``items`` using disk-backed cache.

    Each item is ``(stable_id, embedding_text)``. Missing or stale entries are
    embedded in batches; unchanged entries are reused from memory/disk.
    """
    ordered_items = [(str(item_id), text) for item_id, text in items]
    if not ordered_items:
        return None

    valid_items = [(item_id, text) for item_id, text in ordered_items if text.strip()]
    if not valid_items:
        return None

    store = _load_runtime_store(namespace)
    missing: list[tuple[int, str, str]] = []
    for index, (item_id, text) in enumerate(ordered_items):
        if not text.strip():
            continue
        content_hash = _hash_text(text)
        cached = store.get(item_id)
        if cached is None or cached[0] != content_hash:
            missing.append((index, item_id, text))

    if missing:
        from consolidation_memory.backends import encode_documents

        texts_to_embed = [text for _, _, text in missing]
        try:
            fresh_vecs = encode_documents(texts_to_embed)
        except Exception as exc:
            logger.warning(
                "Incremental embedding failed for namespace=%s: %s",
                namespace,
                exc,
                exc_info=True,
            )
            return None

        if fresh_vecs.shape[0] != len(missing):
            logger.warning(
                "Incremental embedding returned unexpected shape for namespace=%s",
                namespace,
            )
            return None

        changed = False
        for (index, item_id, text), vector in zip(missing, fresh_vecs, strict=True):
            content_hash = _hash_text(text)
            store[item_id] = (content_hash, np.asarray(vector, dtype=np.float32))
            changed = True

        if retain_ids is not None:
            stale_ids = [item_id for item_id in store if item_id not in retain_ids]
            for item_id in stale_ids:
                store.pop(item_id, None)
            if stale_ids:
                changed = True

        if changed:
            with _lock:
                _runtime_stores[namespace] = store
            try:
                _save_disk_store(namespace, store)
            except Exception as exc:
                logger.warning(
                    "Failed to persist embedding disk cache for namespace=%s: %s",
                    namespace,
                    exc,
                )

    rows: list[np.ndarray] = []
    for item_id, text in ordered_items:
        if not text.strip():
            rows.append(np.zeros((get_dimension(),), dtype=np.float32))
            continue
        cached = store.get(item_id)
        if cached is None:
            return None
        rows.append(cached[1])

    return np.stack(rows).astype(np.float32, copy=False)


def get_dimension() -> int:
    from consolidation_memory.backends import get_dimension as backend_dimension

    return int(backend_dimension())