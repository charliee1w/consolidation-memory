"""Maintenance operations callable from CLI and browser UI."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np


def warmup_recall_caches(*, include_claims: bool = False) -> dict[str, Any]:
    """Pre-build recall embedding caches (topics, records, optional claims)."""
    import time

    from consolidation_memory.client import MemoryClient
    from consolidation_memory.config import get_active_project, override_config
    from consolidation_memory.tool_adapter import warm_recall_caches as _warm

    started = time.perf_counter()
    client = MemoryClient()
    try:
        if include_claims:
            with override_config(WARMUP_PRIME_CLAIM_CACHE=True):
                stats = _warm(client)
        else:
            stats = _warm(client)
    finally:
        client.close()

    elapsed_seconds = round(time.perf_counter() - started, 3)
    return {
        "status": "ok",
        "project": get_active_project(),
        "elapsed_seconds": elapsed_seconds,
        **stats,
    }


def reindex_all_episodes() -> dict[str, Any]:
    """Re-embed all episodes with the current backend and rebuild FAISS."""
    from consolidation_memory.backends import encode_documents, get_dimension
    from consolidation_memory.config import get_config
    from consolidation_memory.database import ensure_schema, get_all_episodes
    from consolidation_memory.episode_embedding import embedding_text_for_episode_row
    from consolidation_memory.vector_store import VectorStore

    try:
        import faiss
    except ImportError as exc:
        return {
            "status": "error",
            "message": "Reindex requires faiss-cpu. Install with: pip install consolidation-memory[fastembed]",
            "error": str(exc),
        }

    cfg = get_config()
    ensure_schema()
    episodes = get_all_episodes(include_deleted=False)
    if not episodes:
        return {"status": "ok", "episodes_reindexed": 0, "message": "No episodes to reindex."}

    dim = get_dimension()
    batch_size = 50
    all_ids: list[str] = []
    all_vecs: list[np.ndarray] = []
    failed_batches = 0

    for index in range(0, len(episodes), batch_size):
        batch = episodes[index : index + batch_size]
        texts = [embedding_text_for_episode_row(ep) for ep in batch]
        try:
            vecs = encode_documents(texts)
            all_ids.extend(ep["id"] for ep in batch)
            all_vecs.append(vecs)
        except Exception:
            failed_batches += 1

    if not all_vecs:
        return {
            "status": "error",
            "message": "No embeddings produced. Check embedding backend connectivity.",
            "failed_batches": failed_batches,
        }

    all_vecs_arr = np.vstack(all_vecs)
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(all_vecs_arr)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    parent = str(cfg.FAISS_INDEX_PATH.parent)

    idx_fd, idx_tmp = tempfile.mkstemp(dir=parent, suffix=".faiss.tmp")
    os.close(idx_fd)
    try:
        faiss.write_index(faiss_index, idx_tmp)
    except Exception as exc:
        os.unlink(idx_tmp)
        return {"status": "error", "message": "Failed to write new FAISS index.", "error": str(exc)}

    map_fd, map_tmp = tempfile.mkstemp(dir=parent, suffix=".json.tmp")
    try:
        with os.fdopen(map_fd, "w", encoding="utf-8") as handle:
            json.dump(all_ids, handle)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except Exception as exc:
        os.unlink(idx_tmp)
        os.unlink(map_tmp)
        return {"status": "error", "message": "Failed to write id map.", "error": str(exc)}

    tomb_fd, tomb_tmp = tempfile.mkstemp(dir=parent, suffix=".json.tmp")
    try:
        with os.fdopen(tomb_fd, "w", encoding="utf-8") as handle:
            json.dump([], handle)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except Exception as exc:
        os.unlink(idx_tmp)
        os.unlink(map_tmp)
        os.unlink(tomb_tmp)
        return {"status": "error", "message": "Failed to write tombstones.", "error": str(exc)}

    os.replace(map_tmp, str(cfg.FAISS_ID_MAP_PATH))
    os.replace(idx_tmp, str(cfg.FAISS_INDEX_PATH))
    os.replace(tomb_tmp, str(cfg.FAISS_TOMBSTONE_PATH))

    VectorStore.signal_reload()
    VectorStore()._save_embedding_metadata()

    return {
        "status": "ok",
        "episodes_reindexed": len(all_ids),
        "embedding_dimension": dim,
        "failed_batches": failed_batches,
        "message": f"Reindexed {len(all_ids)} episodes.",
    }