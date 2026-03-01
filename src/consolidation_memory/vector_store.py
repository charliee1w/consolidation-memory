"""FAISS vector index wrapper with UUID mapping.

Thread-safe via threading.Lock (MCP server is async but FAISS is not thread-safe).
Uses IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).

Persistence uses atomic write-then-rename: FAISS binary and JSON id map are
written to temp files first, then renamed over the originals. If the process
crashes mid-write, the previous valid files remain intact.

Deletion uses tombstoning: removed UUIDs are added to a set and filtered out
during search. Periodic compaction (during consolidation) rebuilds the index
without tombstoned vectors.
"""

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

import faiss
import numpy as np

from consolidation_memory.config import get_config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: Any = None  # faiss.Index, initialized in _load_or_create
        self._id_map: list[str] = []
        self._uuid_to_pos: dict[str, int] = {}
        self._tombstones: set[str] = set()
        self._last_load_time: float = 0.0
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing FAISS index and id map from disk, or create empty. Validates dimension and id-map integrity on load."""
        cfg = get_config()
        if cfg.FAISS_INDEX_PATH.exists() and cfg.FAISS_ID_MAP_PATH.exists():
            logger.info("Loading FAISS index from %s", cfg.FAISS_INDEX_PATH)
            self._index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))
            with open(cfg.FAISS_ID_MAP_PATH, "r") as f:
                self._id_map = json.load(f)
            self._uuid_to_pos = {uid: i for i, uid in enumerate(self._id_map)}

            if self._index.ntotal != len(self._id_map):
                if len(self._id_map) > self._index.ntotal:
                    # More IDs than vectors — likely a crash between id-map
                    # and index renames. Truncate the id-map to match.
                    logger.warning(
                        "FAISS/id_map mismatch: %d vectors vs %d ids. "
                        "Truncating id-map to match index (safe recovery).",
                        self._index.ntotal, len(self._id_map),
                    )
                    self._id_map = self._id_map[:self._index.ntotal]
                    self._uuid_to_pos = {uid: i for i, uid in enumerate(self._id_map)}
                    self._save()
                else:
                    # Fewer IDs than vectors — data loss, rebuild empty.
                    logger.error(
                        "FAISS/id_map mismatch: %d vectors vs %d ids. "
                        "Rebuilding empty index.",
                        self._index.ntotal, len(self._id_map),
                    )
                    self._index = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
                    self._id_map = []
                    self._uuid_to_pos = {}
                    self._tombstones = set()
                    self._save()
                    self._save_tombstones()
            else:
                logger.info("Loaded %d vectors", self._index.ntotal)
                if (
                    isinstance(self._index, faiss.IndexFlatIP)
                    and self._index.ntotal >= cfg.FAISS_SIZE_WARNING_THRESHOLD
                ):
                    logger.warning(
                        "FAISS index has %d vectors (threshold: %d). "
                        "Auto-upgrade to IndexIVFFlat will trigger on next add.",
                        self._index.ntotal, cfg.FAISS_SIZE_WARNING_THRESHOLD,
                    )

            if self._index.ntotal > 0 and self._index.d != cfg.EMBEDDING_DIMENSION:
                logger.error(
                    "FAISS dimension mismatch: index=%d, config=%d. Rebuilding empty index.",
                    self._index.d, cfg.EMBEDDING_DIMENSION,
                )
                self._index = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
                self._id_map = []
                self._uuid_to_pos = {}
                self._tombstones = set()
                self._save()
                self._save_tombstones()
        else:
            logger.info("Creating new FAISS index (dim=%d)", cfg.EMBEDDING_DIMENSION)
            self._index = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
            self._id_map = []
            self._uuid_to_pos = {}
            self._tombstones = set()

        if cfg.FAISS_TOMBSTONE_PATH.exists():
            try:
                with open(cfg.FAISS_TOMBSTONE_PATH, "r") as f:
                    self._tombstones = set(json.load(f))
                if self._tombstones:
                    logger.info("Loaded %d tombstones", len(self._tombstones))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load tombstones, starting fresh: %s", e)
                self._tombstones = set()

        self._last_load_time = time.time()

    def _save(self) -> None:
        """Atomic save: write to temp files, then rename over originals.

        Rename order matters for crash safety: the id-map is renamed first
        because it is the source of truth. If a crash occurs between the two
        renames, the mismatch detector in ``_load_or_create`` will find more
        IDs than vectors, which is recoverable (extra IDs are harmless —
        they map to positions that don't exist and are skipped). The reverse
        (fewer IDs than vectors) would silently lose mappings.
        """
        cfg = get_config()
        parent = cfg.FAISS_INDEX_PATH.parent
        parent.mkdir(parents=True, exist_ok=True)

        # Write both files to temp paths first, then rename in safe order.
        idx_fd, idx_tmp = tempfile.mkstemp(dir=str(parent), suffix=".faiss.tmp")
        os.close(idx_fd)
        try:
            faiss.write_index(self._index, idx_tmp)
        except Exception:
            os.unlink(idx_tmp)
            raise

        map_fd, map_tmp = tempfile.mkstemp(dir=str(parent), suffix=".json.tmp")
        try:
            with os.fdopen(map_fd, "w") as f:
                json.dump(self._id_map, f)
        except Exception:
            os.unlink(idx_tmp)
            os.unlink(map_tmp)
            raise

        # Rename id-map first (source of truth), then index.
        os.replace(map_tmp, str(cfg.FAISS_ID_MAP_PATH))
        os.replace(idx_tmp, str(cfg.FAISS_INDEX_PATH))

    def _save_tombstones(self) -> None:
        """Atomic save of tombstone set."""
        cfg = get_config()
        parent = cfg.FAISS_TOMBSTONE_PATH.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(list(self._tombstones), f)
        except Exception:
            os.unlink(tmp)
            raise
        os.replace(tmp, str(cfg.FAISS_TOMBSTONE_PATH))

    # ── Index upgrade ────────────────────────────────────────────────────────

    def _maybe_upgrade_index(self, force: bool = False) -> bool:
        """Upgrade from IndexFlatIP to IndexIVFFlat when index exceeds threshold.

        Must be called with ``self._lock`` held. Extracts all vectors from the
        current flat index, trains an IVF index, adds the vectors, enables
        direct map for reconstruct support, and atomically saves to disk.

        Args:
            force: If True, skip the threshold check (used after compaction
                   to restore an IVF index that was temporarily flattened).

        Returns True if upgrade was performed, False otherwise.
        If training fails, keeps the existing IndexFlatIP and logs a warning.
        """
        if not isinstance(self._index, faiss.IndexFlatIP):
            return False
        if not force and self._index.ntotal < get_config().FAISS_IVF_UPGRADE_THRESHOLD:
            return False

        n = self._index.ntotal
        dim = self._index.d
        nlist = max(1, min(int(n ** 0.5), 4096))

        logger.info(
            "Upgrading FAISS index from IndexFlatIP to IndexIVFFlat "
            "(n=%d, nlist=%d, dim=%d)",
            n, nlist, dim,
        )

        try:
            # Bulk extract all vectors from flat index via internal storage pointer.
            # This is O(1) vs O(n) individual reconstruct() calls.
            vectors = faiss.rev_swig_ptr(self._index.get_xb(), n * dim).reshape(n, dim).copy()

            # Create and train IVF index
            quantizer = faiss.IndexFlatIP(dim)
            ivf_index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT,
            )
            ivf_index.train(vectors)
            ivf_index.add(vectors)

            # Enable reconstruct support and set search quality
            ivf_index.make_direct_map()
            ivf_index.nprobe = max(1, min(nlist // 4, 64))

            # Atomic swap
            self._index = ivf_index
            self._save()

            logger.info(
                "FAISS index upgrade complete: IndexIVFFlat with "
                "nlist=%d, nprobe=%d",
                nlist, ivf_index.nprobe,
            )
            return True

        except Exception:
            logger.warning(
                "FAISS index upgrade to IndexIVFFlat failed, keeping IndexFlatIP",
                exc_info=True,
            )
            return False

    # ── Concurrency ──────────────────────────────────────────────────────────

    def reload_if_stale(self) -> bool:
        """Reload FAISS index if the reload signal file is newer than last load. Thread-safe via double-checked lock. Returns True if reloaded."""
        cfg = get_config()
        if not cfg.FAISS_RELOAD_SIGNAL.exists():
            return False
        try:
            signal_mtime = cfg.FAISS_RELOAD_SIGNAL.stat().st_mtime
        except OSError:
            return False
        if signal_mtime <= self._last_load_time:
            return False
        with self._lock:
            try:
                signal_mtime = cfg.FAISS_RELOAD_SIGNAL.stat().st_mtime
            except OSError:
                return False
            if signal_mtime <= self._last_load_time:
                return False
            logger.info("FAISS reload signal detected, reloading index...")
            self._load_or_create()
            return True

    @staticmethod
    def signal_reload() -> None:
        """Write reload signal file to notify other processes to reload FAISS index."""
        cfg = get_config()
        cfg.FAISS_RELOAD_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
        cfg.FAISS_RELOAD_SIGNAL.write_text(str(time.time()), encoding="utf-8")
        logger.info("Wrote FAISS reload signal")

    # ── Add ──────────────────────────────────────────────────────────────────

    def add(self, episode_id: str, embedding: np.ndarray) -> None:
        """Add a single vector with its episode UUID. Persists to disk immediately."""
        with self._lock:
            vec = embedding.reshape(1, -1).astype(np.float32)
            self._index.add(vec)
            self._id_map.append(episode_id)
            self._uuid_to_pos[episode_id] = len(self._id_map) - 1
            self._save()
            self._maybe_upgrade_index()

    def add_batch(self, episode_ids: list[str], embeddings: np.ndarray) -> None:
        """Add multiple vectors with UUIDs. More efficient than repeated add() calls."""
        with self._lock:
            vecs = embeddings.astype(np.float32)
            self._index.add(vecs)
            start = len(self._id_map)
            self._id_map.extend(episode_ids)
            for i, uid in enumerate(episode_ids):
                self._uuid_to_pos[uid] = start + i
            self._save()
            self._maybe_upgrade_index()

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (uuid, similarity) pairs via cosine similarity, excluding tombstoned entries."""
        with self._lock:
            if self._index.ntotal == 0:
                logger.debug("search: index empty, returning []")
                return []
            effective_size = self._index.ntotal - len(self._tombstones)
            if effective_size <= 0:
                logger.debug("search: all %d vectors tombstoned, returning []", self._index.ntotal)
                return []
            # Over-fetch to compensate for tombstoned vectors being filtered out.
            # Use configured padding if set, otherwise auto-scale based on
            # tombstone ratio. Capped at k*3 to avoid fetching the entire index
            # when tombstone counts are pathologically high (>50% of index).
            # At extreme ratios, this may return fewer than k results —
            # compaction should have triggered well before that point.
            cfg = get_config()
            if cfg.FAISS_SEARCH_FETCH_K_PADDING > 0:
                fetch_k = min(k + cfg.FAISS_SEARCH_FETCH_K_PADDING, self._index.ntotal)
            else:
                auto_padding = min(len(self._tombstones), k * 3)
                fetch_k = min(k + auto_padding, self._index.ntotal)
            # Absolute cap: never fetch more than max(k*3, 200) to prevent
            # pathological cases when callers request large k with filters.
            fetch_k = min(fetch_k, max(k * 3, 200), self._index.ntotal)
            logger.debug(
                "search: ntotal=%d, tombstones=%d, effective=%d, k=%d, fetch_k=%d",
                self._index.ntotal, len(self._tombstones), effective_size, k, fetch_k,
            )
            vec = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(vec, fetch_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                uid = self._id_map[idx]
                if uid in self._tombstones:
                    continue
                results.append((uid, float(score)))
                if len(results) >= k:
                    break
            if results:
                logger.debug(
                    "search: returning %d results, top similarity=%.4f",
                    len(results), results[0][1],
                )
            else:
                logger.debug("search: no results survived tombstone filtering")
            return results

    # ── Remove (tombstone-based, O(1)) ───────────────────────────────────────

    def remove(self, episode_id: str) -> bool:
        """Tombstone a vector by UUID. O(1), does not rebuild index."""
        with self._lock:
            if episode_id not in self._uuid_to_pos:
                return False
            self._tombstones.add(episode_id)
            self._save_tombstones()
            return True

    def remove_batch(self, episode_ids: list[str]) -> int:
        """Tombstone multiple vectors by UUID. Returns count actually tombstoned."""
        with self._lock:
            count = 0
            for uid in episode_ids:
                if uid in self._uuid_to_pos and uid not in self._tombstones:
                    self._tombstones.add(uid)
                    count += 1
            if count > 0:
                self._save_tombstones()
            return count

    # ── Compaction ───────────────────────────────────────────────────────────

    def compact(self) -> int:
        """Rebuild FAISS index without tombstoned vectors. Returns count of tombstones removed."""
        with self._lock:
            if not self._tombstones:
                return 0
            removed = len(self._tombstones)
            cfg = get_config()
            # Remember if the index was IVF before compaction so we can restore it
            was_ivf = not isinstance(self._index, faiss.IndexFlatIP)

            keep_positions = [
                i for i, uid in enumerate(self._id_map)
                if uid not in self._tombstones
            ]

            if not keep_positions:
                self._index = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
                self._id_map = []
                self._uuid_to_pos = {}
                self._tombstones = set()
                self._save()
                self._save_tombstones()
                return removed

            if isinstance(self._index, faiss.IndexFlatIP):
                # Bulk extract from flat index, then select kept positions
                n = self._index.ntotal
                dim = self._index.d
                all_vectors = faiss.rev_swig_ptr(self._index.get_xb(), n * dim).reshape(n, dim)
                kept_vectors = all_vectors[keep_positions].copy()
            else:
                # IVF index: reconstruct individually (no get_xb)
                kept_vectors = np.zeros(
                    (len(keep_positions), cfg.EMBEDDING_DIMENSION), dtype=np.float32
                )
                for new_i, old_i in enumerate(keep_positions):
                    kept_vectors[new_i] = self._index.reconstruct(old_i)

            new_id_map = [self._id_map[i] for i in keep_positions]
            self._index = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
            self._index.add(kept_vectors)
            self._id_map = new_id_map
            self._uuid_to_pos = {uid: i for i, uid in enumerate(self._id_map)}
            self._tombstones = set()
            self._save()
            self._save_tombstones()
            logger.info("Compacted FAISS index: removed %d tombstoned vectors", removed)
            # If the index was IVF before, force upgrade regardless of threshold
            if was_ivf and len(keep_positions) >= 100:
                self._maybe_upgrade_index(force=True)
            else:
                self._maybe_upgrade_index()
            return removed

    @property
    def tombstone_ratio(self) -> float:
        with self._lock:
            total = self._index.ntotal if self._index else 0
            return len(self._tombstones) / total if total > 0 else 0.0

    # ── Reconstruct ──────────────────────────────────────────────────────────

    def reconstruct_batch(self, episode_ids: list[str]) -> tuple[list[str], np.ndarray] | None:
        """Retrieve raw vectors for given episode UUIDs from FAISS. Skips tombstoned or missing entries."""
        with self._lock:
            found_ids = []
            positions = []
            for uid in episode_ids:
                if uid in self._uuid_to_pos and uid not in self._tombstones:
                    found_ids.append(uid)
                    positions.append(self._uuid_to_pos[uid])
            if not positions:
                return None
            if isinstance(self._index, faiss.IndexFlatIP):
                n = self._index.ntotal
                dim = self._index.d
                all_vectors = faiss.rev_swig_ptr(self._index.get_xb(), n * dim).reshape(n, dim)
                vectors = all_vectors[positions].copy()
            else:
                vectors = np.zeros((len(positions), get_config().EMBEDDING_DIMENSION), dtype=np.float32)
                for i, pos in enumerate(positions):
                    vectors[i] = self._index.reconstruct(pos)
            return found_ids, vectors

    @property
    def tombstone_count(self) -> int:
        """Number of tombstoned (soft-deleted) vectors."""
        with self._lock:
            return len(self._tombstones)

    @property
    def size(self) -> int:
        with self._lock:
            total = self._index.ntotal if self._index else 0
            return total - len(self._tombstones)
