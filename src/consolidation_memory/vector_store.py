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

import faiss
import numpy as np

from consolidation_memory.config import (
    FAISS_INDEX_PATH,
    FAISS_ID_MAP_PATH,
    FAISS_TOMBSTONE_PATH,
    FAISS_RELOAD_SIGNAL,
    EMBEDDING_DIMENSION,
    FAISS_SEARCH_FETCH_K_PADDING,
    FAISS_SIZE_WARNING_THRESHOLD,
)

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._index: faiss.IndexFlatIP | None = None
        self._id_map: list[str] = []
        self._uuid_to_pos: dict[str, int] = {}
        self._tombstones: set[str] = set()
        self._last_load_time: float = 0.0
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing FAISS index and id map from disk, or create empty. Validates dimension and id-map integrity on load."""
        if FAISS_INDEX_PATH.exists() and FAISS_ID_MAP_PATH.exists():
            logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
            self._index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(FAISS_ID_MAP_PATH, "r") as f:
                self._id_map = json.load(f)
            self._uuid_to_pos = {uid: i for i, uid in enumerate(self._id_map)}

            if self._index.ntotal != len(self._id_map):
                logger.error(
                    "FAISS/id_map mismatch: %d vectors vs %d ids. Rebuilding empty index.",
                    self._index.ntotal, len(self._id_map),
                )
                self._index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
                self._id_map = []
                self._uuid_to_pos = {}
                self._tombstones = set()
                self._save()
                self._save_tombstones()
            else:
                logger.info("Loaded %d vectors", self._index.ntotal)
                if self._index.ntotal >= FAISS_SIZE_WARNING_THRESHOLD:
                    logger.warning(
                        "FAISS index has %d vectors (threshold: %d). "
                        "Consider migrating from IndexFlatIP to IndexIVFFlat.",
                        self._index.ntotal, FAISS_SIZE_WARNING_THRESHOLD,
                    )

            if self._index.ntotal > 0 and self._index.d != EMBEDDING_DIMENSION:
                logger.error(
                    "FAISS dimension mismatch: index=%d, config=%d. Rebuilding empty index.",
                    self._index.d, EMBEDDING_DIMENSION,
                )
                self._index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
                self._id_map = []
                self._uuid_to_pos = {}
                self._tombstones = set()
                self._save()
                self._save_tombstones()
        else:
            logger.info("Creating new FAISS index (dim=%d)", EMBEDDING_DIMENSION)
            self._index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self._id_map = []
            self._uuid_to_pos = {}
            self._tombstones = set()

        if FAISS_TOMBSTONE_PATH.exists():
            try:
                with open(FAISS_TOMBSTONE_PATH, "r") as f:
                    self._tombstones = set(json.load(f))
                if self._tombstones:
                    logger.info("Loaded %d tombstones", len(self._tombstones))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load tombstones, starting fresh: %s", e)
                self._tombstones = set()

        self._last_load_time = time.time()

    def _save(self) -> None:
        """Atomic save: write to temp files, then rename over originals."""
        parent = FAISS_INDEX_PATH.parent
        parent.mkdir(parents=True, exist_ok=True)

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

        os.replace(idx_tmp, str(FAISS_INDEX_PATH))
        os.replace(map_tmp, str(FAISS_ID_MAP_PATH))

    def _save_tombstones(self) -> None:
        """Atomic save of tombstone set."""
        parent = FAISS_TOMBSTONE_PATH.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(list(self._tombstones), f)
        except Exception:
            os.unlink(tmp)
            raise
        os.replace(tmp, str(FAISS_TOMBSTONE_PATH))

    # ── Concurrency ──────────────────────────────────────────────────────────

    def reload_if_stale(self) -> bool:
        """Reload FAISS index if the reload signal file is newer than last load. Thread-safe via double-checked lock. Returns True if reloaded."""
        if not FAISS_RELOAD_SIGNAL.exists():
            return False
        try:
            signal_mtime = FAISS_RELOAD_SIGNAL.stat().st_mtime
        except OSError:
            return False
        if signal_mtime <= self._last_load_time:
            return False
        with self._lock:
            try:
                signal_mtime = FAISS_RELOAD_SIGNAL.stat().st_mtime
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
        FAISS_RELOAD_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
        FAISS_RELOAD_SIGNAL.write_text(str(time.time()), encoding="utf-8")
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
            if FAISS_SEARCH_FETCH_K_PADDING > 0:
                fetch_k = min(k + FAISS_SEARCH_FETCH_K_PADDING, self._index.ntotal)
            else:
                auto_padding = min(len(self._tombstones), k * 3)
                fetch_k = min(k + auto_padding, self._index.ntotal)
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

            keep_positions = [
                i for i, uid in enumerate(self._id_map)
                if uid not in self._tombstones
            ]

            if not keep_positions:
                self._index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
                self._id_map = []
                self._uuid_to_pos = {}
                self._tombstones = set()
                self._save()
                self._save_tombstones()
                return removed

            kept_vectors = np.zeros(
                (len(keep_positions), EMBEDDING_DIMENSION), dtype=np.float32
            )
            for new_i, old_i in enumerate(keep_positions):
                kept_vectors[new_i] = self._index.reconstruct(old_i)

            new_id_map = [self._id_map[i] for i in keep_positions]
            self._index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self._index.add(kept_vectors)
            self._id_map = new_id_map
            self._uuid_to_pos = {uid: i for i, uid in enumerate(self._id_map)}
            self._tombstones = set()
            self._save()
            self._save_tombstones()
            logger.info("Compacted FAISS index: removed %d tombstoned vectors", removed)
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
            vectors = np.zeros((len(positions), EMBEDDING_DIMENSION), dtype=np.float32)
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
