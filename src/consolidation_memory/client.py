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
import threading
from datetime import datetime, timezone

from consolidation_memory import __version__
from consolidation_memory.types import (
    ContentType,
    StoreResult,
    BatchStoreResult,
    RecallResult,
    SearchResult,
    ForgetResult,
    StatusResult,
    ExportResult,
    CorrectResult,
    CompactResult,
)

logger = logging.getLogger("consolidation_memory")


class MemoryClient:
    """Persistent semantic memory client.

    Owns the vector store, database lifecycle, and optional background
    consolidation thread.  All public methods are synchronous and thread-safe.

    Args:
        auto_consolidate: Start background consolidation thread. Default True
            (respects ``CONSOLIDATION_AUTO_RUN`` config).
    """

    def __init__(self, auto_consolidate: bool = True) -> None:
        from consolidation_memory.config import (
            DATA_DIR,
            KNOWLEDGE_DIR,
            CONSOLIDATION_LOG_DIR,
            LOG_DIR,
            BACKUP_DIR,
            CONSOLIDATION_AUTO_RUN,
        )
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.vector_store import VectorStore

        # Ensure directories
        for d in [DATA_DIR, KNOWLEDGE_DIR, CONSOLIDATION_LOG_DIR, LOG_DIR, BACKUP_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        ensure_schema()
        self._vector_store = VectorStore()
        self._check_embedding_backend()

        # Consolidation threading
        self._consolidation_lock = threading.Lock()
        self._consolidation_stop = threading.Event()
        self._consolidation_thread: threading.Thread | None = None

        if auto_consolidate and CONSOLIDATION_AUTO_RUN:
            self._start_consolidation_thread()

        logger.info(
            "MemoryClient initialized (vectors=%d, version=%s)",
            self._vector_store.size,
            __version__,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop background threads. Call when done."""
        self._consolidation_stop.set()
        if self._consolidation_thread is not None:
            self._consolidation_thread.join(timeout=30)
            if self._consolidation_thread.is_alive():
                logger.warning(
                    "Consolidation thread did not stop within 30s; "
                    "it will terminate when the process exits (daemon thread)."
                )
            self._consolidation_thread = None

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        content_type: str = "exchange",
        tags: list[str] | None = None,
        surprise: float = 0.5,
    ) -> StoreResult:
        """Store a memory episode.

        Args:
            content: Text content to store.
            content_type: One of 'exchange', 'fact', 'solution', 'preference'.
            tags: Optional topic tags.
            surprise: Novelty score 0.0–1.0.

        Returns:
            StoreResult with status 'stored' or 'duplicate_detected'.
        """
        from consolidation_memory.database import insert_episode, get_episode, soft_delete_episode
        from consolidation_memory.backends import encode_documents
        from consolidation_memory.config import DEDUP_SIMILARITY_THRESHOLD, DEDUP_ENABLED

        self._vector_store.reload_if_stale()

        try:
            embedding = encode_documents([content])
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during store: %s", e)
            return StoreResult(
                status="backend_unavailable",
                message=f"Embedding backend unreachable: {e}",
            )

        # Dedup check
        if DEDUP_ENABLED and self._vector_store.size > 0:
            matches = self._vector_store.search(embedding[0], k=1)
            if matches:
                best_id, best_sim = matches[0]
                if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
                    existing = get_episode(best_id)
                    if existing is not None:
                        logger.info(
                            "Duplicate detected (sim=%.4f >= %.2f): existing=%s",
                            best_sim, DEDUP_SIMILARITY_THRESHOLD, best_id,
                        )
                        return StoreResult(
                            status="duplicate_detected",
                            existing_id=best_id,
                            similarity=round(best_sim, 4),
                            message="Content too similar to existing episode. Not stored.",
                        )

        # Validate content_type
        valid_types = {ct.value for ct in ContentType}
        if content_type not in valid_types:
            logger.warning("Invalid content_type %r, defaulting to 'exchange'", content_type)
            content_type = ContentType.EXCHANGE.value

        episode_id = insert_episode(
            content=content,
            content_type=content_type,
            tags=tags,
            surprise_score=max(0.0, min(1.0, surprise)),
        )

        try:
            self._vector_store.add(episode_id, embedding[0])
        except Exception as e:
            # Rollback: remove the DB record if FAISS add fails to prevent orphans
            soft_delete_episode(episode_id)
            logger.error("FAISS add failed for %s, rolled back DB insert: %s", episode_id, e)
            raise

        logger.info(
            "Stored episode %s (type=%s, surprise=%.2f, tags=%s)",
            episode_id, content_type, surprise, tags,
        )
        return StoreResult(
            status="stored",
            id=episode_id,
            content_type=content_type,
            tags=tags or [],
        )

    def store_batch(
        self,
        episodes: list[dict],
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
        from consolidation_memory.database import insert_episode, get_episode, soft_delete_episode
        from consolidation_memory.backends import encode_documents
        from consolidation_memory.config import DEDUP_SIMILARITY_THRESHOLD, DEDUP_ENABLED

        if not episodes:
            return BatchStoreResult(status="stored", stored=0, duplicates=0)

        self._vector_store.reload_if_stale()

        # Validate and normalize
        valid_types = {ct.value for ct in ContentType}
        items = []
        for ep in episodes:
            ct = ep.get("content_type", "exchange")
            if ct not in valid_types:
                ct = ContentType.EXCHANGE.value
            items.append({
                "content": ep["content"],
                "content_type": ct,
                "tags": ep.get("tags"),
                "surprise": max(0.0, min(1.0, ep.get("surprise", 0.5))),
            })

        # Single embedding call for all texts
        try:
            embeddings = encode_documents([it["content"] for it in items])
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during batch store: %s", e)
            return BatchStoreResult(
                status="backend_unavailable",
                results=[{"status": "error", "message": str(e)}],
            )

        results = []
        stored = 0
        duplicates = 0

        for i, item in enumerate(items):
            emb = embeddings[i]

            # Dedup check
            if DEDUP_ENABLED and self._vector_store.size > 0:
                matches = self._vector_store.search(emb, k=1)
                if matches:
                    best_id, best_sim = matches[0]
                    if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
                        existing = get_episode(best_id)
                        if existing is not None:
                            duplicates += 1
                            results.append({
                                "status": "duplicate_detected",
                                "existing_id": best_id,
                                "similarity": round(float(best_sim), 4),
                            })
                            continue

            episode_id = insert_episode(
                content=item["content"],
                content_type=item["content_type"],
                tags=item["tags"],
                surprise_score=item["surprise"],
            )

            try:
                self._vector_store.add(episode_id, emb)
            except Exception as e:
                soft_delete_episode(episode_id)
                logger.error("FAISS add failed for %s in batch, rolled back: %s", episode_id, e)
                results.append({"status": "error", "message": str(e)})
                continue

            stored += 1
            results.append({
                "status": "stored",
                "id": episode_id,
                "content_type": item["content_type"],
            })

        logger.info("Batch store: %d stored, %d duplicates out of %d", stored, duplicates, len(items))
        return BatchStoreResult(status="stored", stored=stored, duplicates=duplicates, results=results)

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
    ) -> RecallResult:
        """Retrieve relevant memories by semantic similarity.

        Args:
            query: Natural language description of what to recall.
            n_results: Maximum episode results (1–50).
            include_knowledge: Include consolidated knowledge documents.
            content_types: Filter to specific content types (e.g. ['solution', 'fact']).
            tags: Filter to episodes with at least one matching tag.
            after: Only episodes created after this ISO date (e.g. '2025-01-01').
            before: Only episodes created before this ISO date.

        Returns:
            RecallResult with episodes and knowledge lists.
        """
        from consolidation_memory.context_assembler import recall
        from consolidation_memory.database import get_stats

        self._vector_store.reload_if_stale()

        try:
            result = recall(
                query=query,
                n_results=n_results,
                include_knowledge=include_knowledge,
                vector_store=self._vector_store,
                content_types=content_types,
                tags=tags,
                after=after,
                before=before,
            )
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during recall: %s", e)
            return RecallResult(
                message=f"Embedding backend unreachable: {e}",
            )

        stats = get_stats()

        logger.info(
            "Recall query='%s' returned %d episodes, %d knowledge entries",
            query[:80], len(result["episodes"]), len(result["knowledge"]),
        )

        return RecallResult(
            episodes=result["episodes"],
            knowledge=result["knowledge"],
            total_episodes=stats["episodic_buffer"]["total"],
            total_knowledge_topics=stats["knowledge_base"]["total_topics"],
            warnings=result.get("warnings", []),
        )

    def search(
        self,
        query: str | None = None,
        content_types: list[str] | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 20,
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
        import json
        from consolidation_memory.database import search_episodes

        results = search_episodes(
            query=query,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            limit=limit,
        )

        episodes = []
        for ep in results:
            ep_tags = json.loads(ep["tags"]) if isinstance(ep["tags"], str) else ep["tags"]
            episodes.append({
                "id": ep["id"],
                "content": ep["content"],
                "content_type": ep["content_type"],
                "tags": ep_tags,
                "created_at": ep["created_at"],
                "surprise_score": ep["surprise_score"],
                "access_count": ep["access_count"],
            })

        logger.info(
            "Search query=%r returned %d results",
            query, len(episodes),
        )
        return SearchResult(
            episodes=episodes,
            total_matches=len(episodes),
            query=query,
        )

    def status(self) -> StatusResult:
        """Get memory system statistics.

        Returns:
            StatusResult with counts, backend info, health, and last consolidation.
        """
        from consolidation_memory.database import get_stats, get_last_consolidation_run
        from consolidation_memory.config import (
            EMBEDDING_MODEL_NAME, EMBEDDING_BACKEND, DB_PATH,
            CONSOLIDATION_INTERVAL_HOURS, FAISS_COMPACTION_THRESHOLD,
        )

        stats = get_stats()
        last_run = get_last_consolidation_run()

        db_size_mb = 0.0
        if DB_PATH.exists():
            db_size_mb = round(DB_PATH.stat().st_size / (1024 * 1024), 2)

        health = self._compute_health(last_run, CONSOLIDATION_INTERVAL_HOURS, FAISS_COMPACTION_THRESHOLD)

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

        return StatusResult(
            episodic_buffer=stats["episodic_buffer"],
            knowledge_base=stats["knowledge_base"],
            last_consolidation=last_run,
            embedding_backend=EMBEDDING_BACKEND,
            embedding_model=EMBEDDING_MODEL_NAME,
            faiss_index_size=self._vector_store.size,
            faiss_tombstones=self._vector_store.tombstone_count,
            db_size_mb=db_size_mb,
            version=__version__,
            health=health,
            consolidation_metrics=metrics,
            consolidation_quality=quality_summary,
        )

    def forget(self, episode_id: str) -> ForgetResult:
        """Soft-delete an episode.

        Args:
            episode_id: UUID of the episode to forget.

        Returns:
            ForgetResult with status 'forgotten' or 'not_found'.
        """
        from consolidation_memory.database import soft_delete_episode

        self._vector_store.reload_if_stale()

        deleted = soft_delete_episode(episode_id)
        if deleted:
            self._vector_store.remove(episode_id)
            logger.info("Forgot episode %s", episode_id)
            return ForgetResult(status="forgotten", id=episode_id)
        else:
            logger.warning("Episode %s not found for deletion", episode_id)
            return ForgetResult(status="not_found", id=episode_id)

    def export(self) -> ExportResult:
        """Export all episodes and knowledge to a JSON snapshot.

        Returns:
            ExportResult with status, file path, and counts.
        """
        from consolidation_memory.config import BACKUP_DIR, KNOWLEDGE_DIR, MAX_BACKUPS
        from consolidation_memory.database import get_all_episodes, get_all_knowledge_topics

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        episodes = get_all_episodes(include_deleted=False)

        topics = get_all_knowledge_topics()
        knowledge = []
        for topic in topics:
            filepath = KNOWLEDGE_DIR / topic["filename"]
            content = ""
            if filepath.exists():
                content = filepath.read_text(encoding="utf-8")
            knowledge.append({**topic, "file_content": content})

        snapshot = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "episodes": episodes,
            "knowledge_topics": knowledge,
            "stats": {
                "episode_count": len(episodes),
                "knowledge_count": len(knowledge),
            },
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_path = BACKUP_DIR / f"memory_export_{timestamp}.json"
        export_path.write_text(
            json.dumps(snapshot, indent=2, default=str),
            encoding="utf-8",
        )

        # Prune old exports
        existing = sorted(BACKUP_DIR.glob("memory_export_*.json"), reverse=True)
        for old in existing[MAX_BACKUPS:]:
            old.unlink()

        logger.info(
            "Exported %d episodes + %d topics to %s",
            len(episodes), len(knowledge), export_path,
        )

        return ExportResult(
            status="exported",
            path=str(export_path),
            episodes=len(episodes),
            knowledge_topics=len(knowledge),
        )

    def correct(self, topic_filename: str, correction: str) -> CorrectResult:
        """Correct a knowledge document with new information.

        Args:
            topic_filename: Filename of the knowledge topic (e.g. 'vr_setup.md').
            correction: Description of what needs correcting and the correct info.

        Returns:
            CorrectResult with status and updated metadata.
        """
        from consolidation_memory.config import KNOWLEDGE_DIR
        from consolidation_memory.database import get_all_knowledge_topics, upsert_knowledge_topic
        from consolidation_memory.consolidation import (
            _version_knowledge_file, _parse_frontmatter, _count_facts, _normalize_output,
        )
        from consolidation_memory.backends import get_llm_backend

        filepath = KNOWLEDGE_DIR / topic_filename
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
            f"CORRECTION:\n{correction}\n\n"
            f"EXISTING DOCUMENT:\n{existing_content}\n\n"
            f"Output the complete corrected document with updated frontmatter "
            f"(title, summary, tags, confidence).\n"
            f"Do NOT wrap in code fences. Output raw markdown starting with --- frontmatter."
        )

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        from consolidation_memory.config import LLM_CORRECTION_TIMEOUT

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(llm.generate, system_prompt, user_prompt)
                raw = future.result(timeout=LLM_CORRECTION_TIMEOUT)
            corrected = _normalize_output(raw)
        except FuturesTimeoutError:
            return CorrectResult(
                status="error",
                message=f"LLM generation timed out after {LLM_CORRECTION_TIMEOUT}s. "
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
            logger.warning(
                "LLM correction output missing frontmatter title for %s; "
                "writing anyway but this may indicate a bad generation.",
                topic_filename,
            )

        _version_knowledge_file(filepath)
        filepath.write_text(corrected, encoding="utf-8")

        parsed = _parse_frontmatter(corrected)
        meta = parsed["meta"]

        # Update DB entry
        topics = get_all_knowledge_topics()
        for topic in topics:
            if topic["filename"] == topic_filename:
                upsert_knowledge_topic(
                    filename=topic_filename,
                    title=meta.get("title", topic["title"]),
                    summary=meta.get("summary", topic["summary"]),
                    source_episodes=[],
                    fact_count=_count_facts(corrected),
                    confidence=float(meta.get("confidence", topic["confidence"])),
                )
                break

        logger.info("Corrected knowledge topic: %s", topic_filename)

        return CorrectResult(
            status="corrected",
            filename=topic_filename,
            title=meta.get("title", ""),
        )

    def consolidate(self) -> dict:
        """Run consolidation manually. Thread-safe with background thread.

        Returns:
            Dict with consolidation results or ``{"status": "already_running"}``.
        """
        if not self._consolidation_lock.acquire(blocking=False):
            return {"status": "already_running"}
        try:
            from consolidation_memory.consolidation import run_consolidation
            return run_consolidation(vector_store=self._vector_store)
        finally:
            self._consolidation_lock.release()

    def compact(self) -> CompactResult:
        """Compact the FAISS index by removing tombstoned vectors.

        Returns:
            CompactResult with status, tombstones removed, and new index size.
        """
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

    def _compute_health(
        self,
        last_run: dict | None,
        interval_hours: float,
        compaction_threshold: float,
    ) -> dict:
        """Build health assessment dict."""
        issues: list[str] = []

        backend_reachable = self._probe_backend()
        if not backend_reachable:
            issues.append("Embedding backend unreachable")

        tombstone_ratio = self._vector_store.tombstone_ratio
        if tombstone_ratio > compaction_threshold * 0.75:  # warn at 75% of threshold
            issues.append(
                f"FAISS tombstone ratio {tombstone_ratio:.1%} approaching "
                f"compaction threshold {compaction_threshold:.0%}"
            )

        if last_run:
            if last_run.get("status") == "failed":
                issues.append(
                    f"Last consolidation failed: {last_run.get('error_message', 'unknown')}"
                )
            completed_at = last_run.get("completed_at") or last_run.get("started_at")
            if completed_at:
                from datetime import datetime, timezone
                try:
                    last_time = datetime.fromisoformat(completed_at)
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=timezone.utc)
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

    def _probe_backend(self) -> bool:
        """Quick check if embedding backend is reachable. Returns True/False."""
        from consolidation_memory.config import EMBEDDING_BACKEND, EMBEDDING_API_BASE

        if EMBEDDING_BACKEND == "fastembed":
            return True

        from urllib.request import urlopen, Request
        from urllib.error import URLError

        try:
            req = Request(
                f"{EMBEDDING_API_BASE}/models",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=3) as resp:
                resp.read()
            return True
        except (URLError, ConnectionError, TimeoutError, OSError):
            return False

    def _check_embedding_backend(self) -> None:
        """Verify the embedding backend is reachable."""
        from consolidation_memory.config import EMBEDDING_BACKEND, EMBEDDING_API_BASE, EMBEDDING_MODEL_NAME

        if EMBEDDING_BACKEND == "fastembed":
            logger.info("Embedding backend: fastembed (local, no server check needed)")
            return

        from urllib.request import urlopen, Request
        from urllib.error import URLError

        try:
            req = Request(
                f"{EMBEDDING_API_BASE}/models",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
            model_ids = [m.get("id", "") for m in body.get("data", [])]
            if EMBEDDING_MODEL_NAME not in model_ids:
                logger.warning(
                    "Embedding model '%s' not found. Loaded: %s",
                    EMBEDDING_MODEL_NAME, model_ids,
                )
            else:
                logger.info("Embedding backend health check passed (%s).", EMBEDDING_BACKEND)
        except (URLError, ConnectionError, TimeoutError) as e:
            logger.warning(
                "%s not reachable at %s: %s. Store/recall will fail until available.",
                EMBEDDING_BACKEND, EMBEDDING_API_BASE, e,
            )

    def _start_consolidation_thread(self) -> None:
        """Start the background consolidation daemon thread."""
        self._consolidation_stop.clear()
        self._consolidation_thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True,
            name="consolidation-bg",
        )
        self._consolidation_thread.start()

    def _consolidation_loop(self) -> None:
        """Background consolidation thread target."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        from consolidation_memory import config

        interval = config.CONSOLIDATION_INTERVAL_HOURS * 3600
        # Allow internal timeout + 60s buffer before we forcibly give up
        max_duration = config.CONSOLIDATION_MAX_DURATION + 60
        logger.info(
            "Background consolidation thread started (interval: %.1fh, timeout: %ds)",
            config.CONSOLIDATION_INTERVAL_HOURS, max_duration,
        )

        while not self._consolidation_stop.wait(timeout=interval):
            if self._consolidation_stop.is_set():
                break
            if not self._consolidation_lock.acquire(blocking=False):
                logger.info("Consolidation already running, skipping")
                continue
            try:
                from consolidation_memory.consolidation import run_consolidation
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(run_consolidation, vector_store=self._vector_store)
                    try:
                        result = future.result(timeout=max_duration)
                        logger.info(
                            "Background consolidation completed: %s",
                            result.get("status", result),
                        )
                    except FuturesTimeoutError:
                        logger.error(
                            "Background consolidation timed out after %ds; "
                            "releasing lock. The worker thread will be abandoned.",
                            max_duration,
                        )
            except Exception:
                logger.exception("Background consolidation failed")
            finally:
                self._consolidation_lock.release()

        logger.info("Background consolidation thread stopped.")
