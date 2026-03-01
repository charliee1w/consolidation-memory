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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

from consolidation_memory import __version__
from consolidation_memory.types import (
    ContentType,
    ConsolidationReport,
    ContradictionResult,
    HealthStatus,
    StoreResult,
    BatchStoreResult,
    RecallResult,
    SearchResult,
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


class MemoryClient:
    """Persistent semantic memory client.

    Owns the vector store, database lifecycle, and optional background
    consolidation thread.  All public methods are synchronous and thread-safe.

    Args:
        auto_consolidate: Start background consolidation thread. Default True
            (respects ``CONSOLIDATION_AUTO_RUN`` config).
    """

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
        self._check_embedding_backend()

        # Consolidation threading
        self._consolidation_lock = threading.Lock()
        self._consolidation_stop = threading.Event()
        self._consolidation_thread: threading.Thread | None = None
        self._consolidation_pool: ThreadPoolExecutor | None = None

        # Cached backend probe result: (is_reachable, timestamp)
        self._probe_cache: tuple[bool, float] | None = None
        self._probe_cache_ttl = 30.0  # seconds

        if auto_consolidate and cfg.CONSOLIDATION_AUTO_RUN:
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
        if self._consolidation_pool is not None:
            self._consolidation_pool.shutdown(wait=False)
            self._consolidation_pool = None

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
        from consolidation_memory.database import insert_episode, get_episode, hard_delete_episode
        from consolidation_memory.backends import encode_documents
        from consolidation_memory.config import get_config

        cfg = get_config()
        self._vector_store.reload_if_stale()

        try:
            embedding = encode_documents([content])
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during store: %s", e)
            return StoreResult(
                status="backend_unavailable",
                message=f"Embedding backend unreachable: {e}",
            )

        # Dedup check: search top-3 so tombstone-filtered results don't mask real duplicates
        if cfg.DEDUP_ENABLED and self._vector_store.size > 0:
            matches = self._vector_store.search(embedding[0], k=3)
            for match_id, match_sim in matches:
                if match_sim >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                    existing = get_episode(match_id)
                    if existing is not None:
                        logger.info(
                            "Duplicate detected (sim=%.4f >= %.2f): existing=%s",
                            match_sim, cfg.DEDUP_SIMILARITY_THRESHOLD, match_id,
                        )
                        return StoreResult(
                            status="duplicate_detected",
                            existing_id=match_id,
                            similarity=round(match_sim, 4),
                            message="Content too similar to existing episode. Not stored.",
                        )
                else:
                    break  # Results are sorted by similarity; no point checking lower ones

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
            # Hard-delete (not soft-delete) so dedup checks don't still find this orphan
            hard_delete_episode(episode_id)
            logger.error("FAISS add failed for %s, rolled back DB insert: %s", episode_id, e)
            raise

        # Update tag co-occurrence graph
        if tags and len(tags) >= 2:
            from consolidation_memory.database import update_tag_cooccurrence
            try:
                update_tag_cooccurrence(tags)
            except Exception as e:
                logger.warning("Failed to update tag co-occurrence: %s", e)

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
        from consolidation_memory.database import insert_episode, get_episode, hard_delete_episode
        from consolidation_memory.backends import encode_documents
        from consolidation_memory.config import get_config
        import numpy as np

        cfg = get_config()

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
        # Collect non-duplicate items for a single FAISS batch add
        pending_ids: list[str] = []
        pending_embs: list[np.ndarray] = []
        # Track accepted embeddings within this batch for intra-batch dedup
        accepted_embs: list[np.ndarray] = []

        for i, item in enumerate(items):
            emb = embeddings[i]

            # Dedup check: search top-3 so tombstone-filtered results don't mask real duplicates
            if cfg.DEDUP_ENABLED and self._vector_store.size > 0:
                matches = self._vector_store.search(emb, k=3)
                is_dup = False
                for match_id, match_sim in matches:
                    if match_sim >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                        existing = get_episode(match_id)
                        if existing is not None:
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
            )

            pending_ids.append(episode_id)
            pending_embs.append(emb)
            accepted_embs.append(emb.reshape(-1).astype(np.float32))
            stored += 1
            results.append({
                "status": "stored",
                "id": episode_id,
                "content_type": item["content_type"],
            })

        # Single FAISS batch add instead of per-item add()
        if pending_ids:
            try:
                emb_matrix = np.stack(pending_embs)
                self._vector_store.add_batch(pending_ids, emb_matrix)
            except Exception as e:
                # Rollback all DB inserts from this batch
                for eid in pending_ids:
                    hard_delete_episode(eid)
                logger.error("FAISS batch add failed, rolled back %d DB inserts: %s",
                             len(pending_ids), e)
                stored = 0
                results = [r for r in results if r.get("status") != "stored"]
                results.append({"status": "error", "message": str(e)})

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
        include_expired: bool = False,
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
            include_expired: Include temporally expired knowledge records.

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
                include_expired=include_expired,
            )
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during recall: %s", e)
            return RecallResult(
                message=f"Embedding backend unreachable: {e}",
            )

        stats = get_stats()

        records = result.get("records", [])
        logger.info(
            "Recall query='%s' returned %d episodes, %d knowledge entries, %d records",
            query[:80], len(result["episodes"]), len(result["knowledge"]), len(records),
        )

        return RecallResult(
            episodes=result["episodes"],
            knowledge=result["knowledge"],
            records=records,
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
        from consolidation_memory.database import (
            get_stats, get_last_consolidation_run, get_recent_consolidation_runs,
        )
        from consolidation_memory.config import get_config

        cfg = get_config()
        stats = get_stats()
        last_run = get_last_consolidation_run()

        db_size_mb = 0.0
        if cfg.DB_PATH.exists():
            db_size_mb = round(cfg.DB_PATH.stat().st_size / (1024 * 1024), 2)

        health = self._compute_health(last_run, cfg.CONSOLIDATION_INTERVAL_HOURS, cfg.FAISS_COMPACTION_THRESHOLD)

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
        from consolidation_memory.config import get_config
        from consolidation_memory.database import (
            get_all_episodes, get_all_knowledge_topics, get_all_active_records,
        )

        cfg = get_config()
        cfg.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        episodes = get_all_episodes(include_deleted=False)

        topics = get_all_knowledge_topics()
        knowledge = []
        knowledge_resolved = cfg.KNOWLEDGE_DIR.resolve()
        for topic in topics:
            filepath = (cfg.KNOWLEDGE_DIR / topic["filename"]).resolve()
            content = ""
            # Path traversal guard: skip files that resolve outside KNOWLEDGE_DIR
            if filepath.is_relative_to(knowledge_resolved) and filepath.exists():
                content = filepath.read_text(encoding="utf-8")
            knowledge.append({**topic, "file_content": content})

        records = get_all_active_records()

        snapshot = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.1",
            "episodes": episodes,
            "knowledge_topics": knowledge,
            "knowledge_records": records,
            "stats": {
                "episode_count": len(episodes),
                "knowledge_count": len(knowledge),
                "record_count": len(records),
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
        from consolidation_memory.config import get_config
        from consolidation_memory.database import get_all_knowledge_topics, upsert_knowledge_topic
        from consolidation_memory.consolidation.engine import _version_knowledge_file
        from consolidation_memory.consolidation.prompting import (
            _parse_frontmatter, _count_facts, _normalize_output,
        )
        from consolidation_memory.backends import get_llm_backend

        cfg = get_config()
        # Validate filename doesn't escape KNOWLEDGE_DIR (path traversal)
        filepath = (cfg.KNOWLEDGE_DIR / topic_filename).resolve()
        if not filepath.is_relative_to(cfg.KNOWLEDGE_DIR.resolve()):
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
            f"CORRECTION:\n{correction}\n\n"
            f"EXISTING DOCUMENT:\n{existing_content}\n\n"
            f"Output the complete corrected document with updated frontmatter "
            f"(title, summary, tags, confidence).\n"
            f"Do NOT wrap in code fences. Output raw markdown starting with --- frontmatter."
        )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(llm.generate, system_prompt, user_prompt)
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

        from consolidation_memory import topic_cache as _tc, record_cache as _rc
        _tc.invalidate()
        _rc.invalidate()

        logger.info("Corrected knowledge topic: %s", topic_filename)

        return CorrectResult(
            status="corrected",
            filename=topic_filename,
            title=meta.get("title", ""),
        )

    def browse(self) -> BrowseResult:
        """List all knowledge topics with summaries and metadata.

        Returns:
            BrowseResult with topic list and count.
        """
        from consolidation_memory.database import get_all_knowledge_topics, get_all_active_records
        from consolidation_memory.config import get_config

        cfg = get_config()
        topics = get_all_knowledge_topics()
        records = get_all_active_records(include_expired=False)

        # Group record counts by topic_id
        records_by_topic: dict[str, dict[str, int]] = {}
        for rec in records:
            tid = rec["topic_id"]
            if tid not in records_by_topic:
                records_by_topic[tid] = {"facts": 0, "solutions": 0, "preferences": 0, "procedures": 0}
            rt = rec.get("record_type", "fact")
            if rt in records_by_topic[tid]:
                records_by_topic[tid][rt] += 1

        result_topics = []
        for topic in topics:
            filepath = cfg.KNOWLEDGE_DIR / topic["filename"]
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
                "file_exists": filepath.exists(),
                "file_path": str(filepath),
            })

        return BrowseResult(topics=result_topics, total=len(result_topics))

    def read_topic(self, filename: str) -> TopicDetailResult:
        """Read the full rendered markdown content of a knowledge file.

        Args:
            filename: The filename of the knowledge topic (e.g. 'python_setup.md').

        Returns:
            TopicDetailResult with the markdown content.
        """
        from consolidation_memory.config import get_config

        cfg = get_config()
        filepath = (cfg.KNOWLEDGE_DIR / filename).resolve()

        # Path traversal guard
        if not filepath.is_relative_to(cfg.KNOWLEDGE_DIR.resolve()):
            return TopicDetailResult(
                status="error",
                filename=filename,
                message="Invalid filename: path traversal detected.",
            )
        if not filepath.exists():
            return TopicDetailResult(status="not_found", filename=filename)

        content = filepath.read_text(encoding="utf-8")
        return TopicDetailResult(status="ok", filename=filename, content=content)

    def timeline(self, topic: str) -> TimelineResult:
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

        self._vector_store.reload_if_stale()

        # Get all records including expired
        all_records = get_all_active_records(include_expired=True)
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
                    expired_vec = record_vecs[valid_indices.index(rec_idx)] if rec_idx in valid_indices else None
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
                                # Check embedding similarity
                                other_vec_idx = valid_indices.index(other_idx) if other_idx in valid_indices else None
                                if other_vec_idx is not None:
                                    pair_sim = float(
                                        expired_vec.reshape(1, -1) @ record_vecs[other_vec_idx].reshape(-1, 1)
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
        )

        topic_id = None
        if topic:
            topics = get_all_knowledge_topics()
            for t in topics:
                if topic in (t["filename"], t["title"]):
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
        self, episode_id: str | None = None, tag: str | None = None,
    ) -> ProtectResult:
        """Mark episodes as immune to pruning.

        Args:
            episode_id: Protect a specific episode by UUID.
            tag: Protect all episodes with this tag.

        Returns:
            ProtectResult with status and count.
        """
        from consolidation_memory.database import protect_episode, protect_by_tag

        if not episode_id and not tag:
            return ProtectResult(
                status="error", message="Provide either episode_id or tag."
            )

        total = 0
        if episode_id:
            found = protect_episode(episode_id)
            if not found:
                return ProtectResult(status="not_found", message=f"Episode {episode_id} not found.")
            total += 1

        if tag:
            count = protect_by_tag(tag)
            total += count

        return ProtectResult(
            status="protected",
            protected_count=total,
            message=f"Protected {total} episode(s).",
        )

    def consolidate(self) -> ConsolidationReport:
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
        last_run: dict[str, object] | None,
        interval_hours: float,
        compaction_threshold: float,
    ) -> HealthStatus:
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
            if completed_at and isinstance(completed_at, str):
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
        """Quick check if embedding backend is reachable. Cached for 30s."""
        import time
        from consolidation_memory.config import get_config

        cfg = get_config()
        if cfg.EMBEDDING_BACKEND == "fastembed":
            return True

        # Return cached result if fresh
        if self._probe_cache is not None:
            cached_result, cached_at = self._probe_cache
            if time.monotonic() - cached_at < self._probe_cache_ttl:
                return cached_result

        from urllib.request import urlopen, Request
        from urllib.error import URLError

        try:
            req = Request(
                f"{cfg.EMBEDDING_API_BASE}/models",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=3) as resp:
                resp.read()
            self._probe_cache = (True, time.monotonic())
            return True
        except (URLError, ConnectionError, TimeoutError, OSError):
            self._probe_cache = (False, time.monotonic())
            return False

    def _check_embedding_backend(self) -> None:
        """Verify the embedding backend is reachable."""
        from consolidation_memory.config import get_config

        cfg = get_config()
        if cfg.EMBEDDING_BACKEND == "fastembed":
            logger.info("Embedding backend: fastembed (local, no server check needed)")
            return

        from urllib.request import urlopen, Request
        from urllib.error import URLError

        try:
            req = Request(
                f"{cfg.EMBEDDING_API_BASE}/models",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
            model_ids = [m.get("id", "") for m in body.get("data", [])]
            if cfg.EMBEDDING_MODEL_NAME not in model_ids:
                logger.warning(
                    "Embedding model '%s' not found. Loaded: %s",
                    cfg.EMBEDDING_MODEL_NAME, model_ids,
                )
            else:
                logger.info("Embedding backend health check passed (%s).", cfg.EMBEDDING_BACKEND)
        except (URLError, ConnectionError, TimeoutError) as e:
            logger.warning(
                "%s not reachable at %s: %s. Store/recall will fail until available.",
                cfg.EMBEDDING_BACKEND, cfg.EMBEDDING_API_BASE, e,
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
        from consolidation_memory.config import get_config

        cfg = get_config()
        interval = cfg.CONSOLIDATION_INTERVAL_HOURS * 3600
        # Allow internal timeout + 60s buffer before we forcibly give up
        max_duration = cfg.CONSOLIDATION_MAX_DURATION + 60
        logger.info(
            "Background consolidation thread started (interval: %.1fh, timeout: %ds)",
            cfg.CONSOLIDATION_INTERVAL_HOURS, max_duration,
        )

        # Use a single shared pool for consolidation runs instead of creating
        # a new ThreadPoolExecutor on every cycle (prevents zombie threads on timeout).
        if self._consolidation_pool is None:
            self._consolidation_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="consolidation"
            )

        while not self._consolidation_stop.wait(timeout=interval):
            if self._consolidation_stop.is_set():
                break
            if not self._consolidation_lock.acquire(blocking=False):
                logger.info("Consolidation already running, skipping")
                continue
            try:
                from consolidation_memory.consolidation import run_consolidation
                future = self._consolidation_pool.submit(
                    run_consolidation, vector_store=self._vector_store
                )
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
