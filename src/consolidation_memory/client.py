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
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

from consolidation_memory import __version__
from consolidation_memory.utils import parse_datetime, parse_json_list
from consolidation_memory.types import (
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
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


_MD_KV_BULLET_RE = re.compile(r"^[-*]\s+(?:\*\*(.+?)\*\*|([^:]+)):\s*(.+)$")
_MD_CONTEXT_RE = re.compile(r"^\*Context:\s*(.+?)\*\s*$")
_MD_VALUE_CONTEXT_RE = re.compile(r"^(.*?)\s+\(([^()]+)\)\s*$")


def _parse_markdown_kv_bullet(line: str) -> tuple[str, str] | None:
    """Parse markdown bullet lines like '- **Key**: Value'."""
    match = _MD_KV_BULLET_RE.match(line)
    if not match:
        return None
    key = (match.group(1) or match.group(2) or "").strip()
    value = match.group(3).strip()
    if not key or not value:
        return None
    return key, value


def _extract_records_from_markdown(body: str) -> list[dict[str, str]]:
    """Extract structured records from rendered markdown sections."""
    records: list[dict[str, str]] = []
    lines = body.splitlines()
    section: str | None = None
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        lowered = stripped.lower()

        if lowered == "## facts":
            section = "facts"
            i += 1
            continue
        if lowered == "## solutions":
            section = "solutions"
            i += 1
            continue
        if lowered == "## preferences":
            section = "preferences"
            i += 1
            continue
        if lowered == "## procedures":
            section = "procedures"
            i += 1
            continue

        if section == "facts":
            parsed = _parse_markdown_kv_bullet(stripped)
            if parsed:
                subject, info = parsed
                records.append({"type": "fact", "subject": subject, "info": info})
        elif section == "preferences":
            parsed = _parse_markdown_kv_bullet(stripped)
            if parsed:
                key, value_text = parsed
                rec: dict[str, str] = {"type": "preference", "key": key, "value": value_text}
                ctx_match = _MD_VALUE_CONTEXT_RE.match(value_text)
                if ctx_match:
                    rec["value"] = ctx_match.group(1).strip()
                    rec["context"] = ctx_match.group(2).strip()
                records.append(rec)
        elif section in {"solutions", "procedures"} and stripped.startswith("### "):
            header = stripped[4:].strip()
            j = i + 1
            body_lines: list[str] = []
            context = ""
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt.startswith("## ") or nxt.startswith("### "):
                    break
                context_match = _MD_CONTEXT_RE.match(nxt)
                if context_match:
                    context = context_match.group(1).strip()
                elif nxt:
                    body_lines.append(nxt)
                j += 1

            text_body = "\n".join(body_lines).strip()
            if header and text_body:
                if section == "solutions":
                    rec = {"type": "solution", "problem": header, "fix": text_body}
                else:
                    rec = {"type": "procedure", "trigger": header, "steps": text_body}
                if context:
                    rec["context"] = context
                records.append(rec)
            i = j
            continue

        i += 1

    return records


def _parse_claim_payload(payload_raw: object) -> dict[str, object]:
    """Parse claim payload column into a normalized dict."""
    if isinstance(payload_raw, dict):
        return dict(payload_raw)
    if isinstance(payload_raw, str):
        try:
            parsed = json.loads(payload_raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


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

        if auto_consolidate and cfg.CONSOLIDATION_AUTO_RUN:
            self._start_consolidation_thread()

        logger.info(
            "MemoryClient initialized (vectors=%d, version=%s)",
            self._vector_store.size,
            __version__,
        )

        # Discover and load plugins, then fire startup hook
        from consolidation_memory.plugins import get_plugin_manager
        if cfg.PLUGINS_ENABLED:
            get_plugin_manager().load_plugins()
        get_plugin_manager().fire("on_startup", client=self)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop background threads. Call when done."""
        from consolidation_memory.plugins import get_plugin_manager
        get_plugin_manager().fire("on_shutdown")

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

        content_type = _normalize_content_type(content_type)

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

        self._persist_episode_anchors(episode_id=episode_id, content=content)

        # Tag co-occurrence is non-critical — log and continue on failure
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

        from consolidation_memory.plugins import get_plugin_manager
        get_plugin_manager().fire(
            "on_store",
            episode_id=episode_id,
            content=content,
            content_type=content_type,
            tags=tags or [],
            surprise=surprise,
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
        # Map episode_id -> item for plugin hooks
        stored_items_by_id: dict[str, dict] = {}

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
            try:
                emb_matrix = np.stack(pending_embs)
                self._vector_store.add_batch(pending_ids, emb_matrix)
                batch_add_succeeded = True
            except Exception as e:
                # Rollback all DB inserts from this batch
                for eid in pending_ids:
                    hard_delete_episode(eid)
                logger.error("FAISS batch add failed, rolled back %d DB inserts: %s",
                             len(pending_ids), e)
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
        as_of: str | None = None,
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
            as_of: ISO datetime for temporal belief queries. Returns knowledge
                state as it was at that point in time, including records that
                have since been superseded.

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
                as_of=as_of,
            )
        except ConnectionError as e:
            logger.error("Embedding backend unreachable during recall: %s", e)
            self._record_recall_signal(fallback=True)
            return RecallResult(
                message=f"Embedding backend unreachable: {e}",
            )

        stats = get_stats()

        records = result.get("records", [])
        claims = result.get("claims", [])
        logger.info(
            "Recall query='%s' returned %d episodes, %d knowledge entries, %d records, %d claims",
            query[:80], len(result["episodes"]), len(result["knowledge"]), len(records), len(claims),
        )

        recall_result = RecallResult(
            episodes=result["episodes"],
            knowledge=result["knowledge"],
            records=records,
            claims=claims,
            total_episodes=stats["episodic_buffer"]["total"],
            total_knowledge_topics=stats["knowledge_base"]["total_topics"],
            warnings=result.get("warnings", []),
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
            episodes.append({
                "id": ep["id"],
                "content": ep["content"],
                "content_type": ep["content_type"],
                "tags": parse_json_list(ep["tags"]),
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

    def browse_claims(
        self,
        claim_type: str | None = None,
        as_of: str | None = None,
        limit: int = 50,
    ) -> ClaimBrowseResult:
        """List claims, optionally filtered by type and temporal snapshot."""
        from consolidation_memory.database import get_active_claims, get_claims_as_of

        bounded_limit = max(1, min(limit, 200))
        if as_of:
            rows = get_claims_as_of(as_of=as_of, claim_type=claim_type, limit=bounded_limit)
        else:
            rows = get_active_claims(claim_type=claim_type, limit=bounded_limit)

        claims: list[dict[str, object]] = []
        for row in rows:
            payload = _parse_claim_payload(row.get("payload"))
            claims.append({
                "id": row.get("id", ""),
                "claim_type": row.get("claim_type", ""),
                "canonical_text": row.get("canonical_text", ""),
                "payload": payload,
                "status": row.get("status", ""),
                "confidence": row.get("confidence", 0.0),
                "valid_from": row.get("valid_from"),
                "valid_until": row.get("valid_until"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            })

        logger.info(
            "Browse claims claim_type=%r as_of=%r returned %d results",
            claim_type,
            as_of,
            len(claims),
        )
        return ClaimBrowseResult(
            claims=claims,
            total=len(claims),
            claim_type=claim_type,
            as_of=as_of,
        )

    def search_claims(
        self,
        query: str,
        claim_type: str | None = None,
        as_of: str | None = None,
        limit: int = 50,
    ) -> ClaimSearchResult:
        """Search claims using deterministic phrase/keyword ranking."""
        bounded_limit = max(1, min(limit, 200))
        normalized_query = query.strip()
        if not normalized_query:
            return ClaimSearchResult(
                claims=[],
                total_matches=0,
                query=query,
                claim_type=claim_type,
                as_of=as_of,
                message="Query must not be empty.",
            )

        fetch_limit = min(max(bounded_limit * 5, bounded_limit), 1000)
        browse_result = self.browse_claims(
            claim_type=claim_type,
            as_of=as_of,
            limit=fetch_limit,
        )
        if not browse_result.claims:
            return ClaimSearchResult(
                claims=[],
                total_matches=0,
                query=normalized_query,
                claim_type=claim_type,
                as_of=as_of,
            )

        query_lower = normalized_query.lower()
        query_terms = [term for term in query_lower.split() if term]

        scored: list[dict[str, object]] = []
        for claim in browse_result.claims:
            payload_text = json.dumps(claim.get("payload", {}), sort_keys=True, default=str)
            haystack = f"{claim.get('canonical_text', '')} {payload_text}".lower()
            if not haystack.strip():
                continue

            phrase_hit = 1.0 if query_lower in haystack else 0.0
            term_hits = sum(1 for term in query_terms if term in haystack)
            if phrase_hit == 0.0 and term_hits == 0:
                continue

            term_score = (term_hits / len(query_terms)) if query_terms else 0.0
            confidence = float(claim.get("confidence", 0.8) or 0.8)
            relevance = (phrase_hit + term_score) * (0.5 + 0.5 * confidence)

            ranked = dict(claim)
            ranked["relevance"] = round(relevance, 3)
            scored.append(ranked)

        def _claim_sort_key(item: dict[str, object]) -> tuple[float, str]:
            raw_relevance = item.get("relevance", 0.0)
            relevance = float(raw_relevance) if isinstance(raw_relevance, (int, float)) else 0.0
            return relevance, str(item.get("updated_at", ""))

        scored.sort(key=_claim_sort_key, reverse=True)
        top_claims = scored[:bounded_limit]

        logger.info(
            "Search claims query=%r claim_type=%r as_of=%r returned %d matches",
            normalized_query,
            claim_type,
            as_of,
            len(top_claims),
        )
        return ClaimSearchResult(
            claims=top_claims,
            total_matches=len(top_claims),
            query=normalized_query,
            claim_type=claim_type,
            as_of=as_of,
        )

    def detect_drift(
        self,
        base_ref: str | None = None,
        repo_path: str | None = None,
    ) -> DriftOutput:
        """Detect code drift and challenge anchored claims."""
        from consolidation_memory.drift import detect_code_drift

        result = detect_code_drift(base_ref=base_ref, repo_path=repo_path)
        logger.info(
            "Detect drift base_ref=%r repo_path=%r impacted=%d challenged=%d",
            base_ref,
            repo_path,
            len(result["impacted_claim_ids"]),
            len(result["challenged_claim_ids"]),
        )
        return result

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

        utility_state = self._compute_consolidation_utility()
        utility_scheduler = {
            "enabled": bool(cfg.CONSOLIDATION_AUTO_RUN),
            "threshold": cfg.CONSOLIDATION_UTILITY_THRESHOLD,
            "weights": dict(cfg.CONSOLIDATION_UTILITY_WEIGHTS),
            "score": utility_state["score"],
            "normalized_signals": utility_state["normalized_signals"],
            "weighted_components": utility_state["weighted_components"],
            "raw_signals": utility_state["raw_signals"],
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

            from consolidation_memory.plugins import get_plugin_manager
            get_plugin_manager().fire("on_forget", episode_id=episode_id)

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
            get_all_active_records,
            get_all_claim_edges,
            get_all_claim_events,
            get_all_claim_sources,
            get_all_claims,
            get_all_episode_anchors,
            get_all_episodes,
            get_all_knowledge_topics,
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
        claims = get_all_claims()
        claim_edges = get_all_claim_edges()
        claim_sources = get_all_claim_sources()
        claim_events = get_all_claim_events()
        episode_anchors = get_all_episode_anchors()

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

    def correct(self, topic_filename: str, correction: str) -> CorrectResult:
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
            get_all_knowledge_topics,
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

        _version_knowledge_file(filepath)
        filepath.write_text(corrected, encoding="utf-8")

        parsed = _parse_frontmatter(corrected)
        meta = parsed["meta"]
        body = parsed.get("body", "")

        topics = get_all_knowledge_topics()
        existing_topic = next((t for t in topics if t["filename"] == topic_filename), None)
        source_eps = parse_json_list(existing_topic.get("source_episodes") if existing_topic else None)
        existing_confidence = float(existing_topic["confidence"]) if existing_topic else 0.8

        try:
            confidence = float(meta.get("confidence", existing_confidence))
        except (TypeError, ValueError):
            confidence = existing_confidence
        confidence = max(0.0, min(1.0, confidence))

        parsed_records = _extract_records_from_markdown(body)
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

        topic_id = upsert_knowledge_topic(
            filename=topic_filename,
            title=str(meta.get("title", existing_topic["title"] if existing_topic else topic_filename)),
            summary=str(meta.get("summary", existing_topic["summary"] if existing_topic else "")),
            source_episodes=source_eps,
            fact_count=len(record_rows),
            confidence=confidence,
        )

        for old in get_records_by_topic(topic_id, include_expired=False):
            expire_record(old["id"], valid_until=now_ts)

        if record_rows:
            insert_knowledge_records(topic_id, record_rows, source_episodes=source_eps)

        from consolidation_memory import topic_cache as _tc, record_cache as _rc
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
                records_by_topic[tid] = {"fact": 0, "solution": 0, "preference": 0, "procedure": 0}
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
            if last_run.get("status") == RUN_STATUS_FAILED:
                issues.append(
                    f"Last consolidation failed: {last_run.get('error_message', 'unknown')}"
                )
            completed_at = last_run.get("completed_at") or last_run.get("started_at")
            if completed_at and isinstance(completed_at, str):
                from datetime import datetime, timezone
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

    def _record_recall_signal(
        self,
        *,
        miss: bool = False,
        fallback: bool = False,
        timestamp_monotonic: float | None = None,
    ) -> None:
        """Record recall miss/fallback events for utility scheduling."""
        if not miss and not fallback:
            return
        ts = timestamp_monotonic if timestamp_monotonic is not None else time.monotonic()
        with self._scheduler_signal_lock:
            if miss:
                self._recall_miss_events.append(ts)
            if fallback:
                self._recall_fallback_events.append(ts)

    def _recent_recall_signal_counts(
        self,
        lookback_seconds: float,
        now_monotonic: float | None = None,
    ) -> tuple[int, int]:
        """Return miss/fallback counts within lookback window."""
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        cutoff = now - max(1.0, lookback_seconds)
        with self._scheduler_signal_lock:
            while self._recall_miss_events and self._recall_miss_events[0] < cutoff:
                self._recall_miss_events.popleft()
            while self._recall_fallback_events and self._recall_fallback_events[0] < cutoff:
                self._recall_fallback_events.popleft()
            return len(self._recall_miss_events), len(self._recall_fallback_events)

    def _compute_consolidation_utility(
        self,
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, object]:
        """Compute current utility score and signal breakdown."""
        from datetime import timedelta
        from consolidation_memory.config import get_config
        from consolidation_memory.consolidation.utility_scheduler import compute_utility_score
        from consolidation_memory.database import (
            count_active_challenged_claims,
            count_contradictions_since,
            get_stats,
        )

        cfg = get_config()
        lookback_seconds = max(300.0, cfg.CONSOLIDATION_INTERVAL_HOURS * 3600.0)
        miss_count, fallback_count = self._recent_recall_signal_counts(
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

    def _should_trigger_consolidation(
        self,
        *,
        now_monotonic: float,
        last_run_monotonic: float,
        interval_seconds: float,
        utility_score: float,
    ) -> tuple[bool, str]:
        """Decide whether to trigger consolidation this cycle."""
        if now_monotonic - last_run_monotonic >= interval_seconds:
            return True, "interval"
        from consolidation_memory.config import get_config

        cfg = get_config()
        if utility_score >= cfg.CONSOLIDATION_UTILITY_THRESHOLD:
            return True, "utility"
        return False, "none"

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
        poll_interval = min(interval, 60.0)
        # Allow internal timeout + 60s buffer before we forcibly give up
        max_duration = cfg.CONSOLIDATION_MAX_DURATION + 60
        logger.info(
            "Background consolidation thread started "
            "(interval: %.1fh, poll: %.0fs, utility_threshold: %.2f, timeout: %ds)",
            cfg.CONSOLIDATION_INTERVAL_HOURS,
            poll_interval,
            cfg.CONSOLIDATION_UTILITY_THRESHOLD,
            max_duration,
        )

        # Use a single shared pool for consolidation runs instead of creating
        # a new ThreadPoolExecutor on every cycle (prevents zombie threads on timeout).
        if self._consolidation_pool is None:
            self._consolidation_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="consolidation"
            )

        last_run_monotonic = time.monotonic()

        while not self._consolidation_stop.wait(timeout=poll_interval):
            if self._consolidation_stop.is_set():
                break
            if self._consolidation_pool is None:
                break
            now_monotonic = time.monotonic()
            utility_state = self._compute_consolidation_utility(now_monotonic=now_monotonic)
            score_value = utility_state.get("score")
            utility_score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
            should_run, trigger_reason = self._should_trigger_consolidation(
                now_monotonic=now_monotonic,
                last_run_monotonic=last_run_monotonic,
                interval_seconds=interval,
                utility_score=utility_score,
            )
            if not should_run:
                continue
            if not self._consolidation_lock.acquire(blocking=False):
                logger.info("Consolidation already running, skipping")
                continue
            try:
                from consolidation_memory.consolidation import run_consolidation
                logger.info(
                    "Consolidation trigger=%s utility_score=%.3f components=%s raw=%s",
                    trigger_reason,
                    utility_score,
                    utility_state["weighted_components"],
                    utility_state["raw_signals"],
                )
                future = self._consolidation_pool.submit(
                    run_consolidation, vector_store=self._vector_store
                )
                try:
                    result = future.result(timeout=max_duration)
                    last_run_monotonic = time.monotonic()
                    logger.info(
                        "Background consolidation completed: %s",
                        result.get("status", result),
                    )
                except FuturesTimeoutError:
                    future.cancel()
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
