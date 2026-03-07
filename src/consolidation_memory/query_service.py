"""Canonical service layer for trust-preserving query semantics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from os import PathLike
from typing import Mapping

from consolidation_memory.query_semantics import parse_claim_payload
from consolidation_memory.types import (
    ClaimBrowseResult,
    ClaimSearchResult,
    DriftOutput,
    RecallResult,
    SearchResult,
)
from consolidation_memory.utils import parse_json_list

logger = logging.getLogger("consolidation_memory")


@dataclass(frozen=True)
class RecallQuery:
    """Canonical recall query envelope."""

    query: str
    n_results: int = 10
    include_knowledge: bool = True
    content_types: list[str] | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    include_expired: bool = False
    as_of: str | None = None


@dataclass(frozen=True)
class EpisodeSearchQuery:
    """Canonical episode keyword-search envelope."""

    query: str | None = None
    content_types: list[str] | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    limit: int = 20


@dataclass(frozen=True)
class ClaimBrowseQuery:
    """Canonical claim-browse envelope."""

    claim_type: str | None = None
    as_of: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class ClaimSearchQuery:
    """Canonical claim-search envelope."""

    query: str
    claim_type: str | None = None
    as_of: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class DriftQuery:
    """Canonical drift-detection envelope."""

    base_ref: str | None = None
    repo_path: str | PathLike[str] | None = None


class CanonicalQueryService:
    """First-class query service used by every external adapter surface."""

    def __init__(self, vector_store) -> None:
        self._vector_store = vector_store

    def recall(
        self,
        query: RecallQuery,
        *,
        scope_filter: Mapping[str, str | None] | None = None,
    ) -> RecallResult:
        """Execute canonical recall semantics (temporal + trust-aware)."""
        from consolidation_memory.context_assembler import recall as assemble_recall
        from consolidation_memory.database import get_stats

        payload = assemble_recall(
            query=query.query,
            n_results=query.n_results,
            include_knowledge=query.include_knowledge,
            vector_store=self._vector_store,
            content_types=query.content_types,
            tags=query.tags,
            after=query.after,
            before=query.before,
            include_expired=query.include_expired,
            as_of=query.as_of,
            scope=dict(scope_filter) if scope_filter is not None else None,
        )
        stats = get_stats()

        return RecallResult(
            episodes=list(payload.get("episodes", [])),
            knowledge=list(payload.get("knowledge", [])),
            records=list(payload.get("records", [])),
            claims=list(payload.get("claims", [])),
            total_episodes=stats["episodic_buffer"]["total"],
            total_knowledge_topics=stats["knowledge_base"]["total_topics"],
            warnings=list(payload.get("warnings", [])),
        )

    def search(
        self,
        query: EpisodeSearchQuery,
        *,
        scope_filter: Mapping[str, str | None] | None = None,
    ) -> SearchResult:
        """Execute canonical keyword/metadata episode search semantics."""
        from consolidation_memory.database import search_episodes

        rows = search_episodes(
            query=query.query,
            content_types=query.content_types,
            tags=query.tags,
            after=query.after,
            before=query.before,
            scope=scope_filter,
            limit=query.limit,
        )

        episodes = [{
            "id": ep["id"],
            "content": ep["content"],
            "content_type": ep["content_type"],
            "tags": parse_json_list(ep["tags"]),
            "created_at": ep["created_at"],
            "surprise_score": ep["surprise_score"],
            "access_count": ep["access_count"],
        } for ep in rows]
        return SearchResult(
            episodes=episodes,
            total_matches=len(episodes),
            query=query.query,
        )

    def browse_claims(self, query: ClaimBrowseQuery) -> ClaimBrowseResult:
        """Execute canonical claim browse semantics with temporal support."""
        from consolidation_memory.database import get_active_claims, get_claims_as_of

        bounded_limit = max(1, min(query.limit, 200))
        if query.as_of:
            rows = get_claims_as_of(
                as_of=query.as_of,
                claim_type=query.claim_type,
                limit=bounded_limit,
            )
        else:
            rows = get_active_claims(
                claim_type=query.claim_type,
                limit=bounded_limit,
            )

        claims: list[dict[str, object]] = []
        for row in rows:
            claims.append({
                "id": row.get("id", ""),
                "claim_type": row.get("claim_type", ""),
                "canonical_text": row.get("canonical_text", ""),
                "payload": parse_claim_payload(row.get("payload")),
                "status": row.get("status", ""),
                "confidence": row.get("confidence", 0.0),
                "valid_from": row.get("valid_from"),
                "valid_until": row.get("valid_until"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            })

        return ClaimBrowseResult(
            claims=claims,
            total=len(claims),
            claim_type=query.claim_type,
            as_of=query.as_of,
        )

    def search_claims(self, query: ClaimSearchQuery) -> ClaimSearchResult:
        """Execute canonical claim search semantics using browse snapshot + ranking."""
        bounded_limit = max(1, min(query.limit, 200))
        normalized_query = query.query.strip()
        if not normalized_query:
            return ClaimSearchResult(
                claims=[],
                total_matches=0,
                query=query.query,
                claim_type=query.claim_type,
                as_of=query.as_of,
                message="Query must not be empty.",
            )

        fetch_limit = min(max(bounded_limit * 5, bounded_limit), 1000)
        browse_result = self.browse_claims(
            ClaimBrowseQuery(
                claim_type=query.claim_type,
                as_of=query.as_of,
                limit=fetch_limit,
            )
        )
        if not browse_result.claims:
            return ClaimSearchResult(
                claims=[],
                total_matches=0,
                query=normalized_query,
                claim_type=query.claim_type,
                as_of=query.as_of,
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

        return ClaimSearchResult(
            claims=top_claims,
            total_matches=len(top_claims),
            query=normalized_query,
            claim_type=query.claim_type,
            as_of=query.as_of,
        )

    def detect_drift(self, query: DriftQuery) -> DriftOutput:
        """Execute canonical drift detection + challenge semantics."""
        from consolidation_memory.drift import detect_code_drift

        return detect_code_drift(
            base_ref=query.base_ref,
            repo_path=query.repo_path,
        )


__all__ = [
    "CanonicalQueryService",
    "ClaimBrowseQuery",
    "ClaimSearchQuery",
    "DriftQuery",
    "EpisodeSearchQuery",
    "RecallQuery",
]

