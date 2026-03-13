"""Canonical service layer for trust-preserving query semantics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from os import PathLike
from typing import Mapping

from consolidation_memory.query_semantics import (
    coerce_numeric_float,
    filter_claims_for_scope,
    parse_claim_payload,
    strategy_reuse_profile,
)
from consolidation_memory.types import (
    ClaimBrowseResult,
    ClaimSearchResult,
    DriftOutput,
    OutcomeBrowseResult,
    OutcomeType,
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
class OutcomeBrowseQuery:
    """Canonical outcome-browse envelope."""

    outcome_type: OutcomeType | None = None
    action_key: str | None = None
    source_claim_id: str | None = None
    source_record_id: str | None = None
    source_episode_id: str | None = None
    as_of: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class DriftQuery:
    """Canonical drift-detection envelope."""

    base_ref: str | None = None
    repo_path: str | PathLike[str] | None = None
    scope: Mapping[str, str | None] | None = None


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
        stats = get_stats(scope=scope_filter)

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

    def browse_claims(
        self,
        query: ClaimBrowseQuery,
        *,
        scope_filter: Mapping[str, str | None] | None = None,
    ) -> ClaimBrowseResult:
        """Execute canonical claim browse semantics with temporal support."""
        from consolidation_memory.database import get_active_claims, get_claims_as_of

        bounded_limit = max(1, min(query.limit, 200))
        page_size = bounded_limit
        if scope_filter:
            page_size = min(max(bounded_limit * 5, 50), 1000)

        def _fetch_rows(*, offset: int, limit: int) -> list[dict[str, object]]:
            if query.as_of:
                return get_claims_as_of(
                    as_of=query.as_of,
                    claim_type=query.claim_type,
                    limit=limit,
                    offset=offset,
                )
            return get_active_claims(
                claim_type=query.claim_type,
                limit=limit,
                offset=offset,
            )

        def _rows_to_claims(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
            return claims

        scoped_claims: list[dict[str, object]]
        if not scope_filter:
            rows = _fetch_rows(offset=0, limit=page_size)
            scoped_claims = _rows_to_claims(rows)
        else:
            scoped_claims = []
            offset = 0
            while len(scoped_claims) < bounded_limit:
                rows = _fetch_rows(offset=offset, limit=page_size)
                if not rows:
                    break
                remaining = bounded_limit - len(scoped_claims)
                scoped_claims.extend(
                    filter_claims_for_scope(_rows_to_claims(rows), scope_filter)[:remaining]
                )
                offset += len(rows)
                if len(rows) < page_size:
                    break

        strategy_ids = [
            str(claim["id"])
            for claim in scoped_claims
            if claim.get("id") and claim.get("claim_type") == "strategy"
        ]
        strategy_evidence = self._strategy_evidence_by_claim(
            strategy_ids,
            as_of=query.as_of,
            scope_filter=scope_filter,
        )
        if strategy_evidence:
            for claim in scoped_claims:
                if claim.get("claim_type") != "strategy":
                    continue
                claim_id = str(claim.get("id", ""))
                claim["strategy_evidence"] = strategy_reuse_profile(
                    strategy_evidence.get(claim_id)
                )

        return ClaimBrowseResult(
            claims=scoped_claims[:bounded_limit],
            total=min(len(scoped_claims), bounded_limit),
            claim_type=query.claim_type,
            as_of=query.as_of,
        )

    def search_claims(
        self,
        query: ClaimSearchQuery,
        *,
        scope_filter: Mapping[str, str | None] | None = None,
    ) -> ClaimSearchResult:
        """Execute canonical claim search semantics using browse snapshot + ranking."""
        from consolidation_memory.database import get_active_claims, get_claims_as_of

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

        page_size = min(max(bounded_limit * 5, 50), 250)

        def _fetch_rows(*, offset: int, limit: int) -> list[dict[str, object]]:
            if query.as_of:
                return get_claims_as_of(
                    as_of=query.as_of,
                    claim_type=query.claim_type,
                    limit=limit,
                    offset=offset,
                )
            return get_active_claims(
                claim_type=query.claim_type,
                limit=limit,
                offset=offset,
            )

        def _rows_to_claims(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
            return claims

        all_claims: list[dict[str, object]] = []
        offset = 0
        while True:
            rows = _fetch_rows(offset=offset, limit=page_size)
            if not rows:
                break
            page_claims = _rows_to_claims(rows)
            if scope_filter:
                page_claims = filter_claims_for_scope(page_claims, scope_filter)
            all_claims.extend(page_claims)
            offset += len(rows)
            if len(rows) < page_size:
                break

        if not all_claims:
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
        strategy_ids = [
            str(claim["id"])
            for claim in all_claims
            if claim.get("id") and claim.get("claim_type") == "strategy"
        ]
        strategy_evidence = self._strategy_evidence_by_claim(
            strategy_ids,
            as_of=query.as_of,
            scope_filter=scope_filter,
        )
        for claim in all_claims:
            payload_text = json.dumps(claim.get("payload", {}), sort_keys=True, default=str)
            haystack = f"{claim.get('canonical_text', '')} {payload_text}".lower()
            if not haystack.strip():
                continue

            phrase_hit = 1.0 if query_lower in haystack else 0.0
            term_hits = sum(1 for term in query_terms if term in haystack)
            if phrase_hit == 0.0 and term_hits == 0:
                continue

            term_score = (term_hits / len(query_terms)) if query_terms else 0.0
            raw_confidence = claim.get("confidence", 0.8)
            if isinstance(raw_confidence, bool):
                confidence = 0.8
            elif isinstance(raw_confidence, (int, float)):
                confidence = float(raw_confidence)
            elif isinstance(raw_confidence, str):
                try:
                    confidence = float(raw_confidence)
                except ValueError:
                    confidence = 0.8
            else:
                confidence = 0.8
            relevance = (phrase_hit + term_score) * (0.5 + 0.5 * confidence)

            ranked = dict(claim)
            if ranked.get("claim_type") == "strategy":
                claim_id = str(ranked.get("id", ""))
                profile = strategy_reuse_profile(strategy_evidence.get(claim_id))
                ranked["strategy_evidence"] = profile
                relevance *= coerce_numeric_float(
                    profile.get("reuse_multiplier"),
                    default=1.0,
                )
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

    def _strategy_evidence_by_claim(
        self,
        claim_ids: list[str],
        *,
        as_of: str | None,
        scope_filter: Mapping[str, str | None] | None,
    ) -> dict[str, dict[str, object]]:
        if not claim_ids:
            return {}
        from consolidation_memory.database import get_claim_outcome_evidence

        return get_claim_outcome_evidence(
            claim_ids,
            as_of=as_of,
            scope=scope_filter,
        )

    def browse_outcomes(
        self,
        query: OutcomeBrowseQuery,
        *,
        scope_filter: Mapping[str, str | None] | None = None,
    ) -> OutcomeBrowseResult:
        """Execute canonical outcome browse semantics with scope and temporal filters."""
        from consolidation_memory.database import (
            get_action_outcome_refs_by_outcome_ids,
            get_action_outcome_sources_by_outcome_ids,
            get_action_outcomes,
        )

        bounded_limit = max(1, min(query.limit, 200))
        rows = get_action_outcomes(
            outcome_type=query.outcome_type,
            action_key=query.action_key,
            source_claim_id=query.source_claim_id,
            source_record_id=query.source_record_id,
            source_episode_id=query.source_episode_id,
            as_of=query.as_of,
            limit=bounded_limit,
            offset=0,
            scope=scope_filter,
        )
        outcome_ids = [str(row.get("id")) for row in rows if row.get("id")]
        source_rows = get_action_outcome_sources_by_outcome_ids(outcome_ids)
        ref_rows = get_action_outcome_refs_by_outcome_ids(outcome_ids)

        sources_by_outcome: dict[str, dict[str, list[str]]] = {}
        for source_row in source_rows:
            outcome_id = str(source_row.get("outcome_id") or "")
            if not outcome_id:
                continue
            slot = sources_by_outcome.setdefault(
                outcome_id,
                {
                    "source_claim_ids": [],
                    "source_record_ids": [],
                    "source_episode_ids": [],
                },
            )
            claim_id = source_row.get("source_claim_id")
            record_id = source_row.get("source_record_id")
            episode_id = source_row.get("source_episode_id")
            if claim_id is not None:
                token = str(claim_id)
                if token not in slot["source_claim_ids"]:
                    slot["source_claim_ids"].append(token)
            if record_id is not None:
                token = str(record_id)
                if token not in slot["source_record_ids"]:
                    slot["source_record_ids"].append(token)
            if episode_id is not None:
                token = str(episode_id)
                if token not in slot["source_episode_ids"]:
                    slot["source_episode_ids"].append(token)

        anchors_by_outcome: dict[str, list[dict[str, str]]] = {}
        issues_by_outcome: dict[str, list[str]] = {}
        prs_by_outcome: dict[str, list[str]] = {}
        for ref_row in ref_rows:
            outcome_id = str(ref_row.get("outcome_id") or "")
            if not outcome_id:
                continue
            ref_type = str(ref_row.get("ref_type") or "").strip()
            ref_key = str(ref_row.get("ref_key") or "").strip()
            ref_value = str(ref_row.get("ref_value") or "").strip()
            if not ref_type or not ref_value:
                continue
            if ref_type == "code_anchor":
                anchors = anchors_by_outcome.setdefault(outcome_id, [])
                anchor = {"anchor_type": ref_key, "anchor_value": ref_value}
                if anchor not in anchors:
                    anchors.append(anchor)
            elif ref_type == "issue":
                issue_ids = issues_by_outcome.setdefault(outcome_id, [])
                if ref_value not in issue_ids:
                    issue_ids.append(ref_value)
            elif ref_type == "pr":
                pr_ids = prs_by_outcome.setdefault(outcome_id, [])
                if ref_value not in pr_ids:
                    pr_ids.append(ref_value)

        def _parse_json_field(value: object) -> object:
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return {}
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    return {"raw": text}
            if value is None:
                return {}
            return value

        outcomes: list[dict[str, object]] = []
        for row in rows:
            outcome_id = str(row.get("id") or "")
            linked_sources = sources_by_outcome.get(
                outcome_id,
                {
                    "source_claim_ids": [],
                    "source_record_ids": [],
                    "source_episode_ids": [],
                },
            )
            outcomes.append(
                {
                    "id": outcome_id,
                    "action_key": row.get("action_key"),
                    "action_summary": row.get("action_summary"),
                    "outcome_type": row.get("outcome_type"),
                    "summary": row.get("summary"),
                    "details": _parse_json_field(row.get("details")),
                    "confidence": row.get("confidence"),
                    "provenance": _parse_json_field(row.get("provenance")),
                    "observed_at": row.get("observed_at"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "source_claim_ids": list(linked_sources["source_claim_ids"]),
                    "source_record_ids": list(linked_sources["source_record_ids"]),
                    "source_episode_ids": list(linked_sources["source_episode_ids"]),
                    "code_anchors": list(anchors_by_outcome.get(outcome_id, [])),
                    "issue_ids": list(issues_by_outcome.get(outcome_id, [])),
                    "pr_ids": list(prs_by_outcome.get(outcome_id, [])),
                }
            )

        return OutcomeBrowseResult(
            outcomes=outcomes,
            total=len(outcomes),
            outcome_type=query.outcome_type,
            action_key=query.action_key,
            source_claim_id=query.source_claim_id,
            source_record_id=query.source_record_id,
            source_episode_id=query.source_episode_id,
            as_of=query.as_of,
        )

    def detect_drift(self, query: DriftQuery) -> DriftOutput:
        """Execute canonical drift detection + challenge semantics."""
        from consolidation_memory.drift import detect_code_drift

        return detect_code_drift(
            base_ref=query.base_ref,
            repo_path=query.repo_path,
            scope=query.scope,
        )


__all__ = [
    "CanonicalQueryService",
    "ClaimBrowseQuery",
    "ClaimSearchQuery",
    "DriftQuery",
    "EpisodeSearchQuery",
    "OutcomeBrowseQuery",
    "RecallQuery",
]
