"""Tests for hypothesis competition (competing claims with lowered precision)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from consolidation_memory.claim_graph import claim_from_record
from consolidation_memory.config import override_config
from consolidation_memory.consolidation.engine import _merge_into_existing
from consolidation_memory.context_assembler import _search_claims, recall
from consolidation_memory.database import (
    ensure_schema,
    get_connection,
    get_contradictions,
    insert_claim_edge,
    insert_claim_sources,
    insert_episode,
    insert_knowledge_records,
    upsert_claim,
    upsert_knowledge_topic,
)
from consolidation_memory.hypothesis_competition import competing_hypothesis_precision
from consolidation_memory.schemas import dispatch_tool_call
from consolidation_memory.types import RecallResult
from consolidation_memory.vector_store import VectorStore
from tests.helpers import mock_encode


class TestCompetingHypothesisPrecision:
    def test_lowers_precision_by_factor(self):
        assert competing_hypothesis_precision(1.0, 0.55) == pytest.approx(0.55)

    def test_clamps_factor_bounds(self):
        assert competing_hypothesis_precision(1.0, 1.5) == pytest.approx(1.0)
        assert competing_hypothesis_precision(1.0, -0.2) == pytest.approx(0.0)


class TestHypothesisCompetitionMerge:
    @staticmethod
    def _unit_vec(seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        vec = rng.randn(8).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def test_competition_mode_retains_record_and_lowers_precision(self, tmp_data_dir):
        ensure_schema()

        topic_id = upsert_knowledge_topic(
            filename="hypothesis_competition.md",
            title="Hypothesis Competition",
            summary="S",
            source_episodes=["ep-old"],
        )
        old_record = {"type": "fact", "subject": "Python", "info": "version 3.11"}
        old_record_ids = insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": old_record,
                    "embedding_text": "Python version 3.11",
                    "confidence": 0.8,
                },
            ],
        )
        assert len(old_record_ids) == 1
        old_record_id = old_record_ids[0]

        new_record = {"type": "fact", "subject": "Python", "info": "version 3.12"}
        extraction_data = {
            "title": "Hypothesis Competition",
            "summary": "S2",
            "tags": [],
            "records": [new_record],
        }
        merged_data = {
            "title": "Hypothesis Competition",
            "summary": "S2",
            "tags": [],
            "records": [new_record],
        }
        existing = {
            "id": topic_id,
            "filename": "hypothesis_competition.md",
            "title": "Hypothesis Competition",
            "summary": "S",
        }
        insert_episode(content="contradiction source", tags=[], episode_id="ep-compete-1")

        def mock_encode(texts):
            return np.array([self._unit_vec(42) for _ in texts], dtype=np.float32)

        with (
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                return_value=(merged_data, 1),
            ),
            patch("consolidation_memory.consolidation.engine.encode_documents", mock_encode),
            override_config(
                RENDER_MARKDOWN=False,
                MERGE_DROP_DETECTION_ENABLED=False,
                CONTRADICTION_LLM_ENABLED=False,
                CONTRADICTION_SIMILARITY_THRESHOLD=0.5,
                HYPOTHESIS_COMPETITION_ENABLED=True,
                HYPOTHESIS_COMPETITION_PRECISION_FACTOR=0.55,
            ),
        ):
            status, _ = _merge_into_existing(
                existing=existing,
                extraction_data=extraction_data,
                cluster_episodes=[{"id": "ep-compete-1", "content": "c", "tags": "[]"}],
                cluster_ep_ids=["ep-compete-1"],
                confidence=0.8,
            )

        assert status == "updated"

        with get_connection() as conn:
            row = conn.execute(
                "SELECT valid_until FROM knowledge_records WHERE id = ?",
                (old_record_id,),
            ).fetchone()
            assert row is not None
            assert row["valid_until"] is None

        rows = get_contradictions(topic_id=topic_id)
        assert rows
        assert rows[0]["resolution"] == "competing_hypotheses"

        old_claim_id = claim_from_record(old_record)["id"]
        new_claim_id = claim_from_record(new_record)["id"]
        with get_connection() as conn:
            old_precision = conn.execute(
                "SELECT precision FROM claims WHERE id = ?",
                (old_claim_id,),
            ).fetchone()["precision"]
            new_precision = conn.execute(
                "SELECT precision FROM claims WHERE id = ?",
                (new_claim_id,),
            ).fetchone()["precision"]
        # Contradiction events lower precision first; competition applies an extra factor.
        expected = competing_hypothesis_precision(0.85, 0.55)
        assert old_precision == pytest.approx(expected)
        assert new_precision == pytest.approx(expected)


class TestHypothesisCompetitionRecall:
    def test_includes_expired_contradicting_partner_when_enabled(self, tmp_data_dir):
        ensure_schema()
        vs = VectorStore()

        ep_id = insert_episode(
            content="python runtime version details",
            content_type="fact",
            tags=[],
            surprise_score=0.5,
        )
        vs.add(ep_id, mock_encode(["python runtime version"])[0])

        active_id = "claim-active-python"
        expired_id = "claim-expired-python"
        upsert_claim(
            claim_id=active_id,
            claim_type="fact",
            canonical_text="type=fact | subject=python | info=runtime version 3.12",
            payload={"type": "fact", "subject": "python", "info": "runtime version 3.12"},
            confidence=0.9,
        )
        upsert_claim(
            claim_id=expired_id,
            claim_type="fact",
            canonical_text="type=fact | subject=python | info=runtime version 3.11",
            payload={"type": "fact", "subject": "python", "info": "runtime version 3.11"},
            confidence=0.8,
            status="expired",
            valid_until=datetime.now(timezone.utc).isoformat(),
        )
        insert_claim_sources(active_id, [{"source_episode_id": ep_id}])
        insert_claim_sources(expired_id, [{"source_episode_id": ep_id}])
        insert_claim_edge(
            from_claim_id=active_id,
            to_claim_id=expired_id,
            edge_type="contradicts",
            details={"reason": "test"},
        )

        query_vec = mock_encode(["python runtime version"])[0]
        with (
            patch(
                "consolidation_memory.context_assembler.backends.encode_query",
                side_effect=lambda text: query_vec,
            ),
            patch(
                "consolidation_memory.context_assembler.backends.encode_documents",
                side_effect=lambda texts: np.stack([query_vec for _ in texts]),
            ),
        ):
            without = recall(
                "python runtime version",
                n_results=3,
                include_knowledge=True,
                vector_store=vs,
                hypothesis_competition=False,
            )
            with_mode = recall(
                "python runtime version",
                n_results=3,
                include_knowledge=True,
                vector_store=vs,
                hypothesis_competition=True,
            )

        without_ids = {claim["id"] for claim in without["claims"]}
        with_ids = {claim["id"] for claim in with_mode["claims"]}
        assert active_id in without_ids
        assert expired_id not in without_ids
        assert active_id in with_ids
        assert expired_id in with_ids

        competing = next(
            claim for claim in with_mode["claims"] if claim["id"] == expired_id
        )
        assert competing.get("competing_hypothesis") is True
        assert "Competing hypothesis" in competing.get("uncertainty", "")
        assert any("competing hypothes" in warning for warning in with_mode["warnings"])

    def test_search_claims_respects_hypothesis_competition_flag(self, tmp_data_dir):
        ensure_schema()

        active_id = "claim-flag-active"
        challenged_id = "claim-flag-challenged"
        upsert_claim(
            claim_id=active_id,
            claim_type="fact",
            canonical_text="python runtime current",
            payload={"subject": "python", "info": "current"},
            confidence=0.9,
        )
        upsert_claim(
            claim_id=challenged_id,
            claim_type="fact",
            canonical_text="python runtime challenged",
            payload={"subject": "python", "info": "challenged"},
            confidence=0.7,
            status="challenged",
        )
        insert_claim_edge(
            from_claim_id=active_id,
            to_claim_id=challenged_id,
            edge_type="contradicts",
            details={},
        )

        query_vec = mock_encode(["python runtime"])[0]
        with patch(
            "consolidation_memory.context_assembler.backends.encode_documents",
            side_effect=lambda texts: np.stack([query_vec for _ in texts]),
        ):
            off, _ = _search_claims("python runtime", query_vec, hypothesis_competition=False)
            on, warnings = _search_claims(
                "python runtime",
                query_vec,
                hypothesis_competition=True,
            )

        assert challenged_id not in {row["id"] for row in off}
        assert challenged_id in {row["id"] for row in on}
        assert any("competing hypothes" in warning for warning in warnings)


class TestHypothesisCompetitionDispatch:
    def test_dispatch_passes_hypothesis_competition_to_client(self):
        client = MagicMock()
        client.query_recall.return_value = RecallResult(episodes=[])

        dispatch_tool_call(
            client,
            "memory_recall",
            {"query": "python runtime", "hypothesis_competition": True},
        )

        client.query_recall.assert_called_once()
        assert client.query_recall.call_args.kwargs["hypothesis_competition"] is True