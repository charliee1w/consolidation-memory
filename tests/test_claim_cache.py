"""Tests for claim embedding cache behavior."""

from unittest.mock import patch

from consolidation_memory import claim_cache
from tests.helpers import mock_encode


class TestClaimCache:
    def test_empty_claims_returns_none(self, tmp_data_dir):
        claim_cache.invalidate()
        vecs = claim_cache.get_claim_vecs([], [])
        assert vecs is None

    def test_reuses_embeddings_for_identical_snapshot(self, tmp_data_dir):
        claim_cache.invalidate()
        claims = [{
            "id": "claim-1",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "canonical_text": "python runtime is 3.12",
            "payload": {"subject": "python", "info": "3.12"},
        }]
        payloads, texts = claim_cache.build_claim_texts(claims)
        assert payloads[0]["subject"] == "python"

        with patch("consolidation_memory.backends.encode_documents", side_effect=mock_encode) as mock_embed:
            vecs1 = claim_cache.get_claim_vecs(claims, texts)
            vecs2 = claim_cache.get_claim_vecs(claims, texts)
        assert vecs1 is not None
        assert vecs2 is not None
        assert mock_embed.call_count == 1

    def test_snapshot_change_reembeds(self, tmp_data_dir):
        claim_cache.invalidate()
        claims = [{
            "id": "claim-1",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "canonical_text": "python runtime is 3.12",
            "payload": {"subject": "python", "info": "3.12"},
        }]
        _, texts = claim_cache.build_claim_texts(claims)

        with patch("consolidation_memory.backends.encode_documents", side_effect=mock_encode) as mock_embed:
            claim_cache.get_claim_vecs(claims, texts)
            claims_changed = [dict(claims[0])]
            claims_changed[0]["updated_at"] = "2026-01-02T00:00:00+00:00"
            _, texts_changed = claim_cache.build_claim_texts(claims_changed)
            claim_cache.get_claim_vecs(claims_changed, texts_changed)
        assert mock_embed.call_count == 2
