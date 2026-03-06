"""Tests for anchor extraction and ingestion integration."""

from unittest.mock import patch

import pytest

from consolidation_memory.anchors import extract_anchors
from consolidation_memory.client import MemoryClient
from consolidation_memory.database import ensure_schema, get_connection
from tests.helpers import mock_encode as _mock_encode


class TestAnchorExtraction:
    def test_path_extraction_windows_and_posix(self):
        text = (
            r"Edit C:\Users\gore\repo\app\main.py and "
            r"./src/consolidation_memory/client.py then /opt/service/app/server.ts"
        )
        anchors = extract_anchors(text)
        paths = [a["anchor_value"] for a in anchors if a["anchor_type"] == "path"]
        assert paths == [
            r"C:\Users\gore\repo\app\main.py",
            "./src/consolidation_memory/client.py",
            "/opt/service/app/server.ts",
        ]

    def test_commit_hash_extraction(self):
        text = "Rollback to ABC1234 and 0123456789abcdef0123456789ABCDEF01234567"
        anchors = extract_anchors(text)
        commits = [a["anchor_value"] for a in anchors if a["anchor_type"] == "commit"]
        assert commits == [
            "abc1234",
            "0123456789abcdef0123456789abcdef01234567",
        ]

    def test_low_false_positive_sanity(self):
        text = "Release notes: score 1234567, ticket PROJ-123, no code paths here."
        assert extract_anchors(text) == []


@pytest.fixture()
def client():
    """Create a MemoryClient with mocked embedding backend."""
    ensure_schema()
    with (
        patch("consolidation_memory.backends.encode_documents", side_effect=_mock_encode),
        patch("consolidation_memory.backends.get_dimension", return_value=384),
    ):
        c = MemoryClient(auto_consolidate=False)
        yield c
        c.close()


class TestAnchorIngestionIntegration:
    def test_store_writes_anchors(self, client):
        result = client.store(
            "Run pytest for src/consolidation_memory/client.py at abc1234",
            content_type="solution",
        )
        assert result.status == "stored"
        assert result.id is not None

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT anchor_type, anchor_value FROM episode_anchors WHERE episode_id = ?",
                (result.id,),
            ).fetchall()
        anchors = {(row["anchor_type"], row["anchor_value"]) for row in rows}
        assert ("path", "src/consolidation_memory/client.py") in anchors
        assert ("commit", "abc1234") in anchors
        assert ("tool", "pytest") in anchors

    def test_store_batch_writes_anchors_for_accepted_items(self, client):
        existing = client.store("duplicate path src/app/main.py", content_type="fact")
        assert existing.status == "stored"

        result = client.store_batch(
            [
                {"content": "duplicate path src/app/main.py", "content_type": "fact"},
                {
                    "content": r"Use docker on C:\Users\gore\repo\api\server.py with commit def5678",
                    "content_type": "solution",
                },
                {"content": "plain text no anchors", "content_type": "exchange"},
            ],
        )

        assert result.status == "stored"
        duplicate_rows = [r for r in result.results if r.get("status") == "duplicate_detected"]
        stored_rows = [r for r in result.results if r.get("status") == "stored"]
        assert len(duplicate_rows) == 1
        assert len(stored_rows) == 2

        with get_connection() as conn:
            anchored_count = conn.execute(
                "SELECT COUNT(*) AS c FROM episode_anchors WHERE episode_id = ?",
                (stored_rows[0]["id"],),
            ).fetchone()["c"]
            plain_count = conn.execute(
                "SELECT COUNT(*) AS c FROM episode_anchors WHERE episode_id = ?",
                (stored_rows[1]["id"],),
            ).fetchone()["c"]

        assert anchored_count >= 2
        assert plain_count == 0
