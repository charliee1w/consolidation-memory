"""Tests for schema-guided knowledge extraction (v0.5.0 records)."""

from unittest.mock import patch

from consolidation_memory.database import (
    ensure_schema,
    get_all_active_records,
    get_record_count,
    get_records_by_topic,
    get_stats,
    increment_record_access,
    insert_knowledge_records,
    soft_delete_records_by_topic,
    upsert_knowledge_topic,
)
from consolidation_memory.types import RecordType
from tests.helpers import mock_encode


# ── Database CRUD ────────────────────────────────────────────────────────────

class TestRecordCRUD:
    def test_insert_and_retrieve(self, tmp_data_dir):
        ensure_schema()
        topic_id = upsert_knowledge_topic(
            filename="test.md", title="Test", summary="Test topic",
            source_episodes=["ep1"],
        )
        records = [
            {
                "record_type": "fact",
                "content": {"type": "fact", "subject": "Python", "info": "Version 3.12"},
                "embedding_text": "Python: Version 3.12",
            },
            {
                "record_type": "solution",
                "content": {"type": "solution", "problem": "ImportError", "fix": "pip install x"},
                "embedding_text": "Problem: ImportError. Fix: pip install x",
                "confidence": 0.9,
            },
        ]
        ids = insert_knowledge_records(topic_id, records, source_episodes=["ep1"])
        assert len(ids) == 2

        all_records = get_all_active_records()
        assert len(all_records) == 2
        assert all_records[0]["topic_title"] == "Test"

    def test_get_records_by_topic(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="t.md", title="T", summary="S", source_episodes=[],
        )
        insert_knowledge_records(tid, [{
            "record_type": "preference",
            "content": {"type": "preference", "key": "theme", "value": "dark"},
            "embedding_text": "Preference theme: dark",
        }])
        recs = get_records_by_topic(tid)
        assert len(recs) == 1
        assert recs[0]["record_type"] == "preference"

    def test_soft_delete_by_topic(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="del.md", title="Del", summary="S", source_episodes=[],
        )
        insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "x"},
            {"record_type": "fact", "content": {}, "embedding_text": "y"},
        ])
        assert get_record_count() == 2
        deleted = soft_delete_records_by_topic(tid)
        assert deleted == 2
        assert get_record_count() == 0

    def test_increment_access(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="acc.md", title="Acc", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "z"},
        ])
        increment_record_access(ids)
        increment_record_access(ids)
        recs = get_records_by_topic(tid)
        assert recs[0]["access_count"] == 2

    def test_empty_insert(self, tmp_data_dir):
        ensure_schema()
        ids = insert_knowledge_records("nonexistent", [])
        assert ids == []

    def test_insert_procedure_record(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="proc.md", title="Proc", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [{
            "record_type": "procedure",
            "content": {
                "type": "procedure",
                "trigger": "before committing code",
                "steps": "run tests, then lint, then commit",
            },
            "embedding_text": "Procedure: before committing code -> run tests, then lint, then commit",
        }])
        assert len(ids) == 1
        recs = get_records_by_topic(tid)
        assert len(recs) == 1
        assert recs[0]["record_type"] == "procedure"

    def test_stats_include_records(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="stats.md", title="Stats", summary="S", source_episodes=[],
        )
        insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "a"},
            {"record_type": "solution", "content": {}, "embedding_text": "b"},
            {"record_type": "preference", "content": {}, "embedding_text": "c"},
            {"record_type": "procedure", "content": {}, "embedding_text": "d"},
        ])
        stats = get_stats()
        kb = stats["knowledge_base"]
        assert kb["total_records"] == 4
        assert kb["records_by_type"]["facts"] == 1
        assert kb["records_by_type"]["solutions"] == 1
        assert kb["records_by_type"]["preferences"] == 1
        assert kb["records_by_type"]["procedures"] == 1


# ── Types ────────────────────────────────────────────────────────────────────

class TestRecordTypes:
    def test_record_type_enum(self):
        assert RecordType.FACT.value == "fact"
        assert RecordType.SOLUTION.value == "solution"
        assert RecordType.PREFERENCE.value == "preference"
        assert RecordType.PROCEDURE.value == "procedure"

    def test_recall_result_has_records(self):
        from consolidation_memory.types import RecallResult
        r = RecallResult()
        assert r.records == []
        assert r.claims == []


# ── Extraction validation ────────────────────────────────────────────────────

class TestExtractionValidation:
    def test_valid_output(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {
            "title": "Python Setup",
            "summary": "Python 3.12 installed with pip and venv on Ubuntu",
            "tags": ["python"],
            "records": [
                {"type": "fact", "subject": "Python", "info": "Version 3.12.1 installed"},
            ],
        }
        valid, failures = _validate_extraction_output(data, [])
        assert valid, failures

    def test_missing_title(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {"summary": "x", "records": [{"type": "fact", "subject": "a", "info": "b"}]}
        valid, failures = _validate_extraction_output(data, [])
        assert not valid
        assert any("title" in f.lower() for f in failures)

    def test_no_records(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {"title": "T", "summary": "S", "records": []}
        valid, failures = _validate_extraction_output(data, [])
        assert not valid
        assert any("No records" in f for f in failures)

    def test_invalid_record_type(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {
            "title": "T", "summary": "S",
            "records": [{"type": "invalid", "subject": "a"}],
        }
        valid, failures = _validate_extraction_output(data, [])
        assert not valid
        assert any("invalid type" in f for f in failures)

    def test_fact_missing_fields(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {
            "title": "T", "summary": "S",
            "records": [{"type": "fact", "subject": ""}],
        }
        valid, failures = _validate_extraction_output(data, [])
        assert not valid

    def test_vague_summary(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {
            "title": "T", "summary": "This document discusses things",
            "records": [{"type": "fact", "subject": "a", "info": "b"}],
        }
        valid, failures = _validate_extraction_output(data, [])
        assert not valid
        assert any("vague" in f.lower() for f in failures)

    def test_valid_procedure_record(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {
            "title": "Deploy Workflow",
            "summary": "Standard deployment uses pytest then docker build",
            "tags": ["deploy"],
            "records": [
                {"type": "procedure", "trigger": "deploying to production",
                 "steps": "run pytest, build docker image, push to registry"},
            ],
        }
        valid, failures = _validate_extraction_output(data, [])
        assert valid, failures

    def test_procedure_missing_fields(self):
        from consolidation_memory.consolidation.prompting import _validate_extraction_output
        data = {
            "title": "T", "summary": "S",
            "records": [{"type": "procedure", "trigger": ""}],
        }
        valid, failures = _validate_extraction_output(data, [])
        assert not valid
        assert any("procedure missing" in f for f in failures)

    def test_extract_with_validation_uses_json_schema(self):
        from consolidation_memory.consolidation.prompting import _llm_extract_with_validation

        captured_schema = []

        def fake_call(prompt, max_retries=3, json_schema=None):
            captured_schema.append(json_schema)
            return (
                '{"title":"T","summary":"Python 3.12 is installed",'
                '"tags":["python"],'
                '"records":[{"type":"fact","subject":"Python","info":"3.12"}]}'
            )

        with patch(
            "consolidation_memory.consolidation.prompting._call_llm",
            side_effect=fake_call,
        ):
            data, calls = _llm_extract_with_validation(
                "prompt",
                [{"content": "python 3.12", "created_at": "2026-01-01"}],
            )

        assert calls == 1
        assert data["title"] == "T"
        assert captured_schema and captured_schema[0] is not None
        assert "records" in captured_schema[0]["properties"]


# ── JSON parsing ─────────────────────────────────────────────────────────────

class TestJsonParsing:
    def test_parse_clean_json(self):
        from consolidation_memory.consolidation.prompting import _parse_llm_json
        data = _parse_llm_json('{"title": "Test", "records": []}')
        assert data == {"title": "Test", "records": []}

    def test_parse_with_code_fences(self):
        from consolidation_memory.consolidation.prompting import _parse_llm_json
        raw = '```json\n{"title": "Test"}\n```'
        data = _parse_llm_json(raw)
        assert data == {"title": "Test"}

    def test_parse_invalid_returns_none(self):
        from consolidation_memory.consolidation.prompting import _parse_llm_json
        assert _parse_llm_json("not json at all") is None


# ── Embedding text generation ────────────────────────────────────────────────

class TestEmbeddingText:
    def test_fact_embedding(self):
        from consolidation_memory.consolidation.prompting import _embedding_text_for_record
        text = _embedding_text_for_record({"type": "fact", "subject": "Python", "info": "3.12"})
        assert text == "Python: 3.12"

    def test_solution_embedding(self):
        from consolidation_memory.consolidation.prompting import _embedding_text_for_record
        text = _embedding_text_for_record({
            "type": "solution", "problem": "crash", "fix": "restart"
        })
        assert text == "Problem: crash. Fix: restart"

    def test_preference_embedding(self):
        from consolidation_memory.consolidation.prompting import _embedding_text_for_record
        text = _embedding_text_for_record({
            "type": "preference", "key": "theme", "value": "dark"
        })
        assert text == "Preference theme: dark"

    def test_procedure_embedding(self):
        from consolidation_memory.consolidation.prompting import _embedding_text_for_record
        text = _embedding_text_for_record({
            "type": "procedure",
            "trigger": "before committing",
            "steps": "run tests then lint",
        })
        assert text == "Procedure: before committing -> run tests then lint"


# ── Markdown rendering ──────────────────────────────────────────────────────

class TestMarkdownRendering:
    def test_renders_all_sections(self):
        from consolidation_memory.consolidation.prompting import _render_markdown_from_records
        records = [
            {"type": "fact", "subject": "Python", "info": "3.12"},
            {"type": "solution", "problem": "Error", "fix": "Fix it", "context": "dev"},
            {"type": "preference", "key": "theme", "value": "dark", "context": "IDE"},
            {"type": "procedure", "trigger": "deploying", "steps": "test then push", "context": "production"},
        ]
        md = _render_markdown_from_records("Title", "Summary", ["tag1"], 0.85, records)
        assert "## Facts" in md
        assert "**Python**: 3.12" in md
        assert "## Solutions" in md
        assert "### Error" in md
        assert "## Preferences" in md
        assert "**theme**: dark (IDE)" in md
        assert "## Procedures" in md
        assert "### deploying" in md
        assert "test then push" in md
        assert "*Context: production*" in md
        assert "confidence: 0.85" in md

    def test_omits_empty_sections(self):
        from consolidation_memory.consolidation.prompting import _render_markdown_from_records
        md = _render_markdown_from_records(
            "Title", "Summary", [], 0.8,
            [{"type": "fact", "subject": "X", "info": "Y"}],
        )
        assert "## Facts" in md
        assert "## Solutions" not in md
        assert "## Preferences" not in md
        assert "## Procedures" not in md


# ── Record cache ─────────────────────────────────────────────────────────────

class TestRecordCache:
    def test_invalidate_bumps_version(self):
        from consolidation_memory import record_cache
        v_before = record_cache._version
        record_cache.invalidate()
        assert record_cache._version == v_before + 1

    def test_empty_records_returns_empty(self, tmp_data_dir):
        ensure_schema()
        from consolidation_memory import record_cache
        record_cache.invalidate()
        records, vecs = record_cache.get_record_vecs()
        assert records == []
        assert vecs is None

    def test_scoped_cache_reuses_embeddings(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="scope-cache.md", title="Scope Cache", summary="S",
            source_episodes=[],
        )
        insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "scope-a"},
            {"record_type": "fact", "content": {}, "embedding_text": "scope-b"},
        ])
        from consolidation_memory import record_cache
        record_cache.invalidate()
        scope = {
            "namespace_slug": "default",
            "project_slug": "default",
            "app_client_name": "legacy_client",
            "app_client_type": "python_sdk",
        }
        with patch("consolidation_memory.record_cache.encode_documents", side_effect=mock_encode) as mock_embed:
            recs1, vecs1 = record_cache.get_record_vecs(include_expired=False, scope=scope)
            recs2, vecs2 = record_cache.get_record_vecs(include_expired=False, scope=scope)
        assert len(recs1) == 2
        assert len(recs2) == 2
        assert vecs1 is not None
        assert vecs2 is not None
        assert mock_embed.call_count == 1

    def test_scoped_cache_invalidate_forces_reembed(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="scope-cache-reset.md", title="Scope Cache Reset", summary="S",
            source_episodes=[],
        )
        insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "scope-reset"},
        ])
        from consolidation_memory import record_cache
        record_cache.invalidate()
        scope = {
            "namespace_slug": "default",
            "project_slug": "default",
            "app_client_name": "legacy_client",
            "app_client_type": "python_sdk",
        }
        with patch("consolidation_memory.record_cache.encode_documents", side_effect=mock_encode) as mock_embed:
            record_cache.get_record_vecs(include_expired=False, scope=scope)
            record_cache.invalidate()
            record_cache.get_record_vecs(include_expired=False, scope=scope)
        assert mock_embed.call_count == 2
