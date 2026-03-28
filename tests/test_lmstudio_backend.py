"""Tests for LM Studio embedding + LLM backend adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from consolidation_memory.backends.lmstudio import (
    LMStudioEmbeddingBackend,
    LMStudioLLMBackend,
)


def _mock_embedding_response(data: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": data}
    return response


class TestLMStudioEmbeddingBackend:
    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_encode_documents_prefixes_input_and_sorts_by_index(self, mock_post):
        mock_post.return_value = _mock_embedding_response(
            [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ]
        )
        backend = LMStudioEmbeddingBackend(
            api_base="http://localhost:1234/v1/",
            model_name="nomic-embed-text-v1.5",
            dimension=2,
        )

        result = backend.encode_documents(["first", "second"])

        assert result.shape == (2, 2)
        # Sorted by index => embedding[0] comes from index=0 payload [3,4] normalized.
        assert np.allclose(result[0], np.array([0.6, 0.8], dtype=np.float32))
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input"] == ["search_document: first", "search_document: second"]

    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_encode_query_prefixes_search_query(self, mock_post):
        mock_post.return_value = _mock_embedding_response([{"index": 0, "embedding": [1.0, 0.0]}])
        backend = LMStudioEmbeddingBackend(
            api_base="http://localhost:1234/v1",
            model_name="nomic-embed-text-v1.5",
            dimension=2,
        )

        result = backend.encode_query("debug drift")

        assert result.shape == (1, 2)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input"] == ["search_query: debug drift"]


class TestLMStudioLLMBackend:
    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_generate_json_raises_when_structured_output_unsupported(self, mock_post):
        request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
        unsupported_response = httpx.Response(422, request=request)
        unsupported_error = httpx.HTTPStatusError(
            "unsupported response_format",
            request=request,
            response=unsupported_response,
        )
        mock_post.side_effect = unsupported_error

        backend = LMStudioLLMBackend(
            api_base="http://localhost:1234/v1",
            model="qwen2.5-14b-instruct",
        )

        with pytest.raises(ValueError, match="requires structured output"):
            backend.generate_json("system", "user", {"type": "object"})

        assert mock_post.call_count == 1
        payload = mock_post.call_args.kwargs["json"]
        assert "response_format" in payload

    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_generate_raises_on_empty_message_content(self, mock_post):
        bad_response = MagicMock()
        bad_response.raise_for_status = MagicMock()
        bad_response.json.return_value = {"choices": [{"message": {"content": None}}]}
        mock_post.return_value = bad_response

        backend = LMStudioLLMBackend(
            api_base="http://localhost:1234/v1",
            model="qwen2.5-14b-instruct",
        )

        with pytest.raises(ValueError, match="empty response"):
            backend.generate("system", "user")
