"""Tests for backend retry logic, normalization, and error handling."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from consolidation_memory.backends import retry_with_backoff
from consolidation_memory.config import override_config


# ── retry_with_backoff ────────────────────────────────────────────────────────

class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value="ok")
        result = retry_with_backoff(fn, context="test")
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_transient_error(self):
        fn = MagicMock(side_effect=[ConnectionError("fail"), "ok"])
        with patch("consolidation_memory.backends.time.sleep"):
            result = retry_with_backoff(fn, max_retries=3, context="test")
        assert result == "ok"
        assert fn.call_count == 2

    def test_raises_after_max_retries(self):
        fn = MagicMock(side_effect=ConnectionError("fail"))
        with patch("consolidation_memory.backends.time.sleep"):
            with pytest.raises(ConnectionError, match="fail"):
                retry_with_backoff(fn, max_retries=3, context="test")
        assert fn.call_count == 3

    def test_non_transient_error_not_retried(self):
        fn = MagicMock(side_effect=ValueError("not transient"))
        with pytest.raises(ValueError, match="not transient"):
            retry_with_backoff(fn, context="test")
        assert fn.call_count == 1

    def test_custom_transient_exceptions(self):
        fn = MagicMock(side_effect=[RuntimeError("temp"), "ok"])
        with patch("consolidation_memory.backends.time.sleep"):
            result = retry_with_backoff(
                fn, transient_exceptions=(RuntimeError,), context="test",
            )
        assert result == "ok"

    def test_backoff_delays(self):
        fn = MagicMock(side_effect=[TimeoutError(), TimeoutError(), "ok"])
        with patch("consolidation_memory.backends.time.sleep") as mock_sleep:
            retry_with_backoff(fn, max_retries=3, base_delay=1.0, context="test")
        # Delays: 1.0 * 1, 1.0 * 2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


# ── Ollama Backend ────────────────────────────────────────────────────────────

class TestOllamaEmbeddingBackend:
    def test_strips_v1_suffix(self):
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend
        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:1234/v1",
            model_name="nomic-embed-text",
            dimension=768,
        )
        assert backend._api_base == "http://localhost:1234"

    def test_strips_v1_slash_suffix(self):
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend
        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:1234/v1/",
            model_name="nomic-embed-text",
            dimension=768,
        )
        assert backend._api_base == "http://localhost:1234"

    def test_no_strip_without_v1(self):
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend
        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:11434",
            model_name="nomic-embed-text",
            dimension=768,
        )
        assert backend._api_base == "http://localhost:11434"

    def test_normalize(self):
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend
        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:11434",
            model_name="test",
            dimension=3,
        )
        vecs = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        result = backend._normalize(vecs)
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5

    def test_normalize_zero_vector(self):
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend
        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:11434",
            model_name="test",
            dimension=3,
        )
        vecs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = backend._normalize(vecs)
        # Zero vector stays zero, no division by zero
        assert np.allclose(result[0], [0.0, 0.0, 0.0])

    @patch("consolidation_memory.backends.ollama.httpx.post")
    @patch("consolidation_memory.backends.time.sleep")
    def test_encode_documents_retries_on_http_error(self, mock_sleep, mock_post):
        import httpx
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status = MagicMock()

        mock_post.side_effect = [
            httpx.TimeoutException("timeout"),
            mock_response,
        ]

        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:11434",
            model_name="test",
            dimension=3,
        )
        result = backend.encode_documents(["test text"])
        assert result.shape == (1, 3)
        assert mock_post.call_count == 2


class TestOllamaLLMBackend:
    def test_strips_v1_suffix(self):
        from consolidation_memory.backends.ollama import OllamaLLMBackend
        backend = OllamaLLMBackend(api_base="http://localhost:1234/v1")
        assert backend._api_base == "http://localhost:1234"

    @patch("consolidation_memory.backends.ollama.httpx.post")
    @patch("consolidation_memory.backends.time.sleep")
    def test_generate_retries_on_error(self, mock_sleep, mock_post):
        import httpx
        from consolidation_memory.backends.ollama import OllamaLLMBackend

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "generated text"}}
        mock_response.raise_for_status = MagicMock()

        mock_post.side_effect = [
            httpx.TimeoutException("timeout"),
            mock_response,
        ]

        backend = OllamaLLMBackend(api_base="http://localhost:11434")
        result = backend.generate("system", "user prompt")
        assert result == "generated text"
        assert mock_post.call_count == 2


# ── OpenAI Backend ────────────────────────────────────────────────────────────

class TestOpenAIEmbeddingBackend:
    """OpenAI SDK is imported lazily inside __init__, so we mock it via openai module."""

    def _make_backend(self, mock_openai_cls=None, **kwargs):
        """Create an OpenAIEmbeddingBackend with a mocked OpenAI client."""
        defaults = {"model_name": "test", "dimension": 3, "api_key": "key"}
        defaults.update(kwargs)
        if mock_openai_cls is None:
            mock_openai_cls = MagicMock()
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            from importlib import reload
            import consolidation_memory.backends.openai_backend as mod
            reload(mod)
            return mod.OpenAIEmbeddingBackend(**defaults), mock_openai_cls

    def test_normalize(self):
        backend, _ = self._make_backend()
        vecs = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        result = backend._normalize(vecs)
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5

    def test_with_api_base(self):
        mock_cls = MagicMock()
        self._make_backend(mock_openai_cls=mock_cls, api_base="http://custom:8080/v1")
        mock_cls.assert_called_once_with(api_key="key", base_url="http://custom:8080/v1")

    def test_without_api_base(self):
        mock_cls = MagicMock()
        self._make_backend(mock_openai_cls=mock_cls)
        mock_cls.assert_called_once_with(api_key="key")

    def test_encode_documents_with_retry(self):
        mock_client = MagicMock()
        embedding_obj = MagicMock()
        embedding_obj.index = 0
        embedding_obj.embedding = [0.1, 0.2, 0.3]

        response = MagicMock()
        response.data = [embedding_obj]

        mock_client.embeddings.create.side_effect = [
            ConnectionError("temp failure"),
            response,
        ]

        mock_cls = MagicMock(return_value=mock_client)
        with patch("consolidation_memory.backends.time.sleep"):
            backend, _ = self._make_backend(mock_openai_cls=mock_cls)
            result = backend.encode_documents(["test"])
        assert result.shape == (1, 3)
        assert mock_client.embeddings.create.call_count == 2


class TestOpenAILLMBackend:
    def _make_backend(self, mock_openai_cls=None, **kwargs):
        defaults = {"model": "gpt-4o-mini", "api_key": "key"}
        defaults.update(kwargs)
        if mock_openai_cls is None:
            mock_openai_cls = MagicMock()
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            from importlib import reload
            import consolidation_memory.backends.openai_backend as mod
            reload(mod)
            return mod.OpenAILLMBackend(**defaults), mock_openai_cls

    def test_generate_with_retry(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "response text"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.side_effect = [
            TimeoutError("slow"),
            mock_response,
        ]

        mock_cls = MagicMock(return_value=mock_client)
        with patch("consolidation_memory.backends.time.sleep"):
            backend, _ = self._make_backend(mock_openai_cls=mock_cls)
            result = backend.generate("system", "user")
        assert result == "response text"
        assert mock_client.chat.completions.create.call_count == 2


# ── Backend Factory ───────────────────────────────────────────────────────────

class TestBackendFactory:
    def test_unknown_embedding_backend(self):
        with override_config(EMBEDDING_BACKEND="nonexistent"):
            from consolidation_memory.backends import _create_embedding_backend
            with pytest.raises(ValueError, match="Unknown embedding backend"):
                _create_embedding_backend()

    def test_unknown_llm_backend(self):
        with override_config(LLM_BACKEND="nonexistent"):
            from consolidation_memory.backends import _create_llm_backend
            with pytest.raises(ValueError, match="Unknown LLM backend"):
                _create_llm_backend()

    def test_disabled_llm_backend(self):
        with override_config(LLM_BACKEND="disabled"):
            from consolidation_memory.backends import _create_llm_backend
            assert _create_llm_backend() is None
