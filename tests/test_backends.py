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
        from consolidation_memory.backends.base import normalize_l2
        vecs = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        result = normalize_l2(vecs)
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5

    def test_normalize_zero_vector(self):
        from consolidation_memory.backends.base import normalize_l2
        vecs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = normalize_l2(vecs)
        # Zero vector stays zero, no division by zero
        assert np.allclose(result[0], [0.0, 0.0, 0.0])

    def test_normalize_empty_vectors(self):
        from consolidation_memory.backends.base import normalize_l2
        vecs = np.array([], dtype=np.float32)
        result = normalize_l2(vecs)
        assert result.shape == (0, 0)

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

    @patch("consolidation_memory.backends.ollama.httpx.post")
    def test_encode_query_nomic_uses_single_query_prefix(self, mock_post):
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        backend = OllamaEmbeddingBackend(
            api_base="http://localhost:11434",
            model_name="nomic-embed-text",
            dimension=3,
        )
        result = backend.encode_query("test text")
        assert result.shape == (1, 3)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input"] == ["search_query: test text"]


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


# ── FastEmbed Backend ────────────────────────────────────────────────────────

class _FakeFastEmbedModel:
    def __init__(self, dim: int = 3):
        self._dim = dim

    def embed(self, texts):
        return [[1.0] * self._dim for _ in texts]

    def query_embed(self, text):
        return [[1.0] * self._dim]


class TestFastEmbedEmbeddingBackend:
    def test_uses_configured_cache_dir(self, monkeypatch, tmp_path):
        from consolidation_memory.backends.fastembed_backend import FastEmbedEmbeddingBackend

        cache_dir = tmp_path / "fastembed-cache"
        monkeypatch.setenv("CONSOLIDATION_MEMORY_FASTEMBED_CACHE_DIR", str(cache_dir))
        text_embedding_cls = MagicMock(return_value=_FakeFastEmbedModel())

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=text_embedding_cls)}):
            backend = FastEmbedEmbeddingBackend(model_name="test-model")

        assert backend.dimension == 3
        assert cache_dir.is_dir()
        text_embedding_cls.assert_called_once_with("test-model", cache_dir=str(cache_dir))

    def test_recovers_from_missing_cached_model(self, monkeypatch, tmp_path):
        from consolidation_memory.backends.fastembed_backend import FastEmbedEmbeddingBackend

        cache_dir = tmp_path / "fastembed-cache"
        cache_dir.mkdir()
        stale_file = cache_dir / "stale.bin"
        stale_file.write_text("stale")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_FASTEMBED_CACHE_DIR", str(cache_dir))

        text_embedding_cls = MagicMock(
            side_effect=[
                RuntimeError(
                    "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model failed. File doesn't exist"
                ),
                _FakeFastEmbedModel(),
            ]
        )

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=text_embedding_cls)}):
            backend = FastEmbedEmbeddingBackend(model_name="test-model")

        assert backend.dimension == 3
        assert text_embedding_cls.call_count == 2
        assert not stale_file.exists()

    def test_does_not_retry_non_cache_errors(self, monkeypatch, tmp_path):
        from consolidation_memory.backends.fastembed_backend import FastEmbedEmbeddingBackend

        cache_dir = tmp_path / "fastembed-cache"
        monkeypatch.setenv("CONSOLIDATION_MEMORY_FASTEMBED_CACHE_DIR", str(cache_dir))
        text_embedding_cls = MagicMock(side_effect=RuntimeError("unexpected failure"))

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=text_embedding_cls)}):
            with pytest.raises(RuntimeError, match="unexpected failure"):
                FastEmbedEmbeddingBackend(model_name="test-model")

        assert text_embedding_cls.call_count == 1


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
        from consolidation_memory.backends.base import normalize_l2
        vecs = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        result = normalize_l2(vecs)
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


# ── LM Studio Backend ────────────────────────────────────────────────────────

class TestLMStudioLLMBackend:
    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_generate_does_not_retry_http_400(self, mock_post):
        import httpx
        from consolidation_memory.backends.lmstudio import LMStudioLLMBackend

        request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
        response = httpx.Response(400, request=request, text='{"error":"bad request"}')
        err = httpx.HTTPStatusError("bad request", request=request, response=response)
        mock_post.side_effect = err

        backend = LMStudioLLMBackend(api_base="http://localhost:1234/v1", model="test-model")
        with pytest.raises(httpx.HTTPStatusError):
            backend.generate("system", "user")

        assert mock_post.call_count == 1

    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_generate_json_uses_reasoning_content_when_content_empty(self, mock_post):
        from consolidation_memory.backends.lmstudio import LMStudioLLMBackend

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": '{"title":"x","summary":"y","tags":[],"records":[]}',
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        backend = LMStudioLLMBackend(api_base="http://localhost:1234/v1", model="test-model")
        result = backend.generate_json("system", "user", {"type": "object"})

        assert result == '{"title":"x","summary":"y","tags":[],"records":[]}'
        payload = mock_post.call_args.kwargs["json"]
        assert payload["response_format"]["type"] == "json_schema"

    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_generate_json_raises_when_structured_output_rejected(self, mock_post):
        import httpx
        from consolidation_memory.backends.lmstudio import LMStudioLLMBackend

        request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
        response = httpx.Response(
            422,
            request=request,
            text='{"error":"unsupported response_format"}',
        )

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "unprocessable entity",
            request=request,
            response=response,
        )
        mock_post.return_value = mock_response

        backend = LMStudioLLMBackend(api_base="http://localhost:1234/v1", model="test-model")
        with pytest.raises(ValueError, match="rejected response_format=json_schema"):
            backend.generate_json("system", "user", {"type": "object"})

        # No silent fallback request without response_format.
        assert mock_post.call_count == 1

    @patch("consolidation_memory.backends.lmstudio.httpx.post")
    def test_generate_json_raises_when_content_and_reasoning_empty(self, mock_post):
        from consolidation_memory.backends.lmstudio import LMStudioLLMBackend

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "   ",
                        "reasoning_content": "   ",
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        backend = LMStudioLLMBackend(api_base="http://localhost:1234/v1", model="test-model")
        with pytest.raises(ValueError, match="message\\.content is empty"):
            backend.generate_json("system", "user", {"type": "object"})


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
