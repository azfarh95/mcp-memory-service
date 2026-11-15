"""Unit tests for the Voyage cloud embedding adapter.

Mirrors tests/test_external_embeddings.py: requests.post is mocked, no network.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mcp_memory_service.embeddings.voyage_embeddings import (
    VoyageEmbeddingModel,
    get_voyage_embedding_model,
)


def _voyage_response(dim: int, count: int) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {"index": i, "embedding": [0.1] * dim} for i in range(count)
        ]
    }
    return response


@pytest.fixture
def no_keyring(monkeypatch):
    monkeypatch.setenv("MCP_VOYAGE_API_KEY", "test-key-123")
    yield


class TestVoyageEmbeddingModel:
    def test_encode_returns_numpy_array(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024)
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            mock_post.return_value = _voyage_response(1024, 2)
            result = model.encode(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1024)
        assert result.dtype == np.float32

    def test_encode_sends_voyage_payload_shape(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024, model_name="voyage-4")
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            mock_post.return_value = _voyage_response(1024, 1)
            model.encode(["test"], input_type="query")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "voyage-4"
        assert payload["input"] == ["test"]
        assert payload["input_type"] == "query"
        assert payload["output_dimension"] == 1024

    def test_single_string_input(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=512)
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            mock_post.return_value = _voyage_response(512, 1)
            result = model.encode("just one text")
        assert result.shape == (1, 512)

    def test_batching_splits_requests(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=256, batch_size=2)
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            # Each call must return exactly as many embeddings as its input batch.
            mock_post.side_effect = [
                _voyage_response(256, 2),
                _voyage_response(256, 2),
                _voyage_response(256, 1),
            ]
            result = model.encode(["a", "b", "c", "d", "e"])
        assert mock_post.call_count == 3  # 2 + 2 + 1
        assert result.shape == (5, 256)

    def test_dimension_mismatch_raises(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024)
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            mock_post.return_value = _voyage_response(999, 1)
            with pytest.raises(RuntimeError, match="Voyage embedding request failed"):
                model.encode(["text"])

    def test_client_error_fails_fast_without_retries(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024, max_retries=5)
        response = MagicMock()
        response.status_code = 401
        response.text = '{"error": "invalid api key"}'
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post, \
             patch("mcp_memory_service.embeddings.voyage_embeddings.time.sleep") as mock_sleep:
            mock_post.return_value = response
            with pytest.raises(RuntimeError):
                model.encode(["text"])
        assert mock_post.call_count == 1  # 4xx: no retry
        mock_sleep.assert_not_called()

    def test_rate_limit_retries_then_raises(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024, max_retries=2)
        response = MagicMock()
        response.status_code = 429
        response.headers = {}
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post, \
             patch("mcp_memory_service.embeddings.voyage_embeddings.time.sleep") as mock_sleep:
            mock_post.return_value = response
            with pytest.raises(RuntimeError):
                model.encode(["text"])
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 2  # one backoff sleep per attempted retry

    def test_retry_after_header_honored(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024, max_retries=2, retry_base_delay=5.0)
        response = MagicMock()
        response.status_code = 429
        response.headers = {"Retry-After": "7"}
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post, \
             patch("mcp_memory_service.embeddings.voyage_embeddings.time.sleep") as mock_sleep:
            mock_post.return_value = response
            with pytest.raises(RuntimeError):
                model.encode(["text"])
        # attempt 0 waits Retry-After (7s); attempt 1 falls back to exponential (10s)
        mock_sleep.assert_any_call(7.0)
        assert mock_sleep.call_count == 2

    def test_api_key_never_in_error_text(self, no_keyring, caplog):
        secret = "test-key-123"
        model = VoyageEmbeddingModel(output_dimension=1024, max_retries=1)
        response = MagicMock()
        response.status_code = 400
        response.text = f'bad request for key {secret}'
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            mock_post.return_value = response
            with pytest.raises(RuntimeError):
                model.encode(["text"])
        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("MCP_VOYAGE_API_KEY", raising=False)
        monkeypatch.setattr(
            "mcp_memory_service.embeddings.voyage_embeddings.VoyageEmbeddingModel._resolve_api_key",
            staticmethod(lambda service, username: None),
        )
        model = VoyageEmbeddingModel(output_dimension=1024)
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            with pytest.raises(RuntimeError, match="Voyage embedding request failed"):
                model.encode(["text"])
        mock_post.assert_not_called()

    def test_truncates_long_texts(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024, max_chars=10)
        with patch("mcp_memory_service.embeddings.voyage_embeddings.requests.post") as mock_post:
            mock_post.return_value = _voyage_response(1024, 1)
            model.encode(["x" * 1000])
        assert mock_post.call_args.kwargs["json"]["input"] == ["x" * 10]

    def test_get_sentence_embedding_dimension(self, no_keyring):
        model = VoyageEmbeddingModel(output_dimension=1024)
        assert model.get_sentence_embedding_dimension() == 1024
        assert model.embedding_dimension == 1024
        assert model.supports_input_type is True

    def test_factory(self, no_keyring):
        model = get_voyage_embedding_model(output_dimension=1024)
        assert isinstance(model, VoyageEmbeddingModel)
