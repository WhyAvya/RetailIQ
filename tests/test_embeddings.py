from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

import backend.embeddings as embeddings
from backend.embeddings import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSION,
    EMBEDDING_INDEX_VERSION,
    QUERY_TASK_TYPE,
    EmbeddingConfigurationError,
    EmbeddingModelError,
    EmbeddingQuotaError,
    InvalidEmbeddingInputError,
    generate_embeddings,
    generate_query_embedding,
    get_embedding_client,
)
from backend.rag_preparation import DocumentChunk


class FakeModels:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = list(outcomes or [])

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return embedding_response(len(kwargs["contents"]))


class FakeClient:
    def __init__(self, outcomes=None):
        self.models = FakeModels(outcomes)


class FakeApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"private API error {status_code}")
        self.status_code = status_code


def embedding_response(count, dimension=EMBEDDING_DIMENSION, value=1.0):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=[value] * dimension) for _ in range(count)]
    )


def make_chunks(count=3):
    upload_id = str(uuid4())
    return [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            text=f"Document {index}",
            metadata={"upload_id": upload_id, "row_number": index + 1},
        )
        for index in range(count)
    ]


def test_document_embeddings_are_batched_normalized_and_768_dimensions():
    client = FakeClient()
    chunks = make_chunks(5)

    embedded = generate_embeddings(chunks, batch_size=2, client=client)

    assert [len(call["contents"]) for call in client.models.calls] == [2, 2, 1]
    assert all(
        call["config"]
        == {
            "task_type": DOCUMENT_TASK_TYPE,
            "output_dimensionality": EMBEDDING_DIMENSION,
        }
        for call in client.models.calls
    )
    assert len(embedded) == 5
    assert all(len(item.embedding) == EMBEDDING_DIMENSION for item in embedded)
    assert all(np.isfinite(item.embedding).all() for item in embedded)
    assert all(np.linalg.norm(item.embedding) == pytest.approx(1.0) for item in embedded)
    assert [item.chunk_id for item in embedded] == [chunk.chunk_id for chunk in chunks]
    assert embedded[0].metadata == {
        **chunks[0].metadata,
        "embedding_version": EMBEDDING_INDEX_VERSION,
    }


def test_empty_input_does_not_initialize_or_call_client():
    client = FakeClient()

    assert generate_embeddings([], client=client) == []
    assert client.models.calls == []


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_chunk_text_is_rejected(text):
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        text=text,
        metadata={"upload_id": str(uuid4())},
    )

    with pytest.raises(InvalidEmbeddingInputError, match="text cannot be empty"):
        generate_embeddings([chunk], client=FakeClient())


@pytest.mark.parametrize(
    "response",
    [
        embedding_response(1, dimension=10),
        SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float("nan")] * EMBEDDING_DIMENSION)
            ]
        ),
        embedding_response(1, value=0.0),
        embedding_response(2),
    ],
)
def test_invalid_api_output_is_rejected(response):
    with pytest.raises(EmbeddingModelError):
        generate_embeddings(make_chunks(1), client=FakeClient([response]))


def test_query_embedding_uses_retrieval_query_task_type():
    client = FakeClient()

    embedding = generate_query_embedding("  profitable technology sales  ", client)

    assert client.models.calls[0]["contents"] == ["profitable technology sales"]
    assert client.models.calls[0]["config"]["task_type"] == QUERY_TASK_TYPE
    assert client.models.calls[0]["config"]["output_dimensionality"] == 768
    assert len(embedding) == EMBEDDING_DIMENSION
    assert np.linalg.norm(embedding) == pytest.approx(1.0)

    with pytest.raises(InvalidEmbeddingInputError, match="cannot be empty"):
        generate_query_embedding("   ", client)


def test_retryable_api_error_uses_exponential_backoff(monkeypatch):
    client = FakeClient([FakeApiError(503), FakeApiError(500), embedding_response(1)])
    delays = []
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_BACKOFF_SECONDS", 0.25)
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_MAX_BACKOFF_SECONDS", 1.0)
    monkeypatch.setattr(embeddings.time, "sleep", delays.append)

    result = generate_query_embedding("sales", client)

    assert len(result) == EMBEDDING_DIMENSION
    assert delays == [0.25, 0.5]
    assert len(client.models.calls) == 3


def test_quota_exhaustion_is_safe_and_bounded(monkeypatch):
    client = FakeClient([FakeApiError(429), FakeApiError(429)])
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _delay: None)

    with pytest.raises(EmbeddingQuotaError, match="quota is temporarily exhausted"):
        generate_query_embedding("sales", client)

    assert len(client.models.calls) == 2


def test_authentication_failure_is_not_retried(monkeypatch):
    client = FakeClient([FakeApiError(401)])
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_RETRY_ATTEMPTS", 4)

    with pytest.raises(EmbeddingConfigurationError, match="credentials"):
        generate_query_embedding("sales", client)

    assert len(client.models.calls) == 1


def test_missing_api_key_fails_without_calling_sdk(monkeypatch):
    get_embedding_client.cache_clear()
    monkeypatch.setattr(embeddings, "load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(EmbeddingConfigurationError, match="GEMINI_API_KEY"):
        get_embedding_client()

    get_embedding_client.cache_clear()


def test_sdk_client_is_initialized_once_with_configured_timeout(monkeypatch):
    from google import genai

    calls = []
    fake_client = object()
    get_embedding_client.cache_clear()
    monkeypatch.setattr(embeddings, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_TIMEOUT_SECONDS", 12.5)
    monkeypatch.setattr(
        genai,
        "Client",
        lambda **kwargs: calls.append(kwargs) or fake_client,
    )

    assert get_embedding_client() is fake_client
    assert get_embedding_client() is fake_client
    assert calls == [
        {
            "api_key": "test-only-key",
            "http_options": {"timeout": 12500},
        }
    ]

    get_embedding_client.cache_clear()
