from uuid import uuid4

import numpy as np
import pytest

from backend.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingModelError,
    InvalidEmbeddingInputError,
    generate_embeddings,
    generate_query_embedding,
    get_embedding_model,
)
from backend.rag_preparation import DocumentChunk


class FakeEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, sentences, **kwargs):
        self.calls.append(sentences)
        return np.ones((len(sentences), EMBEDDING_DIMENSION), dtype=np.float32)


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


def test_embeddings_are_batched_finite_and_384_dimensions():
    encoder = FakeEncoder()
    chunks = make_chunks(5)

    embedded = generate_embeddings(chunks, batch_size=2, model=encoder)

    assert [len(call) for call in encoder.calls] == [2, 2, 1]
    assert len(embedded) == 5
    assert all(len(item.embedding) == EMBEDDING_DIMENSION for item in embedded)
    assert all(np.isfinite(item.embedding).all() for item in embedded)
    assert [item.chunk_id for item in embedded] == [chunk.chunk_id for chunk in chunks]
    assert embedded[0].metadata == chunks[0].metadata


def test_empty_input_does_not_load_or_call_model():
    encoder = FakeEncoder()

    assert generate_embeddings([], model=encoder) == []
    assert encoder.calls == []


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_chunk_text_is_rejected(text):
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        text=text,
        metadata={"upload_id": str(uuid4())},
    )

    with pytest.raises(InvalidEmbeddingInputError, match="text cannot be empty"):
        generate_embeddings([chunk], model=FakeEncoder())


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1, 10), dtype=np.float32),
        np.full((1, EMBEDDING_DIMENSION), np.nan, dtype=np.float32),
    ],
)
def test_invalid_model_output_is_rejected(vectors):
    class InvalidEncoder:
        def encode(self, sentences, **kwargs):
            return vectors

    with pytest.raises(EmbeddingModelError):
        generate_embeddings(make_chunks(1), model=InvalidEncoder())


def test_model_execution_failure_is_wrapped():
    class FailingEncoder:
        def encode(self, sentences, **kwargs):
            raise OSError("private model error")

    with pytest.raises(EmbeddingModelError, match="generation failed"):
        generate_embeddings(make_chunks(1), model=FailingEncoder())


def test_model_is_loaded_once_and_loading_failures_are_wrapped(monkeypatch):
    import sentence_transformers

    loaded = []
    encoder = FakeEncoder()
    get_embedding_model.cache_clear()
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        lambda model_name: loaded.append(model_name) or encoder,
    )

    assert get_embedding_model() is encoder
    assert get_embedding_model() is encoder
    assert len(loaded) == 1

    get_embedding_model.cache_clear()

    def fail_loading(model_name):
        raise OSError("private loading error")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail_loading)
    with pytest.raises(EmbeddingModelError, match="could not be loaded"):
        get_embedding_model()
    get_embedding_model.cache_clear()


def test_query_embedding_uses_same_validated_model_output():
    encoder = FakeEncoder()

    embedding = generate_query_embedding("  profitable technology sales  ", encoder)

    assert encoder.calls == [["profitable technology sales"]]
    assert len(embedding) == EMBEDDING_DIMENSION
    assert np.isfinite(embedding).all()

    with pytest.raises(InvalidEmbeddingInputError, match="cannot be empty"):
        generate_query_embedding("   ", encoder)
