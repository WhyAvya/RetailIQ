from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from backend.rag_preparation import DocumentChunk, MetadataValue

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """Base error for embedding generation."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the sentence-transformer model cannot be loaded or used."""


class InvalidEmbeddingInputError(EmbeddingError):
    """Raised when chunks cannot safely be embedded."""


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, MetadataValue]
    embedding: list[float]


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceEncoder:
    """Load the configured model once and reuse it for later batches."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        raise EmbeddingModelError("The embedding model could not be loaded.") from exc


def _validate_chunks(chunks: Sequence[DocumentChunk]) -> None:
    chunk_ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, DocumentChunk):
            raise InvalidEmbeddingInputError("Every item must be a DocumentChunk.")
        if not chunk.chunk_id.strip():
            raise InvalidEmbeddingInputError("Chunk IDs cannot be empty.")
        if chunk.chunk_id in chunk_ids:
            raise InvalidEmbeddingInputError("Chunk IDs must be unique.")
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            raise InvalidEmbeddingInputError("Chunk text cannot be empty.")
        chunk_ids.add(chunk.chunk_id)


def _encode_texts(
    texts: list[str],
    batch_size: int,
    model: SentenceEncoder | None,
) -> np.ndarray:
    encoder = model or get_embedding_model()
    try:
        encoded = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise EmbeddingModelError("Embedding generation failed.") from exc

    vectors = np.asarray(encoded, dtype=np.float32)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if vectors.shape != expected_shape:
        raise EmbeddingModelError(
            f"The embedding model returned shape {vectors.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingModelError("The embedding model returned invalid values.")
    return vectors


def generate_embeddings(
    chunks: Sequence[DocumentChunk],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    model: SentenceEncoder | None = None,
) -> list[EmbeddedChunk]:
    """Create validated 384-dimensional embeddings in memory-safe batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    _validate_chunks(chunks)
    if not chunks:
        return []

    embedded_chunks: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = _encode_texts(
            [chunk.text for chunk in batch],
            batch_size,
            model,
        )

        embedded_chunks.extend(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                embedding=vector.tolist(),
            )
            for chunk, vector in zip(batch, vectors)
        )
    return embedded_chunks


def generate_query_embedding(
    query: str,
    model: SentenceEncoder | None = None,
) -> list[float]:
    """Embed one validated natural-language query with the indexing model."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidEmbeddingInputError("The retrieval query cannot be empty.")
    return _encode_texts([query.strip()], 1, model)[0].tolist()
