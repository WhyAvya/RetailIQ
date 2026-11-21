"""Gemini embedding client with validated, retryable document/query batching."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from dotenv import load_dotenv

from backend.rag_preparation import DocumentChunk, MetadataValue
from backend.runtime_config import (
    EMBEDDING_BATCH_SIZE,
    GEMINI_EMBEDDING_BACKOFF_SECONDS,
    GEMINI_EMBEDDING_MAX_BACKOFF_SECONDS,
    GEMINI_EMBEDDING_RETRY_ATTEMPTS,
    GEMINI_EMBEDDING_TIMEOUT_SECONDS,
)

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
EMBEDDING_INDEX_VERSION = "gemini-embedding-001-768-v1"
DEFAULT_EMBEDDING_BATCH_SIZE = EMBEDDING_BATCH_SIZE
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    """Base error for embedding generation."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when Gemini embedding credentials or SDK are unavailable."""


class EmbeddingQuotaError(EmbeddingError):
    """Raised after Gemini quota/rate-limit retries are exhausted."""


class EmbeddingModelError(EmbeddingError):
    """Raised when Gemini embeddings cannot be generated safely."""


class InvalidEmbeddingInputError(EmbeddingError):
    """Raised when chunks cannot safely be embedded."""


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, MetadataValue]
    embedding: list[float]


@lru_cache(maxsize=1)
def get_embedding_client() -> Any:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EmbeddingConfigurationError("GEMINI_API_KEY is not configured.")
    try:
        from google import genai

        return genai.Client(
            api_key=api_key,
            http_options={"timeout": int(GEMINI_EMBEDDING_TIMEOUT_SECONDS * 1000)},
        )
    except EmbeddingConfigurationError:
        raise
    except Exception as exc:
        raise EmbeddingConfigurationError("The Google Gen AI SDK could not be initialized.") from exc


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


def _status_code(error: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_timeout(error: Exception) -> bool:
    return "timeout" in type(error).__name__.casefold() or "timed out" in str(error).casefold()


def _extract_vectors(response: Any, expected_count: int) -> np.ndarray:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        single = getattr(response, "embedding", None)
        embeddings = [single] if single is not None else None
    if not isinstance(embeddings, (list, tuple)) or len(embeddings) != expected_count:
        raise EmbeddingModelError("Gemini returned an unexpected embedding count.")
    values = [getattr(item, "values", None) for item in embeddings]
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (expected_count, EMBEDDING_DIMENSION):
        raise EmbeddingModelError(
            f"Gemini returned embedding shape {vectors.shape}; expected "
            f"({expected_count}, {EMBEDDING_DIMENSION})."
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingModelError("Gemini returned invalid embedding values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingModelError("Gemini returned a zero-length embedding.")
    return vectors / norms


def _embed_batch(texts: list[str], task_type: str, client: Any | None = None) -> np.ndarray:
    service = client or get_embedding_client()
    attempts = max(1, GEMINI_EMBEDDING_RETRY_ATTEMPTS)
    for attempt in range(attempts):
        try:
            response = service.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=texts,
                config={
                    "task_type": task_type,
                    "output_dimensionality": EMBEDDING_DIMENSION,
                },
            )
            return _extract_vectors(response, len(texts))
        except EmbeddingModelError:
            raise
        except Exception as exc:
            status = _status_code(exc)
            retryable = status in RETRYABLE_STATUS_CODES or _is_timeout(exc)
            if retryable and attempt + 1 < attempts:
                delay = min(
                    GEMINI_EMBEDDING_BACKOFF_SECONDS * (2**attempt),
                    GEMINI_EMBEDDING_MAX_BACKOFF_SECONDS,
                )
                time.sleep(max(0.0, delay))
                continue
            if status == 429:
                raise EmbeddingQuotaError(
                    "Gemini embedding quota is temporarily exhausted; retry later."
                ) from exc
            if status in {401, 403}:
                raise EmbeddingConfigurationError(
                    "Gemini rejected the embedding API credentials."
                ) from exc
            if _is_timeout(exc):
                raise EmbeddingModelError("The Gemini embedding request timed out.") from exc
            raise EmbeddingModelError("Gemini embedding generation failed.") from exc
    raise EmbeddingModelError("Gemini embedding generation failed.")


def generate_embeddings(
    chunks: Sequence[DocumentChunk],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    client: Any | None = None,
) -> list[EmbeddedChunk]:
    """Create normalized 768-dimensional document embeddings in API batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    _validate_chunks(chunks)
    if not chunks:
        return []

    embedded_chunks: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = _embed_batch(
            [chunk.text for chunk in batch],
            DOCUMENT_TASK_TYPE,
            client,
        )
        embedded_chunks.extend(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata={**chunk.metadata, "embedding_version": EMBEDDING_INDEX_VERSION},
                embedding=vector.tolist(),
            )
            for chunk, vector in zip(batch, vectors)
        )
    return embedded_chunks


def generate_query_embedding(query: str, client: Any | None = None) -> list[float]:
    """Embed one query in the same normalized Gemini 768-dimensional space."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidEmbeddingInputError("The retrieval query cannot be empty.")
    return _embed_batch([query.strip()], QUERY_TASK_TYPE, client)[0].tolist()
