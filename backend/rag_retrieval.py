from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any
from uuid import UUID

from backend.embeddings import EmbeddingError, generate_query_embedding
from backend.rag_preparation import MetadataValue
from backend.vector_store import ChromaVectorStore, VectorStoreUnavailableError

DEFAULT_TOP_K = 5
MAX_TOP_K = 50


class RetrievalError(RuntimeError):
    """Base error for semantic retrieval."""


class InvalidRetrievalRequestError(RetrievalError):
    """Raised when query parameters are invalid."""


class DatasetNotIndexedError(RetrievalError):
    """Raised when an upload has no chunks in the vector store."""


class RetrievalUnavailableError(RetrievalError):
    """Raised when embeddings or vector search are unavailable."""


@dataclass(frozen=True)
class RetrievalMatch:
    rank: int
    chunk_id: str
    document: str
    source_row: int | None
    metadata: dict[str, MetadataValue]
    similarity_score: float


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
        and not str(key).lower().endswith(("_path", "filepath"))
        and str(key).lower() != "path"
    }


def _first_query_results(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key)
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")
    return values[0]


def retrieve_semantic_matches(
    upload_id: UUID | str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    vector_store: ChromaVectorStore | None = None,
    embedding_function: Callable[[str], list[float]] = generate_query_embedding,
) -> list[RetrievalMatch]:
    """Retrieve ranked chunks for exactly one indexed upload."""
    upload_id_value = str(upload_id).strip()
    if not upload_id_value:
        raise InvalidRetrievalRequestError("upload_id is required.")
    if not isinstance(query, str) or not query.strip():
        raise InvalidRetrievalRequestError("The retrieval query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise InvalidRetrievalRequestError(
            f"top_k must be between 1 and {MAX_TOP_K}."
        )

    try:
        store = vector_store or ChromaVectorStore()
        stored_count = store.count(upload_id_value)
        if stored_count == 0:
            raise DatasetNotIndexedError("The selected dataset has not been indexed.")
        query_embedding = embedding_function(query.strip())
        result = store.query(
            upload_id_value,
            query_embedding,
            min(top_k, stored_count),
        )
    except DatasetNotIndexedError:
        raise
    except (EmbeddingError, VectorStoreUnavailableError) as exc:
        raise RetrievalUnavailableError("Semantic retrieval is unavailable.") from exc

    ids = _first_query_results(result, "ids")
    documents = _first_query_results(result, "documents")
    metadatas = _first_query_results(result, "metadatas")
    distances = _first_query_results(result, "distances")
    if not len(ids) == len(documents) == len(metadatas) == len(distances):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")

    matches = []
    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        if not isinstance(metadata, dict) or metadata.get("upload_id") != upload_id_value:
            raise RetrievalUnavailableError("Vector search returned an invalid result.")
        if not isinstance(chunk_id, str) or not isinstance(document, str):
            raise RetrievalUnavailableError("Vector search returned an invalid result.")
        try:
            similarity = 1.0 - float(distance)
        except (TypeError, ValueError) as exc:
            raise RetrievalUnavailableError(
                "Vector search returned an invalid result."
            ) from exc
        if not math.isfinite(similarity):
            raise RetrievalUnavailableError("Vector search returned an invalid result.")
        similarity = max(-1.0, min(1.0, similarity))

        public_metadata = _safe_metadata(metadata)
        source_row_value = public_metadata.get("row_number")
        source_row = (
            source_row_value
            if isinstance(source_row_value, int) and not isinstance(source_row_value, bool)
            else None
        )
        matches.append(
            RetrievalMatch(
                rank=0,
                chunk_id=chunk_id,
                document=document,
                source_row=source_row,
                metadata=public_metadata,
                similarity_score=similarity,
            )
        )

    matches.sort(key=lambda match: match.similarity_score, reverse=True)
    return [
        RetrievalMatch(
            rank=rank,
            chunk_id=match.chunk_id,
            document=match.document,
            source_row=match.source_row,
            metadata=match.metadata,
            similarity_score=match.similarity_score,
        )
        for rank, match in enumerate(matches, start=1)
    ]
