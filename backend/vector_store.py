from collections.abc import Callable, Sequence
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from backend.embeddings import EmbeddedChunk, generate_embeddings
from backend.dataset_selection import SessionFactory
from backend.rag_preparation import (
    DEFAULT_MAX_CHUNK_CHARS,
    DocumentChunk,
    MetadataValue,
    prepare_selected_dataset_chunks,
)
from backend.runtime_config import CHROMA_STORAGE_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_DIRECTORY = CHROMA_STORAGE_PATH
COLLECTION_NAME = "retailiq_chunks"
DEFAULT_INDEX_BATCH_SIZE = 500
SEARCHABLE_DOCUMENT_CACHE_SIZE = 8


class VectorStoreError(RuntimeError):
    """Base error for persistent vector storage."""


class InvalidIndexInputError(VectorStoreError):
    """Raised when chunks cannot safely be associated with a dataset."""


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when ChromaDB cannot be opened or updated."""


@dataclass(frozen=True)
class IndexingResult:
    upload_id: str
    indexed_chunks: int
    stored_chunks: int
    collection_name: str


def _persistent_client(storage_directory: Path):
    try:
        import chromadb

        storage_directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(storage_directory.resolve()))
    except Exception as exc:
        raise VectorStoreUnavailableError("The vector store is unavailable.") from exc


def _collection(client: Any):
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise VectorStoreUnavailableError("The vector collection is unavailable.") from exc


def _validate_metadata(metadata: dict[str, MetadataValue]) -> dict[str, MetadataValue]:
    if not metadata:
        raise InvalidIndexInputError("Chunk metadata cannot be empty.")
    if any(not isinstance(value, (str, int, float, bool)) for value in metadata.values()):
        raise InvalidIndexInputError("Chunk metadata contains an unsupported value.")
    return dict(metadata)


def _validate_dataset_chunks(
    chunks: Sequence[DocumentChunk], upload_id: str
) -> None:
    chunk_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in chunk_ids:
            raise InvalidIndexInputError("Chunk IDs must be unique.")
        if chunk.metadata.get("upload_id") != upload_id:
            raise InvalidIndexInputError(
                "Every chunk must belong to the requested upload ID."
            )
        _validate_metadata(chunk.metadata)
        chunk_ids.add(chunk.chunk_id)


def _snapshot(collection: Any, upload_id: str) -> dict[str, Any]:
    return collection.get(
        where={"upload_id": upload_id},
        include=["documents", "metadatas", "embeddings"],
    )


def _restore_snapshot(collection: Any, upload_id: str, snapshot: dict[str, Any]) -> None:
    collection.delete(where={"upload_id": upload_id})
    ids = snapshot.get("ids") or []
    if ids:
        collection.upsert(
            ids=ids,
            documents=snapshot.get("documents"),
            metadatas=snapshot.get("metadatas"),
            embeddings=snapshot.get("embeddings"),
        )


class ChromaVectorStore:
    """Persistent Chroma collection with dataset-safe, idempotent indexing."""

    _searchable_document_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    _searchable_document_cache_lock = Lock()

    def __init__(self, storage_directory: Path | None = None):
        self.storage_directory = storage_directory or DEFAULT_CHROMA_DIRECTORY
        self._client = _persistent_client(self.storage_directory)
        self._collection = _collection(self._client)

    def _searchable_cache_key(self, upload_id: UUID | str) -> tuple[str, str]:
        return (str(self.storage_directory.resolve()), str(upload_id))

    def _invalidate_searchable_cache(self, upload_id: UUID | str) -> None:
        key = self._searchable_cache_key(upload_id)
        with self._searchable_document_cache_lock:
            self._searchable_document_cache.pop(key, None)

    def count(self, upload_id: UUID | str) -> int:
        upload_id_value = str(upload_id)
        try:
            return len(
                self._collection.get(
                    where={"upload_id": upload_id_value},
                    include=[],
                )["ids"]
            )
        except Exception as exc:
            raise VectorStoreUnavailableError("Stored chunks could not be counted.") from exc

    def delete(self, upload_id: UUID | str) -> None:
        try:
            self._collection.delete(where={"upload_id": str(upload_id)})
            self._invalidate_searchable_cache(upload_id)
        except Exception as exc:
            raise VectorStoreUnavailableError("Stored chunks could not be deleted.") from exc

    def get(self, upload_id: UUID | str) -> dict[str, Any]:
        try:
            return self._collection.get(
                where={"upload_id": str(upload_id)},
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as exc:
            raise VectorStoreUnavailableError("Stored chunks could not be loaded.") from exc

    def get_documents(self, upload_id: UUID | str) -> dict[str, Any]:
        """Load searchable text and metadata without materializing all embeddings."""
        key = self._searchable_cache_key(upload_id)
        with self._searchable_document_cache_lock:
            cached = self._searchable_document_cache.get(key)
            if cached is not None:
                self._searchable_document_cache.move_to_end(key)
                return cached
        try:
            result = self._collection.get(
                where={"upload_id": str(upload_id)},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                "Stored searchable documents could not be loaded."
            ) from exc
        with self._searchable_document_cache_lock:
            self._searchable_document_cache[key] = result
            self._searchable_document_cache.move_to_end(key)
            while len(self._searchable_document_cache) > SEARCHABLE_DOCUMENT_CACHE_SIZE:
                self._searchable_document_cache.popitem(last=False)
        return result

    def get_embeddings(self, chunk_ids: Sequence[str]) -> dict[str, Any]:
        """Load embeddings for a bounded set of already indexed chunks."""
        if not chunk_ids:
            return {"ids": [], "embeddings": []}
        try:
            return self._collection.get(
                ids=list(chunk_ids),
                include=["embeddings"],
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                "Stored candidate embeddings could not be loaded."
            ) from exc

    def query(
        self,
        upload_id: UUID | str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> dict[str, Any]:
        try:
            return self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                where={"upload_id": str(upload_id)},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreUnavailableError("Vector search is unavailable.") from exc

    def index(
        self,
        upload_id: UUID | str,
        chunks: Sequence[EmbeddedChunk],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    ) -> IndexingResult:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        upload_id_value = str(upload_id).strip()
        if not upload_id_value:
            raise InvalidIndexInputError("upload_id is required.")

        chunk_ids: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in chunk_ids:
                raise InvalidIndexInputError("Chunk IDs must be unique.")
            if chunk.metadata.get("upload_id") != upload_id_value:
                raise InvalidIndexInputError(
                    "Every chunk must belong to the requested upload ID."
                )
            _validate_metadata(chunk.metadata)
            chunk_ids.add(chunk.chunk_id)

        try:
            previous = _snapshot(self._collection, upload_id_value)
        except Exception as exc:
            raise VectorStoreUnavailableError(
                "Existing vector data could not be inspected."
            ) from exc

        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[dict(chunk.metadata) for chunk in batch],
                    embeddings=[chunk.embedding for chunk in batch],
                )

            stale_ids = [
                chunk_id
                for chunk_id in previous.get("ids", [])
                if chunk_id not in chunk_ids
            ]
            if stale_ids:
                self._collection.delete(ids=stale_ids)
        except Exception as exc:
            try:
                _restore_snapshot(self._collection, upload_id_value, previous)
            except Exception as rollback_exc:
                raise VectorStoreUnavailableError(
                    "Indexing failed and the previous vector data could not be restored."
                ) from rollback_exc
            raise VectorStoreUnavailableError(
                "Indexing failed; the previous vector data was restored."
            ) from exc

        self._invalidate_searchable_cache(upload_id_value)

        return IndexingResult(
            upload_id=upload_id_value,
            indexed_chunks=len(chunks),
            stored_chunks=self.count(upload_id_value),
            collection_name=COLLECTION_NAME,
        )


def index_prepared_chunks(
    upload_id: UUID | str,
    chunks: Sequence[DocumentChunk],
    storage_directory: Path | None = None,
    embedding_batch_size: int = 32,
    index_batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    embedding_function: Callable[..., list[EmbeddedChunk]] = generate_embeddings,
) -> IndexingResult:
    """Embed Round 1 chunks and persist them without touching source datasets."""
    upload_id_value = str(upload_id).strip()
    _validate_dataset_chunks(chunks, upload_id_value)
    embedded = embedding_function(chunks, batch_size=embedding_batch_size)
    return ChromaVectorStore(storage_directory).index(
        upload_id_value,
        embedded,
        batch_size=index_batch_size,
    )


def index_selected_dataset(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    upload_storage_directory: Path | None = None,
    vector_storage_directory: Path | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    embedding_batch_size: int = 32,
    index_batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
) -> IndexingResult:
    """Securely load, prepare, embed, and index one existing uploaded dataset."""
    chunks = prepare_selected_dataset_chunks(
        upload_id,
        session_factory=session_factory,
        storage_directory=upload_storage_directory,
        max_chars=max_chunk_chars,
    )
    return index_prepared_chunks(
        upload_id,
        chunks,
        storage_directory=vector_storage_directory,
        embedding_batch_size=embedding_batch_size,
        index_batch_size=index_batch_size,
    )
