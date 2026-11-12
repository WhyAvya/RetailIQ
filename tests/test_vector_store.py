import gc
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from backend.embeddings import EMBEDDING_DIMENSION, EmbeddedChunk
from backend.rag_preparation import DocumentChunk
from backend.vector_store import (
    ChromaVectorStore,
    InvalidIndexInputError,
    VectorStoreUnavailableError,
    index_prepared_chunks,
    index_selected_dataset,
)

FIRST_UPLOAD = UUID("11111111-1111-1111-1111-111111111111")
SECOND_UPLOAD = UUID("22222222-2222-2222-2222-222222222222")


def embedded_chunk(upload_id, index, value=1.0):
    return EmbeddedChunk(
        chunk_id=f"{upload_id}-chunk-{index}",
        text=f"Document {index}",
        metadata={
            "upload_id": str(upload_id),
            "row_number": index,
            "region": "West",
        },
        embedding=[value] * EMBEDDING_DIMENSION,
    )


def release_chroma_clients():
    from chromadb.api.client import SharedSystemClient

    gc.collect()
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def chroma_directory(tmp_path):
    path = tmp_path / "chroma"
    yield path
    release_chroma_clients()


def test_persistence_reindexing_metadata_and_dataset_isolation(chroma_directory):
    store = ChromaVectorStore(chroma_directory)
    first_chunks = [embedded_chunk(FIRST_UPLOAD, index) for index in range(1, 4)]
    second_chunks = [embedded_chunk(SECOND_UPLOAD, index) for index in range(1, 3)]

    first_result = store.index(FIRST_UPLOAD, first_chunks, batch_size=2)
    store.index(SECOND_UPLOAD, second_chunks)
    repeat_result = store.index(FIRST_UPLOAD, first_chunks, batch_size=1)

    assert first_result.indexed_chunks == first_result.stored_chunks == 3
    assert repeat_result.stored_chunks == 3
    assert store.count(FIRST_UPLOAD) == 3
    assert store.count(SECOND_UPLOAD) == 2

    stored = store.get(FIRST_UPLOAD)
    assert len(stored["ids"]) == 3
    assert all(item["upload_id"] == str(FIRST_UPLOAD) for item in stored["metadatas"])
    assert all(item["region"] == "West" for item in stored["metadatas"])
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in stored["embeddings"])
    assert np.isfinite(stored["embeddings"]).all()

    del store
    release_chroma_clients()
    restarted = ChromaVectorStore(chroma_directory)
    assert restarted.count(FIRST_UPLOAD) == 3
    assert restarted.count(SECOND_UPLOAD) == 2


def test_reindex_removes_only_stale_chunks_for_same_dataset(chroma_directory):
    store = ChromaVectorStore(chroma_directory)
    store.index(
        FIRST_UPLOAD,
        [embedded_chunk(FIRST_UPLOAD, index) for index in range(1, 4)],
    )
    store.index(SECOND_UPLOAD, [embedded_chunk(SECOND_UPLOAD, 1)])

    result = store.index(
        FIRST_UPLOAD,
        [embedded_chunk(FIRST_UPLOAD, index) for index in range(1, 3)],
    )

    assert result.stored_chunks == 2
    assert store.count(FIRST_UPLOAD) == 2
    assert store.count(SECOND_UPLOAD) == 1


def test_searchable_document_cache_is_invalidated_by_reindex(chroma_directory):
    store = ChromaVectorStore(chroma_directory)
    store.index(FIRST_UPLOAD, [embedded_chunk(FIRST_UPLOAD, 1)])
    initial = store.get_documents(FIRST_UPLOAD)
    assert initial["documents"] == ["Document 1"]

    replacement = EmbeddedChunk(
        chunk_id=f"{FIRST_UPLOAD}-replacement",
        text="Replacement document",
        metadata={"upload_id": str(FIRST_UPLOAD), "row_number": 2},
        embedding=[1.0] * EMBEDDING_DIMENSION,
    )
    store.index(FIRST_UPLOAD, [replacement])

    refreshed = store.get_documents(FIRST_UPLOAD)
    assert refreshed["documents"] == ["Replacement document"]
    assert store.get_embeddings([replacement.chunk_id])["ids"] == [replacement.chunk_id]


def test_failed_batched_indexing_restores_previous_dataset(chroma_directory, monkeypatch):
    store = ChromaVectorStore(chroma_directory)
    original = [embedded_chunk(FIRST_UPLOAD, index) for index in range(1, 3)]
    store.index(FIRST_UPLOAD, original)
    original_upsert = store._collection.upsert
    calls = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated storage failure")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store._collection, "upsert", fail_second_batch)
    replacement = [embedded_chunk(FIRST_UPLOAD, index) for index in range(1, 4)]

    with pytest.raises(VectorStoreUnavailableError, match="previous vector data was restored"):
        store.index(FIRST_UPLOAD, replacement, batch_size=1)

    restored = store.get(FIRST_UPLOAD)
    assert set(restored["ids"]) == {chunk.chunk_id for chunk in original}
    assert store.count(FIRST_UPLOAD) == 2


def test_index_prepared_chunks_validates_dataset_and_uses_embedding_function(
    chroma_directory,
):
    prepared = [
        DocumentChunk(
            chunk_id="prepared-1",
            text="Prepared document",
            metadata={"upload_id": str(FIRST_UPLOAD), "row_number": 1},
        )
    ]
    received_batch_sizes = []

    def fake_embedding_function(chunks, batch_size):
        received_batch_sizes.append(batch_size)
        return [
            EmbeddedChunk(
                chunk.chunk_id,
                chunk.text,
                chunk.metadata,
                [0.5] * EMBEDDING_DIMENSION,
            )
            for chunk in chunks
        ]

    result = index_prepared_chunks(
        FIRST_UPLOAD,
        prepared,
        storage_directory=chroma_directory,
        embedding_batch_size=7,
        embedding_function=fake_embedding_function,
    )

    assert received_batch_sizes == [7]
    assert result.indexed_chunks == result.stored_chunks == 1

    with pytest.raises(InvalidIndexInputError, match="requested upload ID"):
        index_prepared_chunks(
            SECOND_UPLOAD,
            prepared,
            storage_directory=chroma_directory,
            embedding_function=fake_embedding_function,
        )


def test_unsupported_metadata_is_rejected(chroma_directory):
    chunk = embedded_chunk(FIRST_UPLOAD, 1)
    invalid = EmbeddedChunk(
        chunk.chunk_id,
        chunk.text,
        {**chunk.metadata, "unsupported": ["value"]},
        chunk.embedding,
    )

    with pytest.raises(InvalidIndexInputError, match="unsupported value"):
        ChromaVectorStore(chroma_directory).index(FIRST_UPLOAD, [invalid])


def test_selected_dataset_indexing_reuses_round_one_preparation(monkeypatch):
    prepared = [
        DocumentChunk(
            chunk_id="prepared-1",
            text="Prepared document",
            metadata={"upload_id": str(FIRST_UPLOAD), "row_number": 1},
        )
    ]
    calls = {}

    def fake_prepare(upload_id, **kwargs):
        calls["prepare"] = (upload_id, kwargs)
        return prepared

    def fake_index(upload_id, chunks, **kwargs):
        calls["index"] = (upload_id, chunks, kwargs)
        return "indexed"

    monkeypatch.setattr("backend.vector_store.prepare_selected_dataset_chunks", fake_prepare)
    monkeypatch.setattr("backend.vector_store.index_prepared_chunks", fake_index)

    result = index_selected_dataset(FIRST_UPLOAD)

    assert result == "indexed"
    assert calls["prepare"][0] == FIRST_UPLOAD
    assert calls["index"][0] == FIRST_UPLOAD
    assert calls["index"][1] is prepared
