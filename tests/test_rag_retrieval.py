import gc
from uuid import UUID

import pytest

from backend.embeddings import EMBEDDING_DIMENSION, EmbeddedChunk, EmbeddingModelError
from backend.rag_retrieval import (
    DatasetNotIndexedError,
    InvalidRetrievalRequestError,
    RetrievalUnavailableError,
    retrieve_semantic_matches,
)
from backend.vector_store import ChromaVectorStore, VectorStoreUnavailableError

FIRST_UPLOAD = UUID("11111111-1111-1111-1111-111111111111")
SECOND_UPLOAD = UUID("22222222-2222-2222-2222-222222222222")


def vector(first, second=0.0):
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def embedded_chunk(upload_id, index, embedding, document):
    return EmbeddedChunk(
        chunk_id=f"{upload_id}-chunk-{index}",
        text=document,
        metadata={
            "upload_id": str(upload_id),
            "row_number": index,
            "source_filename": "sales.csv",
            "region": "West",
            "stored_file_path": "private/location.csv",
        },
        embedding=embedding,
    )


@pytest.fixture
def vector_store(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma")
    yield store
    del store
    gc.collect()
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def test_semantic_retrieval_orders_cosine_scores_and_isolates_dataset(vector_store):
    vector_store.index(
        FIRST_UPLOAD,
        [
            embedded_chunk(
                FIRST_UPLOAD,
                7,
                vector(1.0),
                "Technology accessories generated strong profit.",
            ),
            embedded_chunk(
                FIRST_UPLOAD,
                3,
                vector(0.8, 0.6),
                "Furniture generated moderate sales.",
            ),
        ],
    )
    vector_store.index(
        SECOND_UPLOAD,
        [
            embedded_chunk(
                SECOND_UPLOAD,
                99,
                vector(1.0),
                "This perfect match belongs to a different dataset.",
            )
        ],
    )

    matches = retrieve_semantic_matches(
        FIRST_UPLOAD,
        "Which technology sales were profitable?",
        top_k=5,
        vector_store=vector_store,
        embedding_function=lambda _query: vector(1.0),
    )

    assert len(matches) == 2
    assert [match.rank for match in matches] == [1, 2]
    assert matches[0].similarity_score == pytest.approx(1.0, abs=1e-6)
    assert matches[1].similarity_score == pytest.approx(0.8, abs=1e-6)
    assert [match.source_row for match in matches] == [7, 3]
    assert all(match.metadata["upload_id"] == str(FIRST_UPLOAD) for match in matches)
    assert all("different dataset" not in match.document for match in matches)
    assert all("stored_file_path" not in match.metadata for match in matches)


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_or_invalid_query_is_rejected(query):
    with pytest.raises(InvalidRetrievalRequestError, match="cannot be empty"):
        retrieve_semantic_matches(FIRST_UPLOAD, query)


@pytest.mark.parametrize("top_k", [0, -1, 51, True, 1.5])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(InvalidRetrievalRequestError, match="between 1 and 50"):
        retrieve_semantic_matches(FIRST_UPLOAD, "sales", top_k=top_k)


def test_unindexed_dataset_is_reported(vector_store):
    with pytest.raises(DatasetNotIndexedError, match="has not been indexed"):
        retrieve_semantic_matches(
            FIRST_UPLOAD,
            "sales",
            vector_store=vector_store,
            embedding_function=lambda _query: vector(1.0),
        )


def test_embedding_and_vector_store_failures_are_safe():
    class FailingStore:
        def count(self, upload_id):
            raise VectorStoreUnavailableError("private path detail")

    with pytest.raises(RetrievalUnavailableError, match="retrieval is unavailable"):
        retrieve_semantic_matches(FIRST_UPLOAD, "sales", vector_store=FailingStore())

    class IndexedStore:
        def count(self, upload_id):
            return 1

    def fail_embedding(_query):
        raise EmbeddingModelError("private model detail")

    with pytest.raises(RetrievalUnavailableError, match="retrieval is unavailable"):
        retrieve_semantic_matches(
            FIRST_UPLOAD,
            "sales",
            vector_store=IndexedStore(),
            embedding_function=fail_embedding,
        )


def test_mixed_dataset_result_is_rejected():
    class MixedStore:
        def count(self, upload_id):
            return 1

        def query(self, upload_id, query_embedding, top_k):
            return {
                "ids": [["wrong-dataset-chunk"]],
                "documents": [["private document"]],
                "metadatas": [[{"upload_id": str(SECOND_UPLOAD), "row_number": 1}]],
                "distances": [[0.0]],
            }

    with pytest.raises(RetrievalUnavailableError, match="invalid result"):
        retrieve_semantic_matches(
            FIRST_UPLOAD,
            "sales",
            vector_store=MixedStore(),
            embedding_function=lambda _query: vector(1.0),
        )
