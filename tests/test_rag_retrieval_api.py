from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.rag_retrieval import (
    DatasetNotIndexedError,
    RetrievalMatch,
    RetrievalUnavailableError,
)
from backend.upload_metadata import DatabaseOperationError

client = TestClient(main_module.app)


def test_retrieval_endpoint_returns_validated_ranked_response(monkeypatch):
    upload_id = uuid4()
    monkeypatch.setattr(
        main_module,
        "get_upload_metadata",
        lambda selected_id: {"upload_id": str(selected_id), "filename": "sales.csv"},
    )
    monkeypatch.setattr(
        main_module,
        "retrieve_semantic_matches",
        lambda selected_id, query, top_k: [
            RetrievalMatch(
                rank=1,
                chunk_id="chunk-1",
                document="Dataset: sales.csv\nSource row: 4\nSales: 500",
                source_row=4,
                metadata={
                    "upload_id": str(selected_id),
                    "row_number": 4,
                    "region": "West",
                },
                similarity_score=0.91,
            )
        ],
    )

    response = client.post(
        "/api/rag/retrieve",
        json={
            "upload_id": str(upload_id),
            "query": "  highest sales in the west  ",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["upload_id"] == str(upload_id)
    assert body["query"] == "highest sales in the west"
    assert body["matches"][0]["rank"] == 1
    assert body["matches"][0]["source_row"] == 4
    assert body["matches"][0]["similarity_score"] == 0.91
    assert "stored_file_path" not in str(body)


@pytest.mark.parametrize(
    "payload",
    [
        {"upload_id": "not-a-uuid", "query": "sales"},
        {"upload_id": str(uuid4()), "query": "   "},
        {"upload_id": str(uuid4()), "query": "sales", "top_k": 0},
        {"upload_id": str(uuid4()), "query": "sales", "top_k": 51},
        {"upload_id": str(uuid4()), "query": "sales", "top_k": True},
    ],
)
def test_retrieval_endpoint_rejects_invalid_requests(payload):
    response = client.post("/api/rag/retrieve", json=payload)

    assert response.status_code == 422


def test_retrieval_endpoint_handles_missing_and_unindexed_uploads(monkeypatch):
    upload_id = uuid4()
    monkeypatch.setattr(main_module, "get_upload_metadata", lambda _upload_id: None)

    missing = client.post(
        "/api/rag/retrieve",
        json={"upload_id": str(upload_id), "query": "sales"},
    )

    assert missing.status_code == 404
    assert missing.json()["detail"] == "The selected dataset was not found."

    monkeypatch.setattr(
        main_module,
        "get_upload_metadata",
        lambda selected_id: {"upload_id": str(selected_id)},
    )

    def fail_unindexed(*args):
        raise DatasetNotIndexedError("private vector detail")

    monkeypatch.setattr(main_module, "retrieve_semantic_matches", fail_unindexed)
    unindexed = client.post(
        "/api/rag/retrieve",
        json={"upload_id": str(upload_id), "query": "sales"},
    )

    assert unindexed.status_code == 404
    assert unindexed.json()["detail"] == (
        "The selected dataset has not been indexed for retrieval."
    )
    assert "private" not in unindexed.text


def test_retrieval_endpoint_maps_database_and_vector_failures_safely(monkeypatch):
    upload_id = uuid4()

    def fail_database(_upload_id):
        raise DatabaseOperationError("private database detail")

    monkeypatch.setattr(main_module, "get_upload_metadata", fail_database)
    database_failure = client.post(
        "/api/rag/retrieve",
        json={"upload_id": str(upload_id), "query": "sales"},
    )
    assert database_failure.status_code == 503
    assert database_failure.json()["detail"] == (
        "Dataset metadata is temporarily unavailable."
    )
    assert "private" not in database_failure.text

    monkeypatch.setattr(
        main_module,
        "get_upload_metadata",
        lambda selected_id: {"upload_id": str(selected_id)},
    )

    def fail_vector(*args):
        raise RetrievalUnavailableError("private vector detail")

    monkeypatch.setattr(main_module, "retrieve_semantic_matches", fail_vector)
    vector_failure = client.post(
        "/api/rag/retrieve",
        json={"upload_id": str(upload_id), "query": "sales"},
    )
    assert vector_failure.status_code == 503
    assert vector_failure.json()["detail"] == (
        "Semantic retrieval is temporarily unavailable."
    )
    assert "private" not in vector_failure.text
