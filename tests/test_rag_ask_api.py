from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.llm_service import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from backend.rag_pipeline import (
    AnswerSource,
    PipelineDatasetNotFoundError,
    RAGAnswer,
)
from backend.rag_retrieval import DatasetNotIndexedError, RetrievalUnavailableError
from backend.upload_metadata import DatabaseOperationError

client = TestClient(main_module.app)


def test_ask_endpoint_returns_semantic_answer_and_sources(monkeypatch):
    upload_id = uuid4()
    monkeypatch.setattr(
        main_module,
        "answer_question",
        lambda selected_id, question, top_k: RAGAnswer(
            str(selected_id),
            question,
            "Row 1 shows a profitable accessory order.",
            "semantic_rag",
            [
                AnswerSource(
                    "chunk-1",
                    1,
                    0.72,
                    {"upload_id": str(selected_id), "category": "Technology"},
                    "Source row: 1; Profit: 300",
                )
            ],
        ),
    )

    response = client.post(
        "/api/rag/ask",
        json={
            "upload_id": str(upload_id),
            "question": "  Which accessory orders were profitable?  ",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Which accessory orders were profitable?"
    assert body["mode"] == "semantic_rag"
    assert body["sources"][0]["source_row"] == 1
    assert body["sources"][0]["similarity_score"] == 0.72
    assert "stored_file_path" not in str(body)


def test_ask_endpoint_returns_analytics_without_sources(monkeypatch):
    upload_id = uuid4()
    monkeypatch.setattr(
        main_module,
        "answer_question",
        lambda selected_id, question, top_k: RAGAnswer(
            str(selected_id),
            question,
            "Total revenue is 350.00.",
            "analytics",
            [],
            {"intent": "total_revenue", "available": True, "value": 350.0},
        ),
    )

    response = client.post(
        "/api/rag/ask",
        json={"upload_id": str(upload_id), "question": "What is total revenue?"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "analytics"
    assert response.json()["sources"] == []
    assert response.json()["calculation"]["value"] == 350.0


@pytest.mark.parametrize(
    "payload",
    [
        {"upload_id": "invalid", "question": "Question"},
        {"upload_id": str(uuid4()), "question": "   "},
        {"upload_id": str(uuid4()), "question": "Question", "top_k": 0},
        {"upload_id": str(uuid4()), "question": "Question", "top_k": True},
        {"upload_id": str(uuid4()), "question": "Question", "top_k": 51},
    ],
)
def test_ask_endpoint_validates_request(payload):
    assert client.post("/api/rag/ask", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (PipelineDatasetNotFoundError("private"), 404, "The selected dataset was not found."),
        (DatasetNotIndexedError("private"), 404, "The selected dataset has not been indexed for retrieval."),
        (DatabaseOperationError("private"), 503, "Dataset metadata is temporarily unavailable."),
        (RetrievalUnavailableError("private"), 503, "Semantic retrieval is temporarily unavailable."),
        (LLMConfigurationError("private"), 503, "The language model is not configured."),
        (LLMTimeoutError("private"), 504, "The language model request timed out."),
        (LLMRateLimitError("private"), 503, "The language model is temporarily rate limited."),
        (LLMAuthenticationError("private"), 502, "The language model provider rejected authentication."),
        (LLMInvalidResponseError("private"), 502, "The language model service returned an unusable response."),
    ],
)
def test_ask_endpoint_maps_service_errors_safely(monkeypatch, error, status, detail):
    def fail(*_args):
        raise error

    monkeypatch.setattr(main_module, "answer_question", fail)
    response = client.post(
        "/api/rag/ask",
        json={"upload_id": str(uuid4()), "question": "Question"},
    )

    assert response.status_code == status
    assert response.json()["detail"] == detail
    assert "private" not in response.text
