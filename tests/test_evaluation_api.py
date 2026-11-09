from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from evaluation.report_service import EvaluationReportUnavailableError


client = TestClient(app)


def test_evaluation_report_returns_actual_pending_and_fixture_results():
    response = client.get("/api/evaluation/report")

    assert response.status_code == 200
    body = response.json()
    retrieval = body["retrieval"]
    assert retrieval["result_kind"] == "actual_chromadb"
    assert retrieval["semantic_questions_evaluated"] == 15
    assert retrieval["aggregate_metrics"] == {
        "precision_at_3": 0.4444444444444444,
        "recall_at_3": 0.5584126984126985,
        "precision_at_5": 0.3466666666666667,
        "recall_at_5": 0.6668253968253969,
        "mrr": 0.5888888888888889,
    }
    assert len(retrieval["questions"]) == 15
    assert all("matches" not in question for question in retrieval["questions"])

    answer = body["answer_quality"]
    assert answer["live_llm_evaluation"]["status"] == "pending"
    assert answer["live_llm_evaluation"]["faithfulness"] is None
    assert answer["fixture_validation"]["result_kind"] == "test_fixture_only"
    assert answer["numerical_correctness"]["numerical_correctness"] == 1.0


def test_evaluation_report_maps_missing_files_to_safe_error(monkeypatch):
    def unavailable():
        raise EvaluationReportUnavailableError("private report path")

    monkeypatch.setattr(main_module, "load_evaluation_dashboard_report", unavailable)
    response = client.get("/api/evaluation/report")

    assert response.status_code == 503
    assert response.json() == {"detail": "Evaluation reports are temporarily unavailable."}
    assert "private" not in response.text
