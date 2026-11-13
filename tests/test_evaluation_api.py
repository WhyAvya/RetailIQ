from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from evaluation.report_service import EvaluationReportUnavailableError


client = TestClient(app)


def test_evaluation_report_returns_all_verified_benchmark_sections():
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

    optimized = body["optimized_c8_retrieval"]
    assert optimized["semantic_questions_evaluated"] == 5
    assert optimized["metrics"]["precision_at_5"] == {
        "baseline": 0.24,
        "optimized": 1.0,
        "delta": 0.76,
    }
    assert optimized["metrics"]["recall_at_5"]["optimized"] == 0.18160321052985787
    assert optimized["citations_valid"] is True

    structured = body["structured_evaluation"]
    assert structured["questions_evaluated"] == 20
    assert structured["question_count_by_category"] == {
        "numerical": 5,
        "filtering": 5,
        "ranking": 5,
        "mixed": 5,
    }
    assert structured["metrics"] == {
        "numerical": 1.0,
        "filtering": 1.0,
        "ranking": 1.0,
        "mixed": 1.0,
    }

    live_pipeline = body["phase5_live"]
    assert live_pipeline["queries_completed"] == 5
    assert live_pipeline["correct_queries"] == 5
    assert live_pipeline["llm_invocation_count"] == 1
    assert live_pipeline["citation_checks"] == 1
    assert live_pipeline["valid_citation_checks"] == 1
    assert len(live_pipeline["queries"]) == 5
    assert sum(item["llm_invoked"] is True for item in live_pipeline["queries"]) == 1

    answer = body["answer_quality"]
    assert answer["verified_live_measurement"]["answers_evaluated"] == 3
    assert answer["verified_live_measurement"]["faithfulness"] == 0.75
    assert answer["verified_live_measurement"]["answer_relevance"] == 0.8888888888888888
    assert answer["verified_live_measurement"]["unsupported_claim_rate"] == 0.25
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
