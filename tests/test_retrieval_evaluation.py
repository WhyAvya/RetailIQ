from pathlib import Path

import pytest

from backend.rag_retrieval import DatasetNotIndexedError, RetrievalMatch
from evaluation.retrieval_evaluation import (
    DEFAULT_REPORT_PATH,
    calculate_question_metrics,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    save_report,
)


def _match(upload_id, rank, row):
    return RetrievalMatch(
        rank=rank,
        chunk_id=f"chunk-{row}",
        document=f"source row {row}",
        source_row=row,
        metadata={"upload_id": upload_id, "row_number": row},
        similarity_score=1.0 - rank / 10,
    )


def test_metrics_match_verified_known_ranking():
    retrieved = [2, 9, 4, 8, 7]
    relevant = [2, 4, 7]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(3 / 5)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)


def test_first_relevant_at_rank_three_has_one_third_reciprocal_rank():
    assert reciprocal_rank([8, 9, 4, 2], [2, 4]) == pytest.approx(1 / 3)


def test_fewer_than_k_results_use_fixed_k_precision_denominator():
    assert precision_at_k([2], [2, 4], 3) == pytest.approx(1 / 3)
    assert recall_at_k([2], [2, 4], 3) == pytest.approx(1 / 2)


def test_empty_results_and_missing_relevant_rows_return_zero():
    assert calculate_question_metrics([], [1], (3, 5)) == {
        "precision_at_3": 0.0,
        "recall_at_3": 0.0,
        "precision_at_5": 0.0,
        "recall_at_5": 0.0,
        "reciprocal_rank": 0.0,
    }
    assert precision_at_k([1, 2], [], 3) == 0.0
    assert recall_at_k([1, 2], [], 3) == 0.0
    assert reciprocal_rank([1, 2], []) == 0.0


def test_duplicate_source_rows_do_not_inflate_metrics():
    assert precision_at_k([2, 2, 9], [2], 3) == pytest.approx(1 / 3)
    assert recall_at_k([2, 2, 9], [2], 3) == 1.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k([1], [1], k)


def test_evaluator_uses_only_semantic_questions_and_expected_upload_id():
    calls = []

    def retriever(upload_id, question, top_k):
        calls.append((upload_id, question, top_k))
        return [_match(upload_id, rank, row) for rank, row in enumerate([1, 7, 10, 3, 4], start=1)]

    report = evaluate_retrieval(k_values=(3, 5), retriever=retriever)

    assert report["status"] == "completed"
    assert report["benchmark"]["semantic_questions_evaluated"] == 15
    assert report["benchmark"]["numerical_questions_excluded"] == 5
    assert len(calls) == len(report["questions"]) == 15
    assert {call[0] for call in calls} == {"6377aeff-8a64-47ac-9748-28e0b0523d56"}
    assert {call[2] for call in calls} == {5}
    assert all(item["id"].startswith("SEM-") for item in report["questions"])


def test_evaluator_records_retrieval_failures_without_fabricating_rows():
    def unavailable(_upload_id, _question, _top_k):
        raise DatasetNotIndexedError("The selected dataset has not been indexed.")

    report = evaluate_retrieval(retriever=unavailable)

    assert report["status"] == "completed_with_failures"
    assert report["failure_count"] == 15
    assert all(item["retrieved_source_rows"] == [] for item in report["questions"])
    assert all(item["matches"] == [] for item in report["questions"])
    assert all(item["failure"]["type"] == "DatasetNotIndexedError" for item in report["questions"])
    assert all(value == 0.0 for value in report["aggregate_metrics"].values())


def test_evaluator_rejects_cross_dataset_matches():
    def mixed(_upload_id, _question, _top_k):
        return [_match("another-upload", 1, 1)]

    report = evaluate_retrieval(retriever=mixed)

    assert report["failure_count"] == 15
    assert all(item["retrieved_source_rows"] == [] for item in report["questions"])


def test_report_is_saved_as_json(tmp_path):
    report = {"aggregate_metrics": {"mrr": 1.0}}
    output = save_report(report, tmp_path / "report.json")

    assert output == Path(tmp_path / "report.json")
    assert '"mrr": 1.0' in output.read_text(encoding="utf-8")


def test_real_report_contains_all_semantic_questions_and_consistent_aggregates():
    import json

    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    questions = report["questions"]

    assert report["status"] == "completed"
    assert report["failure_count"] == 0
    assert report["retrieval"]["mrr_cutoff"] == 5
    assert len(questions) == report["benchmark"]["semantic_questions_evaluated"] == 15
    assert all(item["id"].startswith("SEM-") for item in questions)
    for metric in ("precision_at_3", "precision_at_5", "recall_at_3", "recall_at_5"):
        assert report["aggregate_metrics"][metric] == pytest.approx(
            sum(item["metrics"][metric] for item in questions) / len(questions)
        )
    assert report["aggregate_metrics"]["mrr"] == pytest.approx(
        sum(item["metrics"]["reciprocal_rank"] for item in questions) / len(questions)
    )
