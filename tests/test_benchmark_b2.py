import json
from pathlib import Path

import pytest

from backend.rag_retrieval import RetrievalMatch
from evaluation.benchmark_b1 import DEFAULT_BENCHMARK_PATH
from evaluation.benchmark_b2 import (
    DEFAULT_REPORT_PATH,
    evaluate_b2,
    record_set_metrics,
    render_markdown,
    save_report,
    values_equal,
)


def _match(upload_id, rank, row):
    return RetrievalMatch(
        rank=rank,
        chunk_id=f"chunk-{row}",
        document=f"source row {row}",
        source_row=row,
        metadata={"upload_id": upload_id, "row_number": row},
        similarity_score=1.0 - rank / 100,
    )


def test_record_set_metrics_score_exact_partial_and_empty_sets():
    assert record_set_metrics([1, 2], [2, 1]) == {
        "precision": 1.0,
        "recall": 1.0,
        "exact_match": True,
        "true_positive_count": 2,
        "false_positive_count": 0,
        "false_negative_count": 0,
    }
    partial = record_set_metrics([1, 3], [1, 2])
    assert partial["precision"] == pytest.approx(0.5)
    assert partial["recall"] == pytest.approx(0.5)
    assert partial["exact_match"] is False
    assert record_set_metrics([], [1])["recall"] == 0.0
    assert record_set_metrics([], [])["exact_match"] is True


def test_deterministic_value_comparison_handles_nested_numeric_results():
    assert values_equal(10.0000001, 10.0)
    assert values_equal({"West": 1.25}, {"West": 1.2500001})
    assert values_equal([1, {"profit": 2.0}], [1, {"profit": 2.0000001}])
    assert not values_equal({"West": 1.0}, {"East": 1.0})
    assert not values_equal(11, 10)


def test_b2_uses_only_development_questions_and_routes_metric_types():
    benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    relevant_by_question = {
        entry["question"]: entry["relevant_source_rows"]
        for entry in benchmark["questions"]
        if entry["split"] == "development"
        and entry["type"] in {"semantic_product_discovery", "mixed"}
    }
    calls = []

    def retriever(upload_id, question, top_k):
        calls.append((upload_id, question, top_k))
        rows = relevant_by_question[question][:top_k]
        return [_match(upload_id, rank, row) for rank, row in enumerate(rows, 1)]

    report = evaluate_b2(retriever=retriever)

    assert report["scope"]["development_questions_evaluated"] == 40
    assert report["scope"]["held_out_questions_excluded"] == 20
    assert report["scope"]["held_out_question_ids_evaluated"] == []
    assert len(report["scope"]["evaluated_question_ids"]) == 40
    assert all("held_out" not in item for item in report["scope"]["evaluated_question_ids"])
    assert len(report["questions"]) == 40
    assert len(calls) == 20
    assert {call[2] for call in calls} == {5}
    assert report["system_under_test"]["llm_calls"] == 0
    assert report["system_under_test"]["chroma_write_operations"] == 0

    metrics = report["metrics"]
    assert metrics["semantic_product_discovery"]["questions_evaluated"] == 13
    assert metrics["semantic_product_discovery"]["precision_at_3"] == 1.0
    assert metrics["semantic_product_discovery"]["precision_at_5"] == 1.0
    assert metrics["semantic_product_discovery"]["recall_at_3"] == pytest.approx(0.6)
    assert metrics["semantic_product_discovery"]["recall_at_5"] == 1.0
    assert metrics["semantic_product_discovery"]["reciprocal_rank_at_5"] == 1.0
    assert metrics["semantic_product_discovery"]["questions_with_full_recall_at_5"] == 13
    assert metrics["metadata_filtering"]["questions_evaluated"] == 10
    assert metrics["metadata_filtering"]["record_set_exact_accuracy"] == pytest.approx(0.6)
    assert metrics["metadata_filtering"]["supported_only_exact_accuracy"] == 1.0
    assert metrics["numerical"]["questions_evaluated"] == 10
    assert metrics["numerical"]["deterministic_answer_accuracy"] == pytest.approx(0.3)
    assert metrics["numerical"]["supported_questions"] == 3
    assert metrics["numerical"]["supported_only_accuracy"] == 1.0
    assert metrics["mixed"]["questions_evaluated"] == 7
    assert metrics["mixed"]["deterministic_answer_accuracy"] == 0.0
    assert report["failure_analysis"]["question_incidence_counts"][
        "unsupported_or_incomplete_calculation"
    ] == 14


def test_report_generation_contains_metrics_failures_and_boundaries(tmp_path):
    benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    relevant_by_question = {
        entry["question"]: entry["relevant_source_rows"]
        for entry in benchmark["questions"]
        if entry["split"] == "development"
        and entry["type"] in {"semantic_product_discovery", "mixed"}
    }

    def retriever(upload_id, question, top_k):
        rows = relevant_by_question[question][:top_k]
        return [_match(upload_id, rank, row) for rank, row in enumerate(rows, 1)]

    report = evaluate_b2(retriever=retriever)
    json_path, markdown_path = save_report(
        report,
        tmp_path / "b2.json",
        tmp_path / "b2.md",
    )

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["metrics"] == report["metrics"]
    assert saved["failure_analysis"] == report["failure_analysis"]
    assert saved["recommendations"]
    assert "Semantic retrieval" in markdown
    assert "Failure types" in markdown
    assert "OpenRouter was not called" in markdown
    assert "No LLM-generated answer" in markdown
    assert render_markdown(report) == markdown
    assert json_path == Path(tmp_path / "b2.json")


def test_saved_real_report_is_development_only_and_aggregates_per_question_metrics():
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    development_ids = {
        entry["id"] for entry in benchmark["questions"] if entry["split"] == "development"
    }
    held_out_ids = {
        entry["id"] for entry in benchmark["questions"] if entry["split"] == "held_out_test"
    }

    result_ids = {entry["id"] for entry in report["questions"]}
    retrieval_results = [
        entry for entry in report["questions"] if "retrieval_metrics" in entry
    ]
    aggregate = report["metrics"]["overall_retrieval"]

    assert result_ids == development_ids
    assert result_ids.isdisjoint(held_out_ids)
    assert len(report["questions"]) == 40
    assert report["scope"]["held_out_question_ids_evaluated"] == []
    assert report["system_under_test"]["llm_calls"] == 0
    assert report["system_under_test"]["chroma_write_operations"] == 0
    assert aggregate["questions_evaluated"] == len(retrieval_results) == 20
    for metric in (
        "precision_at_3",
        "precision_at_5",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank_at_5",
    ):
        assert aggregate[metric] == pytest.approx(
            sum(entry["retrieval_metrics"][metric] for entry in retrieval_results)
            / len(retrieval_results)
        )
