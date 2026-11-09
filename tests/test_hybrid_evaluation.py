import json

from backend.rag_retrieval import RetrievalMatch
from evaluation.benchmark_b1 import DEFAULT_BENCHMARK_PATH
from evaluation.hybrid_evaluation import (
    DEFAULT_DEVELOPMENT_REPORT_PATH,
    DEFAULT_HELD_OUT_REPORT_PATH,
    evaluate_hybrid,
    render_markdown,
    save_report,
)


def test_development_evaluator_excludes_held_out_and_generates_report(tmp_path):
    benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    relevant = {
        entry["question"]: entry["relevant_source_rows"]
        for entry in benchmark["questions"]
        if entry["split"] == "development"
    }

    def retriever(upload_id, question, top_k):
        return [
            RetrievalMatch(
                rank,
                f"chunk-{row}",
                f"row {row}",
                row,
                {"upload_id": upload_id, "row_number": row},
                1.0 - rank / 100,
            )
            for rank, row in enumerate(relevant[question][:top_k], 1)
        ]

    report = evaluate_hybrid(
        "development",
        retriever=retriever,
        filtered_retriever=lambda *_args: [],
    )

    held_out_ids = {
        entry["id"] for entry in benchmark["questions"] if entry["split"] == "held_out_test"
    }
    assert report["question_count"] == 40
    assert set(report["question_ids"]).isdisjoint(held_out_ids)
    assert report["llm_calls"] == 0
    assert report["metrics"]["exact_filtering_accuracy"] == 1.0
    assert report["metrics"]["numerical_accuracy"] == 1.0
    assert report["baseline_comparison"]["stable_for_c6"] is True

    json_path, markdown_path = save_report(
        report, tmp_path / "report.json", tmp_path / "report.md"
    )
    assert json.loads(json_path.read_text())["metrics"] == report["metrics"]
    assert render_markdown(report) == markdown_path.read_text()


def test_saved_development_and_held_out_reports_are_disjoint_and_frozen():
    development = json.loads(DEFAULT_DEVELOPMENT_REPORT_PATH.read_text(encoding="utf-8"))
    held_out = json.loads(DEFAULT_HELD_OUT_REPORT_PATH.read_text(encoding="utf-8"))

    assert development["split"] == "development"
    assert held_out["split"] == "held_out_test"
    assert development["question_count"] == 40
    assert held_out["question_count"] == 20
    assert set(development["question_ids"]).isdisjoint(held_out["question_ids"])
    assert development["baseline_comparison"]["stable_for_c6"] is True
    assert development["baseline_comparison"]["serious_regressions"] == []
    assert development["metrics"]["exact_filtering_accuracy"] == 1.0
    assert development["metrics"]["numerical_accuracy"] == 1.0
    assert development["metrics"]["mixed_exact_accuracy"] == 1.0
    assert "baseline_comparison" not in held_out
