"""C5/C6 evaluation for the integrated hybrid RAG pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from backend.chroma_metadata import FULL_DATASET_UPLOAD_ID, load_metadata_source
from backend.dataset_selection import SelectedDataset
from backend.metadata_filtered_retrieval import retrieve_metadata_filtered_matches
from backend.query_planner import plan_query
from backend.rag_pipeline import answer_question
from backend.rag_retrieval import RetrievalMatch, retrieve_semantic_matches
from backend.structured_analytics import apply_plan_filters
from evaluation.benchmark_b1 import DEFAULT_BENCHMARK_PATH
from evaluation.benchmark_b2 import DEFAULT_REPORT_PATH as B2_REPORT_PATH
from evaluation.benchmark_b2 import values_equal
from evaluation.retrieval_evaluation import calculate_question_metrics


REPORT_ROOT = Path(__file__).resolve().parent / "reports"
DEFAULT_DEVELOPMENT_REPORT_PATH = REPORT_ROOT / "hybrid_c5_development.json"
DEFAULT_DEVELOPMENT_MARKDOWN_PATH = REPORT_ROOT / "hybrid_c5_development.md"
DEFAULT_HELD_OUT_REPORT_PATH = REPORT_ROOT / "hybrid_c6_held_out.json"
DEFAULT_HELD_OUT_MARKDOWN_PATH = REPORT_ROOT / "hybrid_c6_held_out.md"
Retriever = Callable[[str, str, int], list[RetrievalMatch]]
FilteredRetriever = Callable[[str, Any, int], list[RetrievalMatch]]


class NoEvaluationLLM:
    def generate_grounded_answer(self, *_args: Any) -> str:
        raise AssertionError("C5/C6 deterministic evaluation must not call an LLM.")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _source_rows(records: list[dict[str, Any]]) -> list[int]:
    rows: list[int] = []
    for record in records:
        if "source_row" in record:
            rows.append(int(record["source_row"]))
        elif record.get("result_kind") == "row_ranking" and "source_row" in record:
            rows.append(int(record["source_row"]))
    return rows


def _mixed_correct(entry: dict[str, Any], calculation: dict[str, Any]) -> bool:
    expected = {
        key: value
        for key, value in entry["ground_truth"].items()
        if key not in {"verification", "matching_row_count"}
    }
    records = calculation.get("records", [])
    value = calculation.get("value")
    row_records = [
        record for record in records if record.get("result_kind") != "group_ranking"
    ]
    group_records = [
        record for record in records if record.get("result_kind") == "group_ranking"
    ]
    if "top_record" in expected:
        return bool(
            row_records
            and row_records[0].get("source_row") == expected["top_record"]["source_row"]
            and values_equal(value, expected.get("total_sales"))
        )
    if "top_records" in expected:
        return (
            [record.get("source_row") for record in row_records]
            == [record["source_row"] for record in expected["top_records"]]
            and values_equal(value, expected.get("total_profit"))
        )
    if "sub_category" in expected:
        return bool(
            records
            and records[0].get("Sub-Category") == expected["sub_category"]
            and values_equal(value, expected["sales"])
        )
    if "customer_id" in expected:
        return bool(
            records
            and records[0].get("Customer ID") == expected["customer_id"]
            and values_equal(records[0].get("Sales"), expected["sales"])
            and records[0].get("distinct_order_count") == expected["distinct_orders"]
        )
    if "region" in expected and "top_source_row" in expected:
        return bool(
            group_records
            and group_records[0].get("Region") == expected["region"]
            and values_equal(group_records[0].get("Sales"), expected["region_sales"])
            and row_records
            and row_records[0].get("source_row") == expected["top_source_row"]
            and values_equal(row_records[0].get("Sales"), expected["top_sales"])
        )
    if "top_source_row" in expected:
        expected_value = next(
            (
                expected[key]
                for key in ("average_profit", "count", "total_sales", "profit_margin_percent")
                if key in expected
            ),
            None,
        )
        return bool(
            row_records
            and row_records[0].get("source_row") == expected["top_source_row"]
            and values_equal(value, expected_value)
        )
    return False


def _numerical_value(entry: dict[str, Any], calculation: dict[str, Any] | None) -> Any:
    if not calculation:
        return None
    expected = entry["ground_truth"]["value"]
    records = calculation.get("records", [])
    if not isinstance(expected, dict):
        return calculation.get("value")
    if "state" in expected:
        return (
            {"state": records[0].get("State"), "sales": records[0].get("Sales")}
            if records else None
        )
    if "sub_category" in expected:
        return (
            {
                "sub_category": records[0].get("Sub-Category"),
                "profit": records[0].get("Profit"),
            }
            if records else None
        )
    if not records:
        return None
    dimension = next(
        (key for key in ("Region", "Category", "Sub-Category", "order_year") if key in records[0]),
        None,
    )
    if dimension is None:
        return None
    return {str(record[dimension]): record.get("value") for record in records}


def _raw_retrieval_result(
    entry: dict[str, Any], retriever: Retriever
) -> dict[str, Any]:
    matches = retriever(FULL_DATASET_UPLOAD_ID, entry["question"], 5)
    rows = [match.source_row for match in matches if match.source_row is not None]
    metrics = calculate_question_metrics(rows, entry["relevant_source_rows"], (3, 5))
    return {
        "id": entry["id"],
        "retrieved_source_rows": rows,
        "metrics": {
            "precision_at_3": metrics["precision_at_3"],
            "precision_at_5": metrics["precision_at_5"],
            "recall_at_3": metrics["recall_at_3"],
            "recall_at_5": metrics["recall_at_5"],
            "reciprocal_rank_at_5": metrics["reciprocal_rank"],
        },
    }


def _aggregate_raw(results: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "precision_at_3",
        "precision_at_5",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank_at_5",
    )
    return {
        "questions_evaluated": len(results),
        **{
            name: _mean([result["metrics"][name] for result in results])
            for name in names
        },
    }


def evaluate_hybrid(
    split: str,
    benchmark_path: Path | str = DEFAULT_BENCHMARK_PATH,
    retriever: Retriever = retrieve_semantic_matches,
    filtered_retriever: FilteredRetriever = retrieve_metadata_filtered_matches,
    include_raw_retrieval: bool = True,
) -> dict[str, Any]:
    if split not in {"development", "held_out_test"}:
        raise ValueError("split must be development or held_out_test")
    benchmark_path = Path(benchmark_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = [entry for entry in benchmark["questions"] if entry["split"] == split]
    expected_count = 40 if split == "development" else 20
    if len(questions) != expected_count:
        raise ValueError(f"expected {expected_count} {split} questions")
    frame = load_metadata_source()
    dataset = SelectedDataset(UUID(FULL_DATASET_UPLOAD_ID), "sample_superstore.csv", frame)
    dimensions = {
        "region": frame["Region"].unique().tolist(),
        "state": frame["State"].unique().tolist(),
        "city": frame["City"].unique().tolist(),
        "customer_name": frame["Customer Name"].unique().tolist(),
    }
    results: list[dict[str, Any]] = []
    for entry in questions:
        plan = plan_query(entry["question"], dimensions)
        answer = answer_question(
            dataset.upload_id,
            entry["question"],
            metadata_lookup=lambda _id: {"upload_id": str(_id)},
            dataset_loader=lambda _id: dataset,
            retriever=retriever,
            filtered_retriever=filtered_retriever,
            llm_service=NoEvaluationLLM(),
        )
        result: dict[str, Any] = {
            "id": entry["id"],
            "type": entry["type"],
            "question": entry["question"],
            "plan": plan.as_dict(),
            "mode": answer.mode,
            "answer": answer.answer,
            "calculation": answer.calculation,
            "source_rows": [source.source_row for source in answer.sources],
        }
        if entry["type"] == "metadata_filtering":
            filtered = apply_plan_filters(frame, plan)
            actual_rows = {int(row) for row in filtered.index}
            expected_rows = set(entry["relevant_source_rows"])
            result["record_set"] = {
                "actual_count": len(actual_rows),
                "expected_count": len(expected_rows),
                "false_positive_count": len(actual_rows - expected_rows),
                "false_negative_count": len(expected_rows - actual_rows),
                "exact_match": actual_rows == expected_rows,
            }
        elif entry["type"] == "numerical":
            actual = _numerical_value(entry, answer.calculation)
            expected = entry["ground_truth"]["value"]
            result["deterministic"] = {
                "actual_value": actual,
                "expected_value": expected,
                "correct": values_equal(actual, expected),
            }
        elif entry["type"] == "semantic_product_discovery":
            records = answer.calculation.get("records", []) if answer.calculation else []
            actual_rows = _source_rows(records)
            expected_rows = entry["relevant_source_rows"]
            result["ranking"] = {
                "actual_source_rows": actual_rows,
                "expected_source_rows": expected_rows,
                "exact_order": actual_rows == expected_rows,
                "exact_set": set(actual_rows) == set(expected_rows),
            }
        else:
            result["mixed"] = {
                "correct": bool(answer.calculation and _mixed_correct(entry, answer.calculation))
            }
        results.append(result)

    metadata = [item for item in results if item["type"] == "metadata_filtering"]
    numerical = [item for item in results if item["type"] == "numerical"]
    rankings = [item for item in results if item["type"] == "semantic_product_discovery"]
    mixed = [item for item in results if item["type"] == "mixed"]
    raw_results: list[dict[str, Any]] = []
    if include_raw_retrieval:
        raw_results = [
            _raw_retrieval_result(entry, retriever)
            for entry in questions
            if entry["type"] in {"semantic_product_discovery", "mixed"}
        ]
    metrics = {
        "exact_filtering_accuracy": _mean(
            [float(item["record_set"]["exact_match"]) for item in metadata]
        ),
        "numerical_accuracy": _mean(
            [float(item["deterministic"]["correct"]) for item in numerical]
        ),
        "ranking_exact_order_accuracy": _mean(
            [float(item["ranking"]["exact_order"]) for item in rankings]
        ),
        "ranking_exact_set_accuracy": _mean(
            [float(item["ranking"]["exact_set"]) for item in rankings]
        ),
        "mixed_exact_accuracy": _mean(
            [float(item["mixed"]["correct"]) for item in mixed]
        ),
        "raw_semantic_retrieval": _aggregate_raw(raw_results),
    }
    report: dict[str, Any] = {
        "evaluation_id": f"retailiq-hybrid-{'c5-development' if split == 'development' else 'c6-held-out'}-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "question_count": len(questions),
        "question_ids": [entry["id"] for entry in questions],
        "llm_calls": 0,
        "metrics": metrics,
        "questions": results,
        "raw_retrieval_questions": raw_results,
    }
    if split == "development":
        baseline = json.loads(B2_REPORT_PATH.read_text(encoding="utf-8"))
        baseline_raw = baseline["metrics"]["overall_retrieval"]
        current_raw = metrics["raw_semantic_retrieval"]
        serious_regressions = []
        for name in ("precision_at_5", "recall_at_5", "reciprocal_rank_at_5"):
            if current_raw[name] + 1e-12 < baseline_raw[name]:
                serious_regressions.append(
                    f"raw {name} fell from {baseline_raw[name]} to {current_raw[name]}"
                )
        if metrics["exact_filtering_accuracy"] < baseline["metrics"]["metadata_filtering"]["record_set_exact_accuracy"]:
            serious_regressions.append("exact filtering accuracy regressed")
        if metrics["numerical_accuracy"] < baseline["metrics"]["numerical"]["deterministic_answer_accuracy"]:
            serious_regressions.append("numerical accuracy regressed")
        report["baseline_comparison"] = {
            "b2_report": "evaluation/reports/benchmark_b2_baseline.json",
            "before": baseline["metrics"],
            "after": metrics,
            "serious_regressions": serious_regressions,
            "stable_for_c6": not serious_regressions,
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    raw = metrics["raw_semantic_retrieval"]
    lines = [
        f"# RetailIQ Hybrid {'C5 Development' if report['split'] == 'development' else 'C6 Held-Out'} Evaluation",
        "",
        f"Questions evaluated: {report['question_count']}. LLM calls: {report['llm_calls']}.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Exact filtering accuracy | {metrics['exact_filtering_accuracy']:.4f} |",
        f"| Numerical accuracy | {metrics['numerical_accuracy']:.4f} |",
        f"| Ranking exact-order accuracy | {metrics['ranking_exact_order_accuracy']:.4f} |",
        f"| Ranking exact-set accuracy | {metrics['ranking_exact_set_accuracy']:.4f} |",
        f"| Mixed exact accuracy | {metrics['mixed_exact_accuracy']:.4f} |",
        f"| Raw retrieval P@3 | {raw['precision_at_3']:.4f} |",
        f"| Raw retrieval P@5 | {raw['precision_at_5']:.4f} |",
        f"| Raw retrieval R@3 | {raw['recall_at_3']:.4f} |",
        f"| Raw retrieval R@5 | {raw['recall_at_5']:.4f} |",
        f"| Raw retrieval MRR@5 | {raw['reciprocal_rank_at_5']:.4f} |",
        "",
        "Deterministic metrics score the integrated planner and Pandas executor. Raw retrieval metrics are reported separately and do not include deterministic ranking gains.",
        "",
    ]
    if "baseline_comparison" in report:
        comparison = report["baseline_comparison"]
        lines.extend(
            [
                "## C6 gate",
                "",
                f"Stable for C6: {comparison['stable_for_c6']}.",
                f"Serious regressions: {comparison['serious_regressions'] or 'none'}.",
                "",
            ]
        )
    failures = [
        item["id"]
        for item in report["questions"]
        if not (
            item.get("record_set", {}).get("exact_match", True)
            and item.get("deterministic", {}).get("correct", True)
            and item.get("ranking", {}).get("exact_order", True)
            and item.get("mixed", {}).get("correct", True)
        )
    ]
    lines.extend(["## Remaining failures", "", ", ".join(failures) if failures else "None.", ""])
    return "\n".join(lines)


def save_report(
    report: dict[str, Any], json_path: Path | str, markdown_path: Path | str
) -> tuple[Path, Path]:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate integrated RetailIQ hybrid RAG.")
    parser.add_argument("--split", choices=("development", "held_out_test"), required=True)
    arguments = parser.parse_args()
    report = evaluate_hybrid(arguments.split)
    paths = (
        (DEFAULT_DEVELOPMENT_REPORT_PATH, DEFAULT_DEVELOPMENT_MARKDOWN_PATH)
        if arguments.split == "development"
        else (DEFAULT_HELD_OUT_REPORT_PATH, DEFAULT_HELD_OUT_MARKDOWN_PATH)
    )
    save_report(report, *paths)
    print(json.dumps(report["metrics"], indent=2))
    if "baseline_comparison" in report:
        print(json.dumps(report["baseline_comparison"], indent=2))


if __name__ == "__main__":
    main()
