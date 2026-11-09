"""Benchmark B2 baseline evaluation over B1 development questions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from backend.analytics import calculate_kpis
from backend.analytics_filters import DashboardFilters, apply_dashboard_filters
from backend.chart_analytics import calculate_sales_by_region
from backend.embeddings import EMBEDDING_MODEL_NAME
from backend.rag_pipeline import route_question
from backend.rag_retrieval import RetrievalError, RetrievalMatch, retrieve_semantic_matches
from backend.vector_store import COLLECTION_NAME
from evaluation.benchmark_b1 import (
    DEFAULT_BENCHMARK_PATH,
    load_source_dataframe,
)
from evaluation.retrieval_evaluation import calculate_question_metrics


DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent / "reports" / "benchmark_b2_baseline.json"
)
DEFAULT_MARKDOWN_PATH = (
    Path(__file__).resolve().parent / "reports" / "benchmark_b2_baseline.md"
)
Retriever = Callable[[str, str, int], list[RetrievalMatch]]
K_VALUES = (3, 5)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _unique_rows(matches: Sequence[RetrievalMatch]) -> list[int]:
    rows: list[int] = []
    seen: set[int] = set()
    for match in matches:
        row = match.source_row
        if isinstance(row, int) and not isinstance(row, bool) and row not in seen:
            rows.append(row)
            seen.add(row)
    return rows


def record_set_metrics(
    actual_rows: Sequence[int], expected_rows: Sequence[int]
) -> dict[str, Any]:
    """Score an unordered exact-filter result without retrieval cutoffs."""
    actual = set(actual_rows)
    expected = set(expected_rows)
    true_positive = len(actual & expected)
    precision = true_positive / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = true_positive / len(expected) if expected else (1.0 if not actual else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "exact_match": actual == expected,
        "true_positive_count": true_positive,
        "false_positive_count": len(actual - expected),
        "false_negative_count": len(expected - actual),
    }


def values_equal(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """Compare deterministic scalar or nested results with numeric tolerance."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance * max(
            1.0, abs(float(expected))
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            values_equal(actual[key], expected[key], tolerance) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            values_equal(left, right, tolerance)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _retrieval_metrics(
    matches: Sequence[RetrievalMatch], relevant_rows: Sequence[int]
) -> tuple[list[int], dict[str, float]]:
    retrieved_rows = _unique_rows(matches)[:5]
    raw = calculate_question_metrics(retrieved_rows, relevant_rows, K_VALUES)
    return retrieved_rows, {
        "precision_at_3": raw["precision_at_3"],
        "precision_at_5": raw["precision_at_5"],
        "recall_at_3": raw["recall_at_3"],
        "recall_at_5": raw["recall_at_5"],
        "reciprocal_rank_at_5": raw["reciprocal_rank"],
    }


def _match_payload(match: RetrievalMatch) -> dict[str, Any]:
    return {
        "rank": match.rank,
        "source_row": match.source_row,
        "similarity_score": match.similarity_score,
        "chunk_id": match.chunk_id,
        "metadata": match.metadata,
    }


def _aggregate_retrieval(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "precision_at_3",
        "precision_at_5",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank_at_5",
    )
    return {
        "questions_evaluated": len(results),
        **{
            metric: _mean([result["retrieval_metrics"][metric] for result in results])
            for metric in metric_names
        },
        "retrieval_failure_count": sum(
            result.get("retrieval_failure") is not None for result in results
        ),
        "questions_with_at_least_one_relevant_at_5": sum(
            result["retrieval_metrics"]["reciprocal_rank_at_5"] > 0
            for result in results
        ),
        "questions_with_full_recall_at_5": sum(
            result["retrieval_metrics"]["recall_at_5"] == 1.0
            for result in results
        ),
    }


def _evaluate_retrieval_question(
    entry: dict[str, Any], upload_id: str, retriever: Retriever
) -> dict[str, Any]:
    matches: list[RetrievalMatch] = []
    failure = None
    try:
        matches = retriever(upload_id, entry["question"], 5)
        if any(match.metadata.get("upload_id") != upload_id for match in matches):
            raise RetrievalError("Retrieval returned rows from another dataset.")
    except RetrievalError as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    retrieved_rows, metrics = _retrieval_metrics(matches, entry["relevant_source_rows"])
    return {
        "id": entry["id"],
        "type": entry["type"],
        "question": entry["question"],
        "evaluation_mode": "chromadb_retrieval",
        "relevant_source_rows": entry["relevant_source_rows"],
        "retrieved_source_rows": retrieved_rows,
        "matches": [_match_payload(match) for match in matches],
        "retrieval_metrics": metrics,
        "at_least_one_relevant_at_5": metrics["reciprocal_rank_at_5"] > 0,
        "full_recall_at_5": metrics["recall_at_5"] == 1.0,
        "retrieval_failure": failure,
    }


METADATA_FILTER_PLANS: dict[str, DashboardFilters | None] = {
    "B1-META-001": DashboardFilters(region="West"),
    "B1-META-002": DashboardFilters(region="East", category="Technology"),
    "B1-META-003": DashboardFilters(region="South", category="Furniture"),
    "B1-META-004": DashboardFilters(region="Central", category="Office Supplies"),
    "B1-META-005": None,
    "B1-META-006": None,
    "B1-META-007": DashboardFilters(start_date=date(2017, 1, 1), end_date=date(2017, 12, 31)),
    "B1-META-008": DashboardFilters(start_date=date(2016, 10, 1), end_date=date(2016, 12, 31)),
    "B1-META-009": None,
    "B1-META-010": None,
}
METADATA_UNSUPPORTED_REASONS = {
    "B1-META-005": "The current exact-filter API has no State filter.",
    "B1-META-006": "The current exact-filter API has no City filter.",
    "B1-META-009": "The current exact-filter API has no Segment or Ship Mode filter.",
    "B1-META-010": "The current exact-filter API has no Segment or Ship Mode filter.",
}


def _evaluate_metadata_question(
    entry: dict[str, Any], source: pd.DataFrame
) -> dict[str, Any]:
    filters = METADATA_FILTER_PLANS[entry["id"]]
    supported = filters is not None
    actual_rows = (
        [int(row) for row in apply_dashboard_filters(source, filters).dataframe.index]
        if supported
        else []
    )
    metrics = record_set_metrics(actual_rows, entry["relevant_source_rows"])
    return {
        "id": entry["id"],
        "type": entry["type"],
        "question": entry["question"],
        "evaluation_mode": "deterministic_exact_filter",
        "supported": supported,
        "filters": filters.as_dict() if filters else None,
        "unsupported_reason": METADATA_UNSUPPORTED_REASONS.get(entry["id"]),
        "expected_row_count": len(entry["relevant_source_rows"]),
        "actual_row_count": len(actual_rows) if supported else None,
        "record_set_metrics": metrics,
        "deterministic_success": supported and metrics["exact_match"],
    }


def _numerical_actual(intent: str, source: pd.DataFrame) -> Any:
    kpis = calculate_kpis(source)
    intent_to_metric = {
        "total_revenue": "total_revenue",
        "total_profit": "total_profit",
        "number_of_orders": "number_of_orders",
        "unique_customers": "unique_customers",
        "average_order_value": "average_order_value",
        "profit_margin": "profit_margin",
    }
    metric_name = intent_to_metric.get(intent)
    if metric_name is not None:
        metric = kpis[metric_name]
        return metric["value"] if metric["available"] else None
    if intent == "highest_revenue_region":
        result = calculate_sales_by_region(source)
        if result["available"] and result["data"]:
            return {
                "region": result["data"][0]["region"],
                "sales": result["data"][0]["revenue"],
            }
    return None


def _evaluate_numerical_question(
    entry: dict[str, Any], source: pd.DataFrame
) -> dict[str, Any]:
    intent = route_question(entry["question"])
    supported = intent not in {"unsupported_numerical", "semantic_rag"}
    actual = _numerical_actual(intent, source) if supported else None
    expected = entry["ground_truth"]["value"]
    accurate = supported and values_equal(actual, expected)
    return {
        "id": entry["id"],
        "type": entry["type"],
        "question": entry["question"],
        "evaluation_mode": "deterministic_analytics",
        "routed_intent": intent,
        "supported": supported,
        "expected_value": expected,
        "actual_value": actual,
        "deterministic_answer_accurate": accurate,
        "unsupported_reason": (
            "The current deterministic question router does not support this calculation."
            if not supported
            else None
        ),
    }


def _evaluate_mixed_structured(
    entry: dict[str, Any], source: pd.DataFrame
) -> dict[str, Any]:
    intent = route_question(entry["question"])
    supported = intent not in {"unsupported_numerical", "semantic_rag"}
    actual = _numerical_actual(intent, source) if supported else None
    # A mixed answer is correct only if the current deterministic path returns
    # the complete structured B1 result, not merely one coincidental component.
    accurate = supported and values_equal(actual, {
        key: value
        for key, value in entry["ground_truth"].items()
        if key not in {"verification", "matching_row_count"}
    })
    return {
        "evaluation_mode": "deterministic_mixed_analysis",
        "routed_intent": intent,
        "supported": supported,
        "actual_value": actual,
        "deterministic_answer_accurate": accurate,
        "unsupported_or_incomplete_reason": (
            None
            if accurate
            else "The current deterministic path does not produce the complete filtered mixed result."
        ),
    }


def _parse_expected_dimension(definition: str, name: str) -> str | None:
    prefix = r"(?<!-)" if name == "Category" else ""
    match = re.search(
        rf"{prefix}{re.escape(name)} == ([A-Za-z ]+?)(?:,| and|\)|$)",
        definition,
    )
    return match.group(1).strip() if match else None


def _retrieval_failure_types(result: dict[str, Any], entry: dict[str, Any]) -> set[str]:
    metrics = result["retrieval_metrics"]
    if result["retrieval_failure"]:
        return {"retrieval_error"}
    failures: set[str] = set()
    if metrics["recall_at_5"] < 1.0:
        failures.add("missing_relevant_rows")
    if metrics["precision_at_5"] < 1.0:
        failures.add("irrelevant_rows_in_top_5")
    definition = entry["relevance_definition"]
    for dimension, metadata_key, label in (
        ("Region", "region", "incorrect_region"),
        ("Category", "category", "incorrect_category"),
        ("Sub-Category", "sub_category", "incorrect_sub_category"),
    ):
        expected = _parse_expected_dimension(definition, dimension)
        if expected and any(
            match["source_row"] not in set(entry["relevant_source_rows"])
            and str(match["metadata"].get(metadata_key, "")).casefold()
            != expected.casefold()
            for match in result["matches"]
        ):
            failures.add(label)
    if "contains" in definition and metrics["precision_at_5"] < 1.0:
        failures.add("product_keyword_mismatch")
    if "sorting by" in definition and metrics["precision_at_5"] < 1.0:
        failures.add("numeric_ranking_mismatch")
    return failures


def _failure_analysis(
    results: Sequence[dict[str, Any]], entries_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for result in results:
        if "retrieval_metrics" in result:
            labels = _retrieval_failure_types(result, entries_by_id[result["id"]])
            if result["type"] == "mixed" and not result.get(
                "deterministic_answer_accurate", False
            ):
                labels.add("unsupported_or_incomplete_calculation")
        elif result["type"] == "metadata_filtering":
            if not result["supported"]:
                labels = {"unsupported_filter_dimension"}
            elif not result["record_set_metrics"]["exact_match"]:
                labels = {"record_set_mismatch"}
            else:
                labels = set()
        elif not result.get("deterministic_answer_accurate", False):
            labels = {"unsupported_or_incomplete_calculation"}
        else:
            labels = set()
        for label in labels:
            counts[label] += 1
            examples.setdefault(label, [])
            if len(examples[label]) < 3:
                examples[label].append(result["id"])
    return {
        "question_incidence_counts": dict(counts.most_common()),
        "example_question_ids": examples,
        "note": "Counts are question incidences and can overlap; one question may exhibit multiple failure types.",
    }


def evaluate_b2(
    benchmark_path: Path | str = DEFAULT_BENCHMARK_PATH,
    retriever: Retriever = retrieve_semantic_matches,
) -> dict[str, Any]:
    """Run the B2 baseline without executing held-out questions or an LLM."""
    benchmark_path = Path(benchmark_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    development = [
        entry for entry in benchmark["questions"] if entry["split"] == "development"
    ]
    if len(development) != 40:
        raise ValueError("Benchmark B2 requires exactly 40 development questions.")
    if any(entry["split"] != "development" for entry in development):
        raise ValueError("Held-out questions cannot be evaluated in B2.")

    source = load_source_dataframe().drop(columns=["_order_date", "_ship_date"])
    upload_id = benchmark["source"]["upload_id"]
    semantic_results = [
        _evaluate_retrieval_question(entry, upload_id, retriever)
        for entry in development
        if entry["type"] == "semantic_product_discovery"
    ]
    metadata_results = [
        _evaluate_metadata_question(entry, source)
        for entry in development
        if entry["type"] == "metadata_filtering"
    ]
    numerical_results = [
        _evaluate_numerical_question(entry, source)
        for entry in development
        if entry["type"] == "numerical"
    ]
    mixed_results = []
    for entry in development:
        if entry["type"] != "mixed":
            continue
        retrieval_result = _evaluate_retrieval_question(entry, upload_id, retriever)
        retrieval_result.update(_evaluate_mixed_structured(entry, source))
        mixed_results.append(retrieval_result)

    retrieval_results = semantic_results + mixed_results
    metadata_exact = sum(result["deterministic_success"] for result in metadata_results)
    metadata_supported = [result for result in metadata_results if result["supported"]]
    numerical_accurate = sum(
        result["deterministic_answer_accurate"] for result in numerical_results
    )
    numerical_supported = [result for result in numerical_results if result["supported"]]
    mixed_accurate = sum(
        result["deterministic_answer_accurate"] for result in mixed_results
    )
    entries_by_id = {entry["id"]: entry for entry in development}
    all_results = semantic_results + metadata_results + numerical_results + mixed_results

    return {
        "evaluation_id": "retailiq-benchmark-b2-baseline-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "scope": {
            "benchmark": "RetailIQ Benchmark B1",
            "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            "development_questions_evaluated": len(development),
            "held_out_questions_excluded": sum(
                entry["split"] == "held_out_test" for entry in benchmark["questions"]
            ),
            "evaluated_question_ids": [entry["id"] for entry in development],
            "held_out_question_ids_evaluated": [],
            "upload_id": upload_id,
            "source_row_count": benchmark["source"]["row_count"],
        },
        "system_under_test": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "vector_collection": COLLECTION_NAME,
            "retrieval_cutoffs": list(K_VALUES),
            "llm_calls": 0,
            "llm_answer_quality_status": "not_evaluated_in_b2",
            "chroma_write_operations": 0,
        },
        "metrics": {
            "overall_retrieval": _aggregate_retrieval(retrieval_results),
            "semantic_product_discovery": _aggregate_retrieval(semantic_results),
            "metadata_filtering": {
                "questions_evaluated": len(metadata_results),
                "record_set_exact_accuracy": metadata_exact / len(metadata_results),
                "supported_questions": len(metadata_supported),
                "supported_only_exact_accuracy": (
                    sum(result["deterministic_success"] for result in metadata_supported)
                    / len(metadata_supported)
                    if metadata_supported
                    else 0.0
                ),
            },
            "numerical": {
                "questions_evaluated": len(numerical_results),
                "deterministic_answer_accuracy": numerical_accurate / len(numerical_results),
                "supported_questions": len(numerical_supported),
                "supported_only_accuracy": (
                    sum(
                        result["deterministic_answer_accurate"]
                        for result in numerical_supported
                    )
                    / len(numerical_supported)
                    if numerical_supported
                    else 0.0
                ),
            },
            "mixed": {
                **_aggregate_retrieval(mixed_results),
                "deterministic_answer_accuracy": mixed_accurate / len(mixed_results),
            },
        },
        "failure_analysis": _failure_analysis(all_results, entries_by_id),
        "recommendations": [
            "Improve semantic ranking for exact top-k numeric comparisons while preserving the current benchmark and held-out split.",
            "Add explicit metadata filtering support for State, City, Segment, and Ship Mode before evaluating those questions again.",
            "Expand deterministic analytics intents for averages, medians, quantities, filtered counts, and mixed retrieval-plus-aggregation questions.",
            "Evaluate LLM-generated answer quality separately; B2 measures no generated answers.",
        ],
        "questions": all_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    retrieval = metrics["overall_retrieval"]
    semantic = metrics["semantic_product_discovery"]
    mixed = metrics["mixed"]
    metadata = metrics["metadata_filtering"]
    numerical = metrics["numerical"]
    failures = report["failure_analysis"]["question_incidence_counts"]
    lines = [
        "# RetailIQ Benchmark B2 Baseline",
        "",
        "B2 evaluated only the 40 B1 development questions. The 20 held-out questions were not executed. ChromaDB was read only and OpenRouter was not called.",
        "",
        "## Baseline metrics",
        "",
        "| Evaluation | Questions | P@3 | P@5 | R@3 | R@5 | MRR@5 | Deterministic accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Semantic retrieval | {semantic['questions_evaluated']} | {semantic['precision_at_3']:.4f} | {semantic['precision_at_5']:.4f} | {semantic['recall_at_3']:.4f} | {semantic['recall_at_5']:.4f} | {semantic['reciprocal_rank_at_5']:.4f} | n/a |",
        f"| Mixed retrieval | {mixed['questions_evaluated']} | {mixed['precision_at_3']:.4f} | {mixed['precision_at_5']:.4f} | {mixed['recall_at_3']:.4f} | {mixed['recall_at_5']:.4f} | {mixed['reciprocal_rank_at_5']:.4f} | {mixed['deterministic_answer_accuracy']:.4f} |",
        f"| All retrieval-scored | {retrieval['questions_evaluated']} | {retrieval['precision_at_3']:.4f} | {retrieval['precision_at_5']:.4f} | {retrieval['recall_at_3']:.4f} | {retrieval['recall_at_5']:.4f} | {retrieval['reciprocal_rank_at_5']:.4f} | n/a |",
        f"| Metadata exact filters | {metadata['questions_evaluated']} | n/a | n/a | n/a | n/a | n/a | {metadata['record_set_exact_accuracy']:.4f} |",
        f"| Numerical analytics | {numerical['questions_evaluated']} | n/a | n/a | n/a | n/a | n/a | {numerical['deterministic_answer_accuracy']:.4f} |",
        "",
        "Metadata supported-only exact accuracy was "
        f"{metadata['supported_only_exact_accuracy']:.4f} across {metadata['supported_questions']} supported questions. "
        "Numerical supported-only accuracy was "
        f"{numerical['supported_only_accuracy']:.4f} across {numerical['supported_questions']} supported questions.",
        "",
        f"At least one relevant row appeared in the top five for {retrieval['questions_with_at_least_one_relevant_at_5']} of {retrieval['questions_evaluated']} retrieval-scored questions; {retrieval['questions_with_full_recall_at_5']} achieved full Recall@5.",
        "",
        "## Failure types",
        "",
    ]
    examples = report["failure_analysis"]["example_question_ids"]
    lines.extend(
        f"- `{name}`: {count} question(s); examples: {', '.join(examples.get(name, []))}"
        for name, count in failures.items()
    )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Retrieval metrics score Chroma row ranking. Metadata and numerical scores test current deterministic capabilities. No LLM-generated answer was produced or assessed, so these results must not be interpreted as answer-quality scores.",
            "",
            "Complete per-question rankings, record-set results, routed intents, failure examples, and ground-truth comparisons are in `benchmark_b2_baseline.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def save_report(
    report: dict[str, Any],
    output_path: Path | str = DEFAULT_REPORT_PATH,
    markdown_path: Path | str = DEFAULT_MARKDOWN_PATH,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    markdown_path = Path(markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return output_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RetailIQ Benchmark B2 baseline.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN_PATH)
    arguments = parser.parse_args()
    report = evaluate_b2(arguments.benchmark)
    json_path, markdown_path = save_report(report, arguments.output, arguments.markdown)
    print(f"Saved B2 JSON report to {json_path}")
    print(f"Saved B2 Markdown report to {markdown_path}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
