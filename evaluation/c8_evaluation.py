"""Frozen C8 development regression and independent evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import pandas as pd

from backend.chroma_metadata import FULL_DATASET_UPLOAD_ID, load_metadata_source
from backend.dataset_selection import SelectedDataset
from backend.metadata_filtered_retrieval import retrieve_metadata_filtered_matches
from backend.query_planner import plan_query
from backend.rag_pipeline import answer_question
from backend.rag_retrieval import RetrievalMatch, retrieve_semantic_matches
from backend.structured_analytics import apply_plan_filters
from evaluation.hybrid_evaluation import evaluate_hybrid, render_markdown as render_development_markdown
from evaluation.retrieval_evaluation import calculate_question_metrics


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "evaluation" / "reports"
BENCHMARK_PATH = ROOT / "evaluation" / "retailiq_c8_independent.json"
DEVELOPMENT_REPORT_PATH = REPORT_ROOT / "c8_development_regression.json"
DEVELOPMENT_MARKDOWN_PATH = REPORT_ROOT / "c8_development_regression.md"
INDEPENDENT_REPORT_PATH = REPORT_ROOT / "c8_independent_evaluation.json"
INDEPENDENT_MARKDOWN_PATH = REPORT_ROOT / "c8_independent_evaluation.md"
POST_FIX_REPORT_PATH = REPORT_ROOT / "c8_post_fix_evaluation.json"
POST_FIX_MARKDOWN_PATH = REPORT_ROOT / "c8_post_fix_evaluation.md"
FREEZE_MANIFEST_PATH = REPORT_ROOT / "c8_freeze_manifest.json"
HISTORICAL_C6_PATHS = (
    ROOT / "evaluation" / "reports" / "hybrid_c6_held_out.json",
    ROOT / "evaluation" / "reports" / "hybrid_c6_held_out.md",
)
FROZEN_BACKEND_FILES = (
    ROOT / "backend" / "query_planner.py",
    ROOT / "backend" / "structured_analytics.py",
    ROOT / "backend" / "metadata_filtered_retrieval.py",
    ROOT / "backend" / "rag_pipeline.py",
    ROOT / "backend" / "llm_service.py",
)

Retriever = Callable[[str, str, int], list[RetrievalMatch]]
FilteredRetriever = Callable[[str, Any, int], list[RetrievalMatch]]


class NoEvaluationLLM:
    def generate_grounded_answer(
        self, _question: str, _matches: list[RetrievalMatch]
    ) -> str:
        return "Semantic-search matches were returned; answer generation was not scored here."


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> dict[str, Any]:
    manifest = {
        "freeze_id": "retailiq-c7-backend-freeze-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_files": {
            file.relative_to(ROOT).as_posix(): _sha256(file)
            for file in FROZEN_BACKEND_FILES
        },
        "historical_c6_files": {
            file.relative_to(ROOT).as_posix(): _sha256(file)
            for file in HISTORICAL_C6_PATHS
        },
        "policy": "Backend hashes were frozen before the independent questions were generated or evaluated.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> bool:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = {**manifest["backend_files"], **manifest["historical_c6_files"]}
    return all(_sha256(ROOT / relative) == digest for relative, digest in files.items())


def _rows(frame: pd.DataFrame, mask: pd.Series) -> list[int]:
    return [int(value) for value in frame.index[mask]]


def _float(value: Any) -> float:
    return round(float(value), 6)


def _entry(
    question_id: str,
    question_type: str,
    question: str,
    ground_truth: dict[str, Any],
    relevant_rows: list[int] | None = None,
) -> dict[str, Any]:
    answer = ground_truth.get("expected_answer")
    if answer is None:
        answer = "Independently calculated from data/sample_superstore.csv with Pandas."
    result = {
        "id": question_id,
        "type": question_type,
        "question": question,
        "expected_answer": answer,
        "ground_truth": {
            "verification": "Direct Pandas calculation from the original CSV; no LLM output was used.",
            **{key: value for key, value in ground_truth.items() if key != "expected_answer"},
        },
    }
    if relevant_rows is not None:
        result["relevant_source_rows"] = relevant_rows
    return result


def build_independent_benchmark(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = load_metadata_source() if frame is None else frame.copy()
    if len(frame) != 9_994:
        raise ValueError("The independent benchmark requires all 9,994 source rows.")
    frame.index = pd.RangeIndex(1, len(frame) + 1)
    dates = pd.to_datetime(frame["Order Date"], format="mixed")
    shipping_days = (
        pd.to_datetime(frame["Ship Date"], format="mixed") - dates
    ).dt.days
    questions: list[dict[str, Any]] = []

    filter_specs = (
        ("C8I2-FIL-001", "How many Consumer line items in the South had no discount?", (frame["Segment"] == "Consumer") & (frame["Region"] == "South") & frame["Discount"].eq(0)),
        ("C8I2-FIL-002", "How many Home Office Office Supplies line items had discount at least 10%?", (frame["Segment"] == "Home Office") & (frame["Category"] == "Office Supplies") & frame["Discount"].ge(0.1)),
        ("C8I2-FIL-003", "How many loss-making Second Class line items were ordered during 2017?", (frame["Ship Mode"] == "Second Class") & dates.dt.year.eq(2017) & frame["Profit"].lt(0)),
        ("C8I2-FIL-004", "How many Technology line items ordered in New York had sales above 750?", (frame["Category"] == "Technology") & (frame["State"] == "New York") & frame["Sales"].gt(750)),
        ("C8I2-FIL-005", "How many West region line items took more than four calendar days to ship?", (frame["Region"] == "West") & shipping_days.gt(4)),
    )
    for question_id, question, mask in filter_specs:
        relevant = _rows(frame, mask)
        questions.append(_entry(question_id, "filtering", question, {"count": len(relevant), "expected_answer": f"{len(relevant)} matching source rows."}, relevant))

    def grouped(mask: pd.Series, dimension: str, metric: str) -> dict[str, float]:
        values = frame.loc[mask].groupby(dimension)[metric].sum()
        return {str(key): _float(value) for key, value in values.items()}

    questions.extend(
        [
            _entry("C8I2-NUM-001", "numerical", "What were total sales for each product category during 2014?", {"kind": "grouped", "dimension": "Category", "metric": "sales", "value": grouped(dates.dt.year.eq(2014), "Category", "Sales")}),
            _entry("C8I2-NUM-002", "numerical", "Show total profit per sub-category in the East region.", {"kind": "grouped", "dimension": "Sub-Category", "metric": "profit", "value": grouped(frame["Region"].eq("East"), "Sub-Category", "Profit")}),
            _entry("C8I2-NUM-003", "numerical", "What was the average order value for the Home Office segment?", {"kind": "scalar", "metric": "sales", "value": _float(frame.loc[frame["Segment"].eq("Home Office")].groupby("Order ID")["Sales"].sum().mean())}),
            _entry("C8I2-NUM-004", "numerical", "What was total sales for customer Aaron Hawkins?", {"kind": "scalar", "metric": "sales", "value": _float(frame.loc[frame["Customer Name"].eq("Aaron Hawkins"), "Sales"].sum())}),
            _entry("C8I2-NUM-005", "numerical", "Show total Furniture sales for each order year.", {"kind": "grouped", "dimension": "order_year", "metric": "sales", "value": {str(key): _float(value) for key, value in frame.loc[frame["Category"].eq("Furniture")].assign(order_year=dates[frame["Category"].eq("Furniture")].dt.year).groupby("order_year")["Sales"].sum().items()}}),
        ]
    )

    def ranked(dimension: str, metric: str, n: int, ascending: bool = False, mask: pd.Series | None = None) -> list[dict[str, Any]]:
        source = frame if mask is None else frame.loc[mask]
        values = source.groupby(dimension)[metric].sum().sort_values(ascending=ascending, kind="stable").head(n)
        return [{dimension: str(key), metric: _float(value)} for key, value in values.items()]

    questions.extend(
        [
            _entry("C8I2-RANK-001", "ranking", "What are the top three states by total sales?", {"dimension": "State", "metric": "Sales", "records": ranked("State", "Sales", 3)}),
            _entry("C8I2-RANK-002", "ranking", "What are the top two product categories by total quantity?", {"dimension": "Category", "metric": "Quantity", "records": ranked("Category", "Quantity", 2)}),
            _entry("C8I2-RANK-003", "ranking", "What are the top four sub-categories by total sales?", {"dimension": "Sub-Category", "metric": "Sales", "records": ranked("Sub-Category", "Sales", 4)}),
            _entry("C8I2-RANK-004", "ranking", "Who are the top four customers by total profit?", {"dimension": "Customer ID", "metric": "Profit", "records": ranked("Customer ID", "Profit", 4)}),
            _entry("C8I2-RANK-005", "ranking", "What are the top three orders by total sales?", {"dimension": "Order ID", "metric": "Sales", "records": ranked("Order ID", "Sales", 3)}),
        ]
    )

    east_tables = (frame["Sub-Category"] == "Tables") & (frame["Region"] == "East") & frame["Discount"].gt(0)
    south_binders = (frame["Sub-Category"] == "Binders") & (frame["Region"] == "South") & frame["Profit"].gt(0)
    corporate = frame["Segment"].eq("Corporate")
    corporate_sales = frame.loc[corporate].groupby("Customer ID")["Sales"].sum().sort_values(ascending=False, kind="stable")
    corporate_customer = str(corporate_sales.index[0])
    mouse = frame["Product Name"].str.contains("mouse", case=False, regex=False, na=False)
    mouse_regions = frame.loc[mouse].groupby("Region")["Profit"].sum().sort_values(ascending=False, kind="stable")
    notebooks = frame["Product Name"].str.contains("notebook", case=False, regex=False, na=False) & frame["Segment"].eq("Consumer")

    def top_rows(mask: pd.Series, metric: str, n: int, ascending: bool = False) -> list[int]:
        return [int(value) for value in frame.loc[mask].sort_values(metric, ascending=ascending, kind="stable").head(n).index]

    questions.extend(
        [
            _entry("C8I2-MIX-001", "mixed", "For discounted table line items in the East, which two had the highest sales and what was total sales?", {"top_source_rows": top_rows(east_tables, "Sales", 2), "value": _float(frame.loc[east_tables, "Sales"].sum())}, _rows(frame, east_tables)),
            _entry("C8I2-MIX-002", "mixed", "For profitable binder line items in the South, which line item had the highest profit and what was median sales?", {"top_source_rows": top_rows(south_binders, "Profit", 1), "value": _float(frame.loc[south_binders, "Sales"].median())}, _rows(frame, south_binders)),
            _entry("C8I2-MIX-003", "mixed", "Among Corporate customers, who had the highest total sales and how many distinct orders did that customer place?", {"customer_id": corporate_customer, "sales": _float(corporate_sales.iloc[0]), "distinct_orders": int(frame.loc[corporate & frame["Customer ID"].eq(corporate_customer), "Order ID"].nunique())}, _rows(frame, corporate)),
            _entry("C8I2-MIX-004", "mixed", "For products containing 'mouse', which region generated the most profit and which matching line item had the highest profit?", {"group": {"Region": str(mouse_regions.index[0]), "Profit": _float(mouse_regions.iloc[0])}, "top_source_rows": top_rows(mouse, "Profit", 1)}, _rows(frame, mouse)),
            _entry("C8I2-MIX-005", "mixed", "For product line items containing 'notebook' bought by Consumer customers, which line item had the highest profit and what was total sales?", {"top_source_rows": top_rows(notebooks, "Profit", 1), "value": _float(frame.loc[notebooks, "Sales"].sum())}, _rows(frame, notebooks)),
        ]
    )

    semantic_specs = (
        ("C8I2-SEM-001", "Find products whose names mention markers.", "marker"),
        ("C8I2-SEM-002", "Which catalog products are speakers?", "speaker"),
        ("C8I2-SEM-003", "Find products related to printers.", "printer"),
        ("C8I2-SEM-004", "Which product names mention lamps?", "lamp"),
        ("C8I2-SEM-005", "Find products whose names include batteries.", "battery"),
    )
    for question_id, question, keyword in semantic_specs:
        relevant = _rows(frame, frame["Product Name"].str.contains(keyword, case=False, regex=False, na=False))
        questions.append(_entry(question_id, "semantic", question, {"relevance_definition": f"Product Name contains '{keyword}' case-insensitively.", "matching_row_count": len(relevant)}, relevant))

    return {
        "benchmark_id": "retailiq-c8-independent-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "data/sample_superstore.csv",
        "source_rows": len(frame),
        "frozen_backend_manifest": "evaluation/reports/c8_freeze_manifest.json",
        "development_use_prohibited": True,
        "relevance_policy": "Filtering and mixed questions use all rows satisfying the stated Pandas predicate. Ranking uses exact stable aggregate or row order. Semantic relevance is the complete case-insensitive Product Name substring set stated per question.",
        "questions": questions,
    }


def save_independent_benchmark(
    benchmark: dict[str, Any], path: Path = BENCHMARK_PATH
) -> Path:
    path.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    return path


def _close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-5
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_close(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_close(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _record_rows(records: list[dict[str, Any]]) -> list[int]:
    return [int(record["source_row"]) for record in records if "source_row" in record and record.get("result_kind") != "group_ranking"]


def evaluate_independent(
    benchmark_path: Path = BENCHMARK_PATH,
    retriever: Retriever = retrieve_semantic_matches,
    filtered_retriever: FilteredRetriever = retrieve_metadata_filtered_matches,
    require_frozen_backend: bool = True,
) -> dict[str, Any]:
    freeze_verified = verify_freeze_manifest()
    if require_frozen_backend and not freeze_verified:
        raise RuntimeError("The frozen backend or historical C6 artifacts changed after freeze.")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    frame = load_metadata_source()
    dataset = SelectedDataset(UUID(FULL_DATASET_UPLOAD_ID), "sample_superstore.csv", frame)
    dimensions = {
        "region": frame["Region"].unique().tolist(),
        "state": frame["State"].unique().tolist(),
        "city": frame["City"].unique().tolist(),
        "customer_name": frame["Customer Name"].unique().tolist(),
    }
    results: list[dict[str, Any]] = []
    semantic_metrics: list[dict[str, float]] = []
    for entry in benchmark["questions"]:
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
        calculation = answer.calculation or {}
        records = calculation.get("records", [])
        result: dict[str, Any] = {
            "id": entry["id"], "type": entry["type"], "question": entry["question"],
            "mode": answer.mode, "plan": plan.as_dict(), "calculation": answer.calculation,
        }
        expected = entry["ground_truth"]
        if entry["type"] == "filtering":
            actual_rows = sorted(int(value) for value in apply_plan_filters(frame, plan).index)
            expected_rows = entry["relevant_source_rows"]
            result["correct"] = actual_rows == expected_rows
            result["actual_count"] = len(actual_rows)
            result["expected_count"] = len(expected_rows)
        elif entry["type"] == "numerical":
            if expected["kind"] == "scalar":
                actual_value = calculation.get("value")
            else:
                dimension = expected["dimension"]
                actual_value = {
                    str(record[dimension]): _float(record["value"])
                    for record in records
                    if dimension in record and record.get("value") is not None
                }
            result["actual_value"] = actual_value
            result["expected_value"] = expected["value"]
            result["correct"] = _close(actual_value, expected["value"])
        elif entry["type"] == "ranking":
            dimension, metric = expected["dimension"], expected["metric"]
            actual_records = [
                {dimension: str(record[dimension]), metric: _float(record[metric])}
                for record in records
                if dimension in record and record.get(metric) is not None
            ]
            result["actual_records"] = actual_records
            result["expected_records"] = expected["records"]
            result["correct"] = _close(actual_records, expected["records"])
        elif entry["type"] == "mixed":
            if entry["id"] == "C8I2-MIX-003":
                actual = (
                    {
                        "customer_id": records[0].get("Customer ID"),
                        "sales": _float(records[0]["Sales"]),
                        "distinct_orders": int(records[0]["distinct_order_count"]),
                    }
                    if records and records[0].get("Sales") is not None and records[0].get("distinct_order_count") is not None
                    else None
                )
                target = {key: expected[key] for key in ("customer_id", "sales", "distinct_orders")}
            elif entry["id"] == "C8I2-MIX-004":
                groups = [record for record in records if record.get("result_kind") == "group_ranking"]
                actual = {
                    "group": {
                        "Region": groups[0]["Region"],
                        "Profit": _float(groups[0]["Profit"]),
                    }
                    if groups and "Region" in groups[0] and groups[0].get("Profit") is not None
                    else None,
                    "top_source_rows": _record_rows(records),
                }
                target = {"group": expected["group"], "top_source_rows": expected["top_source_rows"]}
            else:
                actual = {"top_source_rows": _record_rows(records), "value": calculation.get("value")}
                target = {"top_source_rows": expected["top_source_rows"], "value": expected["value"]}
            result["actual"] = actual
            result["expected"] = target
            result["correct"] = _close(actual, target)
        else:
            matches = retriever(FULL_DATASET_UPLOAD_ID, entry["question"], 5)
            retrieved = [match.source_row for match in matches if match.source_row is not None]
            scored = calculate_question_metrics(retrieved, entry["relevant_source_rows"], (3, 5))
            metrics = {"precision_at_3": scored["precision_at_3"], "precision_at_5": scored["precision_at_5"], "recall_at_3": scored["recall_at_3"], "recall_at_5": scored["recall_at_5"], "reciprocal_rank_at_5": scored["reciprocal_rank"]}
            result["retrieved_source_rows"] = retrieved
            result["metrics"] = metrics
            semantic_metrics.append(metrics)
        results.append(result)

    def accuracy(kind: str) -> float:
        selected = [item for item in results if item["type"] == kind]
        return sum(bool(item.get("correct")) for item in selected) / len(selected)

    def mean(name: str) -> float:
        return sum(item[name] for item in semantic_metrics) / len(semantic_metrics)

    return {
        "evaluation_id": "retailiq-c8-independent-evaluation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_sha256": _sha256(benchmark_path),
        "freeze_manifest_sha256": _sha256(FREEZE_MANIFEST_PATH),
        "original_freeze_verified": freeze_verified,
        "question_count": len(results), "llm_calls": 0,
        "metrics": {
            "filtering_accuracy": accuracy("filtering"),
            "numerical_accuracy": accuracy("numerical"),
            "ranking_accuracy": accuracy("ranking"),
            "mixed_accuracy": accuracy("mixed"),
            "semantic_retrieval": {name: mean(name) for name in ("precision_at_3", "precision_at_5", "recall_at_3", "recall_at_5", "reciprocal_rank_at_5")},
        },
        "results": results,
    }


def render_independent_markdown(report: dict[str, Any]) -> str:
    metrics, semantic = report["metrics"], report["metrics"]["semantic_retrieval"]
    failures = [item["id"] for item in report["results"] if item["type"] != "semantic" and not item.get("correct")]
    freeze_note = (
        "Original backend freeze verified before evaluation."
        if report.get("original_freeze_verified", True)
        else "Original freeze manifest was preserved; its backend hash differs because this is the authorized post-fix evaluation."
    )
    return "\n".join(
        [
            "# RetailIQ C8 Independent Evaluation", "",
            f"Questions: {report['question_count']}. LLM calls: {report['llm_calls']}. {freeze_note}", "",
            "| Metric | Result |", "|---|---:|",
            f"| Filtering accuracy | {metrics['filtering_accuracy']:.4f} |",
            f"| Numerical accuracy | {metrics['numerical_accuracy']:.4f} |",
            f"| Ranking accuracy | {metrics['ranking_accuracy']:.4f} |",
            f"| Mixed accuracy | {metrics['mixed_accuracy']:.4f} |",
            f"| Semantic P@3 | {semantic['precision_at_3']:.4f} |",
            f"| Semantic P@5 | {semantic['precision_at_5']:.4f} |",
            f"| Semantic R@3 | {semantic['recall_at_3']:.4f} |",
            f"| Semantic R@5 | {semantic['recall_at_5']:.4f} |",
            f"| Semantic MRR@5 | {semantic['reciprocal_rank_at_5']:.4f} |", "",
            "Semantic relevance uses each question's complete, frozen Product Name substring set; low recall is expected when that set is much larger than five rows.", "",
            "## Deterministic failures", "", ", ".join(failures) if failures else "None.", "",
        ]
    )


def save_report(report: dict[str, Any], json_path: Path, markdown_path: Path, markdown: str) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("freeze", "generate", "development", "independent", "post_fix")
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        write_freeze_manifest()
        return
    if args.stage == "generate":
        if not verify_freeze_manifest():
            raise RuntimeError("Freeze verification failed before benchmark generation.")
        save_independent_benchmark(build_independent_benchmark())
        return
    if args.stage == "development":
        report = evaluate_hybrid("development")
        report["evaluation_id"] = "retailiq-c8-development-regression-v1"
        save_report(report, DEVELOPMENT_REPORT_PATH, DEVELOPMENT_MARKDOWN_PATH, render_development_markdown(report).replace("C5 Development", "C8 Development Regression"))
        return
    if args.stage == "independent":
        report = evaluate_independent()
        save_report(report, INDEPENDENT_REPORT_PATH, INDEPENDENT_MARKDOWN_PATH, render_independent_markdown(report))
        return
    report = evaluate_independent(require_frozen_backend=False)
    original = json.loads(INDEPENDENT_REPORT_PATH.read_text(encoding="utf-8"))
    report["evaluation_id"] = "retailiq-c8-post-fix-evaluation-v1"
    report["comparison"] = {
        "original_report": "evaluation/reports/c8_independent_evaluation.json",
        "original_report_sha256": _sha256(INDEPENDENT_REPORT_PATH),
        "before": original["metrics"],
        "after": report["metrics"],
        "note": "The original freeze is expected not to verify after the targeted planner fix; the manifest itself is unchanged.",
    }
    save_report(
        report,
        POST_FIX_REPORT_PATH,
        POST_FIX_MARKDOWN_PATH,
        render_independent_markdown(report).replace(
            "C8 Independent Evaluation", "C8 Post-Fix Evaluation"
        ),
    )


if __name__ == "__main__":
    main()
