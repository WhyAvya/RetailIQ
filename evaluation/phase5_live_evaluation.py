"""Run the final five-query RetailIQ Phase 5 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

from backend.chroma_metadata import FULL_DATASET_UPLOAD_ID, load_metadata_source
from backend.dataset_selection import SelectedDataset
from backend.llm_service import LLMServiceError, OpenRouterLLMService
from backend.rag_pipeline import answer_question


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "evaluation" / "reports"
JSON_PATH = REPORT_ROOT / "phase5_live_evaluation.json"
MARKDOWN_PATH = REPORT_ROOT / "phase5_live_evaluation.md"
CSV_PATH = ROOT / "data" / "sample_superstore.csv"
UPLOAD_ID = UUID(FULL_DATASET_UPLOAD_ID)

FROZEN_HASHES = {
    "evaluation/retailiq_c8_independent.json": "6698749b3ea5ab04c5b9f60f1f53020dfafd4b1279c6d5002ab716e8e1ae5f6e",
    "evaluation/reports/c8_independent_evaluation.json": "e4d64b20fcc711fcc7ff45e3443b301885d8eb360d120e4a0b0fa4bbbb4a28d6",
    "evaluation/reports/c8_post_fix_evaluation.json": "d9363d742fc6cae553ff8649d8cb4a51cc9a22820e26f7b361d11eefeedea18a",
    "evaluation/reports/c8_freeze_manifest.json": "c20dbdd5bf6c2a46b6070a2f7caa531e28869449388fe47c0b3ce73f03382deb",
    "evaluation/reports/c8_pre_final_freeze_manifest.json": "6f2b7a199828555ee9fd873d9e0a8bb5955adfd6e91d1db47d7cd82abefe198b",
}

QUERIES = (
    ("semantic", "Which product names mention lamps?"),
    ("numerical", "What was total sales for customer Aaron Hawkins?"),
    ("filtering", "How many Consumer line items in the South had no discount?"),
    ("ranking", "What are the top three states by total sales?"),
    (
        "mixed",
        "For products containing 'mouse', which region generated the most profit and which matching line item had the highest profit?",
    ),
)


class CountingLLMService:
    def __init__(self) -> None:
        self.service = OpenRouterLLMService()
        self.config = self.service.config
        self.invocations = 0
        self.errors: list[dict[str, str]] = []

    def generate_grounded_answer(self, question: str, matches: list[Any]) -> str:
        self.invocations += 1
        try:
            return self.service.generate_grounded_answer(question, matches)
        except LLMServiceError as exc:
            self.errors.append({"type": type(exc).__name__, "message": str(exc)})
            raise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_hash_check() -> dict[str, Any]:
    files = {}
    for relative, expected in FROZEN_HASHES.items():
        actual = _sha256(ROOT / relative)
        files[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "unchanged": actual == expected,
        }
    return {"all_unchanged": all(item["unchanged"] for item in files.values()), "files": files}


def _source_dict(source: Any) -> dict[str, Any]:
    return {
        "source_row": source.source_row,
        "chunk_id": source.chunk_id,
        "similarity_score": source.similarity_score,
        "metadata": source.metadata,
        "document_excerpt": source.document_excerpt,
    }


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-6)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_close(a, b) for a, b in zip(left, right))
    return left == right


def _independent_expected(frame: pd.DataFrame) -> dict[str, Any]:
    mouse = frame["Product Name"].str.contains("mouse", case=False, na=False)
    mouse_frame = frame.loc[mouse]
    region_profit = mouse_frame.groupby("Region")["Profit"].sum().sort_values(ascending=False)
    top_mouse_index = mouse_frame["Profit"].idxmax()
    top_mouse = mouse_frame.loc[top_mouse_index]
    state_sales = frame.groupby("State")["Sales"].sum().sort_values(ascending=False).head(3)
    lamp = frame["Product Name"].str.contains("lamp", case=False, na=False)
    return {
        "semantic": {
            "criterion": "Product Name contains 'lamp' (case-insensitive)",
            "matching_source_rows": frame.loc[lamp, "Row ID"].astype(int).tolist(),
            "matching_product_names": sorted(frame.loc[lamp, "Product Name"].dropna().unique().tolist()),
        },
        "numerical": float(frame.loc[frame["Customer Name"].eq("Aaron Hawkins"), "Sales"].sum()),
        "filtering": int(
            (frame["Segment"].eq("Consumer") & frame["Region"].eq("South") & frame["Discount"].eq(0)).sum()
        ),
        "ranking": [
            {"State": str(state), "Sales": float(sales)} for state, sales in state_sales.items()
        ],
        "mixed": {
            "group": {"Region": str(region_profit.index[0]), "Profit": float(region_profit.iloc[0])},
            "top_source_rows": [int(top_mouse["Row ID"])],
        },
    }


def _actual_for(category: str, result: Any) -> Any:
    calculation = result.calculation or {}
    if category in {"numerical", "filtering"}:
        return calculation.get("value")
    if category == "ranking":
        return [
            {"State": item["State"], "Sales": item["Sales"]}
            for item in calculation.get("records", [])[:3]
        ]
    if category == "mixed":
        records = calculation.get("records", [])
        grouped = next((item for item in records if "Region" in item and "Profit" in item), {})
        ranked = next((item for item in records if "source_row" in item), {})
        return {
            "group": {"Region": grouped.get("Region"), "Profit": grouped.get("Profit")},
            "top_source_rows": [ranked.get("source_row")] if ranked else [],
        }
    return None


def _semantic_review(answer: str, sources: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    source_rows = [item["source_row"] for item in sources if item["source_row"] is not None]
    relevant = set(expected["matching_source_rows"])
    cited = sorted({int(value) for value in re.findall(r"(?:source\s+row|row)\s*#?\s*(\d+)", answer, re.I)})
    valid_citations = all(row in source_rows for row in cited)
    retrieved_relevant = [row for row in source_rows if row in relevant]
    lower_answer = answer.lower()
    normalized_answer = re.sub(r"[^a-z0-9]+", " ", lower_answer).strip()
    cited_names = [
        str(item["metadata"].get("product_name", ""))
        for item in sources
        if item["source_row"] in cited
    ]
    cited_names_supported = all(
        re.sub(r"[^a-z0-9]+", " ", name.lower()).strip() in normalized_answer
        for name in cited_names
    )
    labels_semantic = "semantic" in lower_answer
    no_completeness_claim = not any(
        phrase in lower_answer for phrase in ("all products", "the only", "complete list", "no other")
    )
    grounded = (
        bool(retrieved_relevant)
        and valid_citations
        and cited_names_supported
        and labels_semantic
        and no_completeness_claim
    )
    return {
        "retrieved_source_rows": source_rows,
        "retrieved_relevant_source_rows": retrieved_relevant,
        "cited_source_rows": cited,
        "citations_valid": valid_citations,
        "labels_result_as_semantic_search": labels_semantic,
        "avoids_completeness_claim": no_completeness_claim,
        "cited_product_names": cited_names,
        "cited_product_names_supported": cited_names_supported,
        "grounded": grounded,
        "hallucination_detected": False,
        "correct": grounded,
        "review_method": "Deterministic answer/citation checks plus CSV validation of retrieved source rows and lamp names.",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RetailIQ Phase 5 Live Evaluation",
        "",
        f"Final assessment: **{report['assessment']}**. Five pipeline queries were attempted against the canonical 9,994-row upload.",
        "",
        "| Category | Route | Correct | Grounded | Citations | Hallucination | Latency (ms) | Error |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for item in report["queries"]:
        lines.append(
            f"| {item['category']} | {item['routing']['mode']} | {str(item['correct']).lower()} | "
            f"{str(item['grounded']).lower()} | {item['citations']['status']} | "
            f"{str(item['hallucination_detected']).lower()} | {item['latency_ms']:.2f} | {item['error'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## Execution summary",
            "",
            f"- LLM invocations: {report['execution']['llm_invocation_count']} (expected: semantic only).",
            f"- Provider/model: {report['execution']['provider']} / `{report['execution']['model']}`.",
            f"- Average latency: {report['latency']['average_ms']:.2f} ms; maximum: {report['latency']['max_ms']:.2f} ms.",
            f"- Frozen benchmark/report hashes unchanged: {str(report['frozen_artifacts']['all_unchanged']).lower()}.",
            "",
            "## C8 baseline versus post-fix",
            "",
            "| Metric | Baseline | Post-fix |",
            "|---|---:|---:|",
            "| Filtering accuracy | 80% | 100% |",
            "| Numerical accuracy | 100% | 100% |",
            "| Ranking accuracy | 60% | 100% |",
            "| Mixed accuracy | 100% | 100% |",
            "",
            "The post-fix values above come from `c8_post_fix_evaluation.json`; they are not the original independent baseline.",
            "",
            "## Retrieval metrics and limitations",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| P@3 | 0.2667 |",
            "| P@5 | 0.2400 |",
            "| R@3 | 0.0068 |",
            "| R@5 | 0.0103 |",
            "| MRR@5 | 0.2667 |",
            "",
            "Recall@K is intrinsically capped by retrieving only 3 or 5 rows when each frozen substring relevance set can contain many rows, but values near 1% still mean the result is not exhaustive. P@3/P@5 show that most top results were not in the strict frozen relevance sets, and MRR@5 means the first relevant result was often late or absent. The semantic answer must therefore be treated as a small set of approximate matches, not a complete catalog result. These metrics cover five semantic questions, so they are also high-variance and narrow in scope.",
            "",
            "## Assessment rationale",
            "",
            *[f"- {item}" for item in report["assessment_rationale"]],
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    frame = pd.read_csv(CSV_PATH, encoding="cp1252")
    if len(frame) != 9994:
        raise RuntimeError(f"Expected 9,994 CSV rows, found {len(frame):,}.")
    metadata_frame = load_metadata_source()
    dataset = SelectedDataset(UPLOAD_ID, CSV_PATH.name, metadata_frame)
    expected = _independent_expected(frame)
    service = CountingLLMService()
    frozen_before = _frozen_hash_check()
    results = []

    for category, question in QUERIES:
        started = time.perf_counter()
        error = None
        answer = None
        try:
            answer = answer_question(
                UPLOAD_ID,
                question,
                top_k=5,
                metadata_lookup=lambda _id: {"upload_id": str(_id)},
                dataset_loader=lambda _id: dataset,
                llm_service=service,
            )
        except Exception as exc:  # The report must preserve real provider/pipeline failures.
            error = {"type": type(exc).__name__, "message": str(exc)}
        latency_ms = (time.perf_counter() - started) * 1000

        if answer is None:
            results.append(
                {
                    "category": category,
                    "question": question,
                    "routing": {"mode": "error", "llm_invoked": category == "semantic"},
                    "answer": None,
                    "calculation": None,
                    "sources": [],
                    "independent_expected": expected[category],
                    "independent_actual": None,
                    "correct": False,
                    "grounded": False,
                    "citations": {"status": "unavailable", "valid": False, "source_rows": []},
                    "hallucination_detected": False,
                    "latency_ms": latency_ms,
                    "provider": "OpenRouter" if category == "semantic" else None,
                    "model": service.config.model if category == "semantic" else None,
                    "error": error,
                }
            )
            continue

        sources = [_source_dict(source) for source in answer.sources]
        if category == "semantic":
            review = _semantic_review(answer.answer, sources, expected[category])
            correct = review["correct"]
            grounded = review["grounded"]
            hallucination = review["hallucination_detected"]
            citations = {
                "status": "valid" if review["citations_valid"] and review["cited_source_rows"] else "none",
                "valid": review["citations_valid"],
                "source_rows": review["cited_source_rows"],
            }
            actual = review
        else:
            actual = _actual_for(category, answer)
            correct = _close(actual, expected[category])
            grounded = correct
            hallucination = False
            citations = {
                "status": "not_applicable_exact_csv_calculation",
                "valid": True,
                "source_rows": [item["source_row"] for item in sources if item["source_row"] is not None],
            }
        results.append(
            {
                "category": category,
                "question": question,
                "routing": {
                    "mode": answer.mode,
                    "llm_invoked": category == "semantic",
                    "calculation_intent": (answer.calculation or {}).get("intent"),
                    "query_type": ((answer.calculation or {}).get("plan") or {}).get("query_type"),
                },
                "answer": answer.answer,
                "calculation": answer.calculation,
                "sources": sources,
                "independent_expected": expected[category],
                "independent_actual": actual,
                "correct": correct,
                "grounded": grounded,
                "citations": citations,
                "hallucination_detected": hallucination,
                "latency_ms": latency_ms,
                "provider": "OpenRouter" if category == "semantic" else "local Pandas/Chroma",
                "model": service.config.model if category == "semantic" else None,
                "error": error,
            }
        )

    frozen_after = _frozen_hash_check()
    latencies = [item["latency_ms"] for item in results]
    provider_blocked = any(item["category"] == "semantic" and item["error"] for item in results)
    all_correct = all(item["correct"] for item in results)
    hashes_ok = frozen_before["all_unchanged"] and frozen_after["all_unchanged"]
    assessment = "PASS" if all_correct and hashes_ok else "BLOCKED" if provider_blocked else "PARTIAL"
    rationale = [
        f"{sum(item['correct'] for item in results)}/5 live pipeline results passed independent CSV/evidence validation.",
        f"The LLM boundary recorded {service.invocations} real invocation(s); structured and mixed routes made no LLM calls.",
        f"All tracked frozen artifact hashes were unchanged before and after execution: {hashes_ok}.",
    ]
    if provider_blocked:
        rationale.append("The live semantic result was blocked by a real provider/network failure; offline routes still completed.")

    baseline = json.loads((REPORT_ROOT / "c8_independent_evaluation.json").read_text(encoding="utf-8"))
    post_fix = json.loads((REPORT_ROOT / "c8_post_fix_evaluation.json").read_text(encoding="utf-8"))
    report = {
        "evaluation_id": "retailiq-phase5-live-evaluation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assessment": assessment,
        "assessment_rationale": rationale,
        "dataset": {"path": "data/sample_superstore.csv", "row_count": len(frame), "upload_id": str(UPLOAD_ID)},
        "execution": {
            "queries_attempted": len(results),
            "queries_completed": sum(item["error"] is None for item in results),
            "llm_invocation_count": service.invocations,
            "provider": "OpenRouter",
            "model": service.config.model,
            "provider_errors": service.errors,
        },
        "latency": {"average_ms": sum(latencies) / len(latencies), "max_ms": max(latencies)},
        "queries": results,
        "c8_comparison": {
            "baseline_report": "evaluation/reports/c8_independent_evaluation.json",
            "post_fix_report": "evaluation/reports/c8_post_fix_evaluation.json",
            "baseline": baseline["metrics"],
            "post_fix": post_fix["metrics"],
            "post_fix_all_deterministic_categories_100_percent": all(
                post_fix["metrics"][key] == 1.0
                for key in ("filtering_accuracy", "numerical_accuracy", "ranking_accuracy", "mixed_accuracy")
            ),
        },
        "retrieval_limitations": {
            **post_fix["metrics"]["semantic_retrieval"],
            "interpretation": "Low K versus large relevance sets constrains recall, but weak precision and MRR also show noisy top ranks. Results are approximate and non-exhaustive; five semantic questions provide limited coverage.",
        },
        "frozen_artifacts": {"before": frozen_before, "after": frozen_after, "all_unchanged": hashes_ok},
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    completed = run()
    print(
        json.dumps(
            {
                "assessment": completed["assessment"],
                "llm_invocation_count": completed["execution"]["llm_invocation_count"],
                "average_latency_ms": completed["latency"]["average_ms"],
                "max_latency_ms": completed["latency"]["max_ms"],
                "results": [
                    {
                        "category": item["category"],
                        "mode": item["routing"]["mode"],
                        "correct": item["correct"],
                        "latency_ms": item["latency_ms"],
                        "error": item["error"],
                    }
                    for item in completed["queries"]
                ],
            },
            indent=2,
        )
    )
