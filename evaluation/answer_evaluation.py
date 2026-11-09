import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.analytics import calculate_kpis
from evaluation.benchmark_validation import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_SOURCE_PATH,
    validate_benchmark,
)


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS_PATH = EVALUATION_ROOT / "fixtures" / "answer_annotations.json"
DEFAULT_ANSWER_REPORT_PATH = EVALUATION_ROOT / "reports" / "answer_quality_evaluation.json"
RELEVANCE_MAX_SCORE = 3
NUMERICAL_METRICS = {
    "NUM-001": ("total_revenue", 0),
    "NUM-002": ("total_profit", 0),
    "NUM-003": ("number_of_orders", 0),
    "NUM-004": ("average_order_value", 2),
    "NUM-005": ("profit_margin", 2),
}


class AnswerAnnotationError(ValueError):
    """Raised when human-review annotations are incomplete or inconsistent."""


def _validate_int_rows(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(row, bool) or not isinstance(row, int) or row < 1 for row in value
    ):
        raise AnswerAnnotationError(f"{field} must contain positive integer rows.")
    return value


def score_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    """Score one human-reviewed answer without making model-based judgments."""
    if not isinstance(annotation, dict):
        raise AnswerAnnotationError("An annotation must be an object.")
    for field in ("id", "question_id", "question", "answer", "reviewer"):
        if not isinstance(annotation.get(field), str) or not annotation[field].strip():
            raise AnswerAnnotationError(f"{field} is required.")

    evidence_rows = set(
        _validate_int_rows(annotation.get("retrieved_evidence_rows"), "retrieved_evidence_rows")
    )
    claims = annotation.get("claims")
    if not isinstance(claims, list):
        raise AnswerAnnotationError("claims must be a list.")

    factual_claims = 0
    supported_claims = 0
    claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise AnswerAnnotationError("Every claim must be an object.")
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip() or claim_id in claim_ids:
            raise AnswerAnnotationError("Claim IDs must be non-empty and unique.")
        claim_ids.add(claim_id)
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise AnswerAnnotationError("Every claim must contain text.")
        if not isinstance(claim.get("factual"), bool) or not isinstance(claim.get("supported"), bool):
            raise AnswerAnnotationError("Claim factual and supported labels must be boolean.")
        referenced_rows = set(_validate_int_rows(claim.get("evidence_rows"), "evidence_rows"))
        if not referenced_rows.issubset(evidence_rows):
            raise AnswerAnnotationError("Claim evidence must reference retrieved evidence rows.")
        if claim["supported"] and not referenced_rows:
            raise AnswerAnnotationError("A supported claim requires at least one evidence row.")
        if not claim["factual"] and (claim["supported"] or referenced_rows):
            raise AnswerAnnotationError("Non-factual statements cannot carry evidence labels.")
        if claim["factual"]:
            factual_claims += 1
            supported_claims += int(claim["supported"])

    relevance = annotation.get("relevance_review")
    if not isinstance(relevance, dict):
        raise AnswerAnnotationError("relevance_review is required.")
    relevance_score = relevance.get("score")
    if (
        isinstance(relevance_score, bool)
        or not isinstance(relevance_score, int)
        or not 0 <= relevance_score <= RELEVANCE_MAX_SCORE
    ):
        raise AnswerAnnotationError("Relevance score must be an integer from 0 to 3.")
    if not isinstance(relevance.get("notes"), str) or not relevance["notes"].strip():
        raise AnswerAnnotationError("Relevance review notes are required.")

    unsupported_claims = factual_claims - supported_claims
    faithfulness = supported_claims / factual_claims if factual_claims else 1.0
    unsupported_rate = unsupported_claims / factual_claims if factual_claims else 0.0
    return {
        "id": annotation["id"],
        "question_id": annotation["question_id"],
        "factual_claims": factual_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "faithfulness": faithfulness,
        "unsupported_claim_rate": unsupported_rate,
        "answer_relevance": relevance_score / RELEVANCE_MAX_SCORE,
        "relevance_score": relevance_score,
        "relevance_max_score": RELEVANCE_MAX_SCORE,
    }


def evaluate_annotations(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(annotations, list) or not annotations:
        raise AnswerAnnotationError("At least one reviewed annotation is required.")
    per_question = [score_annotation(annotation) for annotation in annotations]
    total_claims = sum(item["factual_claims"] for item in per_question)
    supported = sum(item["supported_claims"] for item in per_question)
    unsupported = sum(item["unsupported_claims"] for item in per_question)
    return {
        "answers_evaluated": len(per_question),
        "factual_claims": total_claims,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "faithfulness": supported / total_claims if total_claims else 1.0,
        "unsupported_claim_rate": unsupported / total_claims if total_claims else 0.0,
        "answer_relevance": sum(item["answer_relevance"] for item in per_question)
        / len(per_question),
        "per_question": per_question,
    }


def _expected_number(answer: str) -> float:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", answer)
    if not match:
        raise ValueError("The benchmark answer does not contain a numerical value.")
    return float(match.group(0).replace(",", ""))


def verify_numerical_answers(
    benchmark_path: Path | str = DEFAULT_BENCHMARK_PATH,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
) -> dict[str, Any]:
    """Compare five benchmark values with the existing deterministic KPI engine."""
    benchmark = validate_benchmark(benchmark_path, source_path)
    questions = {entry["id"]: entry for entry in benchmark["questions"]}
    metrics = calculate_kpis(pd.read_csv(source_path))
    results = []
    for question_id, (metric_name, decimals) in NUMERICAL_METRICS.items():
        metric = metrics[metric_name]
        expected = _expected_number(questions[question_id]["expected_answer"])
        actual = metric.get("value")
        correct = bool(
            metric.get("available")
            and isinstance(actual, (int, float))
            and math.isclose(round(float(actual), decimals), expected, abs_tol=10 ** -decimals / 2)
        )
        results.append(
            {
                "id": question_id,
                "question": questions[question_id]["question"],
                "metric": metric_name,
                "expected_value": expected,
                "actual_value": actual,
                "display_decimals": decimals,
                "correct": correct,
            }
        )
    correct_count = sum(item["correct"] for item in results)
    return {
        "status": "verified",
        "result_kind": "actual_deterministic",
        "questions_evaluated": len(results),
        "correct_answers": correct_count,
        "numerical_correctness": correct_count / len(results) if results else 0.0,
        "questions": results,
    }


def build_answer_quality_report(
    annotations_path: Path | str = DEFAULT_ANNOTATIONS_PATH,
) -> dict[str, Any]:
    fixture = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    fixture_scores = evaluate_annotations(fixture["annotations"])
    return {
        "evaluation_id": "retailiq-phase5-round3-answer-quality-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_llm_evaluation": {
            "status": "pending",
            "result_kind": "live_llm",
            "reason": "OpenRouter currently returns HTTP 429; no live LLM answers were scored.",
            "answers_evaluated": 0,
            "faithfulness": None,
            "answer_relevance": None,
            "unsupported_claim_rate": None,
        },
        "fixture_validation": {
            "status": "verified_fixture",
            "result_kind": "test_fixture_only",
            "warning": "Fixture scores validate scoring logic and are not actual LLM performance.",
            **fixture_scores,
        },
        "numerical_correctness": verify_numerical_answers(),
    }


def save_answer_quality_report(
    report: dict[str, Any],
    output_path: Path | str = DEFAULT_ANSWER_REPORT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    output = save_answer_quality_report(build_answer_quality_report())
    print(f"Saved answer-quality evaluation to {output}")


if __name__ == "__main__":
    main()
