import json
from pathlib import Path
from typing import Any

from evaluation.answer_evaluation import DEFAULT_ANSWER_REPORT_PATH
from evaluation.retrieval_evaluation import DEFAULT_REPORT_PATH


REPORTS_DIRECTORY = Path(__file__).resolve().parent / "reports"
OPTIMIZED_C8_REPORT_PATH = REPORTS_DIRECTORY / "semantic_retrieval_optimization.json"
STRUCTURED_C8_REPORT_PATH = REPORTS_DIRECTORY / "c8_post_fix_evaluation.json"
PHASE5_LIVE_REPORT_PATH = REPORTS_DIRECTORY / "phase5_live_evaluation.json"
LIVE_ANSWER_REPORT_PATH = REPORTS_DIRECTORY / "hybrid_live_nemotron_evaluation.json"


class EvaluationReportUnavailableError(RuntimeError):
    """Raised when a saved evaluation report cannot be safely loaded."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationReportUnavailableError("A saved evaluation report is unavailable.") from exc
    if not isinstance(value, dict):
        raise EvaluationReportUnavailableError("A saved evaluation report is invalid.")
    return value


def load_evaluation_dashboard_report(
    retrieval_path: Path = DEFAULT_REPORT_PATH,
    answer_path: Path = DEFAULT_ANSWER_REPORT_PATH,
    optimized_c8_path: Path = OPTIMIZED_C8_REPORT_PATH,
    structured_c8_path: Path = STRUCTURED_C8_REPORT_PATH,
    phase5_live_path: Path = PHASE5_LIVE_REPORT_PATH,
    live_answer_path: Path = LIVE_ANSWER_REPORT_PATH,
) -> dict[str, Any]:
    """Return a compact UI-safe view of the verified saved evaluation reports."""
    retrieval = _load_json(retrieval_path)
    answer = _load_json(answer_path)
    optimized_c8 = _load_json(optimized_c8_path)
    structured_c8 = _load_json(structured_c8_path)
    phase5_live = _load_json(phase5_live_path)
    live_answer = _load_json(live_answer_path)

    structured_results = structured_c8.get("results", [])
    structured_categories = ("numerical", "filtering", "ranking", "mixed")
    category_counts = {
        category: sum(
            1 for item in structured_results if item.get("type") == category
        )
        for category in structured_categories
    }

    phase5_queries = phase5_live.get("queries", [])
    citation_queries = [
        item
        for item in phase5_queries
        if item.get("citations", {}).get("status") == "valid"
    ]
    live_scores = live_answer.get("measured_scores", {})

    return {
        "retrieval": {
            "status": retrieval.get("status"),
            "result_kind": "actual_chromadb",
            "generated_at_utc": retrieval.get("generated_at_utc"),
            "semantic_questions_evaluated": retrieval.get("benchmark", {}).get(
                "semantic_questions_evaluated"
            ),
            "aggregate_metrics": retrieval.get("aggregate_metrics", {}),
            "mrr_cutoff": retrieval.get("retrieval", {}).get("mrr_cutoff"),
            "failure_count": retrieval.get("failure_count"),
            "questions": [
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "relevant_source_rows": item.get("relevant_source_rows", []),
                    "retrieved_source_rows": item.get("retrieved_source_rows", []),
                    "metrics": item.get("metrics", {}),
                    "failure": item.get("failure"),
                }
                for item in retrieval.get("questions", [])
            ],
        },
        "optimized_c8_retrieval": {
            "status": optimized_c8.get("assessment"),
            "result_kind": "actual_chromadb_comparison",
            "generated_at_utc": optimized_c8.get("generated_at_utc"),
            "semantic_questions_evaluated": optimized_c8.get("benchmark", {}).get(
                "query_count"
            ),
            "metrics": optimized_c8.get("metrics", {}),
            "citations_valid": optimized_c8.get("citation_source_rows", {}).get(
                "all_valid"
            ),
        },
        "structured_evaluation": {
            "status": "verified_post_fix",
            "result_kind": "actual_deterministic",
            "generated_at_utc": structured_c8.get("generated_at_utc"),
            "questions_evaluated": sum(category_counts.values()),
            "question_count_by_category": category_counts,
            "llm_calls": structured_c8.get("llm_calls"),
            "metrics": {
                key: structured_c8.get("metrics", {}).get(f"{key}_accuracy")
                for key in structured_categories
            },
        },
        "phase5_live": {
            "status": phase5_live.get("assessment"),
            "result_kind": "actual_live_pipeline",
            "generated_at_utc": phase5_live.get("generated_at_utc"),
            "queries_attempted": phase5_live.get("execution", {}).get(
                "queries_attempted"
            ),
            "queries_completed": phase5_live.get("execution", {}).get(
                "queries_completed"
            ),
            "correct_queries": sum(
                1 for item in phase5_queries if item.get("correct") is True
            ),
            "citation_checks": len(citation_queries),
            "valid_citation_checks": sum(
                1
                for item in citation_queries
                if item.get("citations", {}).get("valid") is True
            ),
            "llm_invocation_count": phase5_live.get("execution", {}).get(
                "llm_invocation_count"
            ),
            "provider": phase5_live.get("execution", {}).get("provider"),
            "model": phase5_live.get("execution", {}).get("model"),
            "provider_error_count": len(
                phase5_live.get("execution", {}).get("provider_errors", [])
            ),
            "latency": phase5_live.get("latency", {}),
            "queries": [
                {
                    "category": item.get("category"),
                    "question": item.get("question"),
                    "mode": item.get("routing", {}).get("mode"),
                    "llm_invoked": item.get("routing", {}).get("llm_invoked"),
                    "correct": item.get("correct"),
                    "grounded": item.get("grounded"),
                    "citation_status": item.get("citations", {}).get("status"),
                    "citations_valid": item.get("citations", {}).get("valid"),
                    "latency_ms": item.get("latency_ms"),
                    "provider": item.get("provider"),
                    "model": item.get("model"),
                    "error": item.get("error"),
                }
                for item in phase5_queries
            ],
        },
        "answer_quality": {
            "verified_live_measurement": {
                "status": live_answer.get("status"),
                "result_kind": live_answer.get("result_kind"),
                "generated_at_utc": live_answer.get("generated_at_utc"),
                "model": live_answer.get("model"),
                "answers_evaluated": live_scores.get("answers_evaluated"),
                "faithfulness": live_scores.get("faithfulness"),
                "answer_relevance": live_scores.get("answer_relevance"),
                "unsupported_claim_rate": live_scores.get(
                    "unsupported_claim_rate"
                ),
                "provider_failure_count": live_answer.get(
                    "provider_failure_count"
                ),
                "score_provenance": live_answer.get("score_provenance"),
            },
            "live_llm_evaluation": answer.get("live_llm_evaluation", {}),
            "fixture_validation": answer.get("fixture_validation", {}),
            "numerical_correctness": answer.get("numerical_correctness", {}),
        },
    }
