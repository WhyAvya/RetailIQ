import json
from pathlib import Path
from typing import Any

from evaluation.answer_evaluation import DEFAULT_ANSWER_REPORT_PATH
from evaluation.retrieval_evaluation import DEFAULT_REPORT_PATH


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
) -> dict[str, Any]:
    """Return a UI-safe view of saved actual, fixture, and pending results."""
    retrieval = _load_json(retrieval_path)
    answer = _load_json(answer_path)
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
        "answer_quality": answer,
    }
