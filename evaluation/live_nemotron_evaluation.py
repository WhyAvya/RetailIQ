"""Limited real Nemotron evaluation for the frozen C6 hybrid system."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from backend.chroma_metadata import FULL_DATASET_UPLOAD_ID, load_metadata_source
from backend.dataset_selection import SelectedDataset
from backend.llm_service import LLMServiceError, OpenRouterLLMService
from backend.rag_pipeline import answer_question
from backend.rag_retrieval import RetrievalMatch, retrieve_semantic_matches
from evaluation.answer_evaluation import evaluate_annotations


REPORT_ROOT = Path(__file__).resolve().parent / "reports"
DEFAULT_RAW_PATH = REPORT_ROOT / "hybrid_live_nemotron_raw.json"
DEFAULT_SCORED_PATH = REPORT_ROOT / "hybrid_live_nemotron_evaluation.json"
DEFAULT_MARKDOWN_PATH = REPORT_ROOT / "hybrid_live_nemotron_evaluation.md"
QUESTIONS = (
    ("LIVE-001", "Find products suitable for organizing a small workspace."),
    ("LIVE-002", "Which product names mention recycled materials?"),
    ("LIVE-003", "What products could help with presentations or meetings?"),
)


def run_live_sample(
    llm_service: OpenRouterLLMService | Any | None = None,
    retriever: Callable[[str, str, int], list[RetrievalMatch]] = retrieve_semantic_matches,
) -> dict[str, Any]:
    frame = load_metadata_source()
    dataset = SelectedDataset(UUID(FULL_DATASET_UPLOAD_ID), "sample_superstore.csv", frame)
    service = llm_service or OpenRouterLLMService()
    results = []
    for question_id, question in QUESTIONS:
        try:
            answer = answer_question(
                dataset.upload_id,
                question,
                top_k=5,
                metadata_lookup=lambda _id: {"upload_id": str(_id)},
                dataset_loader=lambda _id: dataset,
                retriever=retriever,
                llm_service=service,
            )
            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "status": "completed",
                    "mode": answer.mode,
                    "answer": answer.answer,
                    "evidence": [
                        {
                            "source_row": source.source_row,
                            "chunk_id": source.chunk_id,
                            "similarity_score": source.similarity_score,
                            "metadata": source.metadata,
                            "document_excerpt": source.document_excerpt,
                        }
                        for source in answer.sources
                    ],
                    "provider_failure": None,
                }
            )
        except LLMServiceError as error:
            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "status": "provider_failure",
                    "mode": "semantic_rag",
                    "answer": None,
                    "evidence": [],
                    "provider_failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
    completed = sum(item["status"] == "completed" for item in results)
    return {
        "evaluation_id": "retailiq-c6-live-nemotron-sample-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_kind": "real_live_llm",
        "model": service.config.model if hasattr(service, "config") else "test-double",
        "request_limit": len(QUESTIONS),
        "requests_attempted": len(results),
        "answers_completed": completed,
        "provider_failure_count": len(results) - completed,
        "questions": results,
    }


def score_live_sample(
    raw_report: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    completed = {
        item["id"]: item
        for item in raw_report["questions"]
        if item["status"] == "completed"
    }
    for annotation in annotations:
        raw = completed.get(annotation.get("question_id"))
        if raw is None:
            raise ValueError("Annotations may only score completed live answers.")
        if annotation.get("question") != raw["question"] or annotation.get("answer") != raw["answer"]:
            raise ValueError("Annotation question and answer must exactly match live output.")
        evidence_rows = {item["source_row"] for item in raw["evidence"]}
        if not set(annotation.get("retrieved_evidence_rows", [])).issubset(evidence_rows):
            raise ValueError("Annotation evidence rows must come from live retrieval.")
    scores = evaluate_annotations(annotations) if annotations else None
    if scores is not None:
        scores["grounding"] = scores["faithfulness"]
    status = (
        "completed"
        if raw_report["provider_failure_count"] == 0 and len(annotations) == raw_report["answers_completed"]
        else "completed_with_provider_failures"
        if raw_report["provider_failure_count"]
        else "partially_scored"
    )
    return {
        **raw_report,
        "status": status,
        "measured_scores": scores,
        "unscored_completed_answers": raw_report["answers_completed"] - len(annotations),
        "score_provenance": "Manual claim/evidence review of real live answers; fixture scores are not included.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    scores = report.get("measured_scores")
    lines = [
        "# RetailIQ C6 Live Nemotron Evaluation",
        "",
        f"Model: `{report['model']}`. Attempted: {report['requests_attempted']}. Completed: {report['answers_completed']}. Provider failures: {report['provider_failure_count']}.",
        "",
    ]
    if scores:
        lines.extend(
            [
                "## Measured real-answer scores",
                "",
                f"- Answers scored: {scores['answers_evaluated']}",
                f"- Faithfulness: {scores['faithfulness']:.4f}",
                f"- Grounding: {scores['grounding']:.4f}",
                f"- Unsupported claim rate: {scores['unsupported_claim_rate']:.4f}",
                f"- Answer relevance: {scores['answer_relevance']:.4f}",
                "",
            ]
        )
    else:
        lines.extend(["No live answer scores were available; no scores were invented.", ""])
    lines.extend(["## Provider outcomes", ""])
    for item in report["questions"]:
        outcome = item["status"] if not item["provider_failure"] else f"{item['status']} ({item['provider_failure']['type']})"
        lines.append(f"- {item['id']}: {outcome}")
    lines.extend(
        [
            "",
            "Scores above are from real live model outputs. Historical fixture-only answer-quality scores are intentionally excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def save_json(report: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def save_scored(report: dict[str, Any]) -> tuple[Path, Path]:
    save_json(report, DEFAULT_SCORED_PATH)
    DEFAULT_MARKDOWN_PATH.write_text(render_markdown(report), encoding="utf-8")
    return DEFAULT_SCORED_PATH, DEFAULT_MARKDOWN_PATH


if __name__ == "__main__":
    report = run_live_sample()
    output = save_json(report, DEFAULT_RAW_PATH)
    print(f"Saved raw live results to {output}")
    print(json.dumps({key: report[key] for key in ("model", "requests_attempted", "answers_completed", "provider_failure_count")}, indent=2))
