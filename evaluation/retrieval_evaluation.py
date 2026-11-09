import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.embeddings import EMBEDDING_MODEL_NAME
from backend.rag_retrieval import RetrievalError, RetrievalMatch, retrieve_semantic_matches
from backend.vector_store import COLLECTION_NAME
from evaluation.benchmark_validation import (
    DEFAULT_BENCHMARK_PATH,
    validate_benchmark,
)


DEFAULT_K_VALUES = (3, 5)
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "reports" / "retrieval_evaluation.json"
Retriever = Callable[[str, str, int], list[RetrievalMatch]]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("K must be a positive integer.")


def _unique_ranked_rows(retrieved_rows: Iterable[int | None]) -> list[int]:
    rows: list[int] = []
    seen: set[int] = set()
    for row in retrieved_rows:
        if isinstance(row, bool) or not isinstance(row, int) or row in seen:
            continue
        rows.append(row)
        seen.add(row)
    return rows


def precision_at_k(
    retrieved_rows: Sequence[int | None],
    relevant_rows: Sequence[int],
    k: int,
) -> float:
    """Return relevant unique rows in the first K ranks divided by fixed K."""
    _validate_k(k)
    ranked = _unique_ranked_rows(retrieved_rows)[:k]
    relevant = set(relevant_rows)
    return sum(row in relevant for row in ranked) / k


def recall_at_k(
    retrieved_rows: Sequence[int | None],
    relevant_rows: Sequence[int],
    k: int,
) -> float:
    """Return relevant unique rows in the first K ranks divided by all relevant rows."""
    _validate_k(k)
    relevant = set(relevant_rows)
    if not relevant:
        return 0.0
    ranked = _unique_ranked_rows(retrieved_rows)[:k]
    return sum(row in relevant for row in ranked) / len(relevant)


def reciprocal_rank(
    retrieved_rows: Sequence[int | None],
    relevant_rows: Sequence[int],
) -> float:
    """Return the reciprocal rank of the first relevant unique source row."""
    relevant = set(relevant_rows)
    if not relevant:
        return 0.0
    for rank, row in enumerate(_unique_ranked_rows(retrieved_rows), start=1):
        if row in relevant:
            return 1.0 / rank
    return 0.0


def calculate_question_metrics(
    retrieved_rows: Sequence[int | None],
    relevant_rows: Sequence[int],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    values = sorted(set(k_values))
    if not values:
        raise ValueError("At least one K value is required.")
    for k in values:
        _validate_k(k)
    metrics: dict[str, float] = {}
    for k in values:
        metrics[f"precision_at_{k}"] = precision_at_k(retrieved_rows, relevant_rows, k)
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_rows, relevant_rows, k)
    metrics["reciprocal_rank"] = reciprocal_rank(retrieved_rows, relevant_rows)
    return metrics


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_failure(error: RetrievalError) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def evaluate_retrieval(
    benchmark_path: Path | str = DEFAULT_BENCHMARK_PATH,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    retriever: Retriever = retrieve_semantic_matches,
) -> dict[str, Any]:
    """Evaluate only semantic benchmark questions using the existing retriever."""
    benchmark_path = Path(benchmark_path)
    benchmark = validate_benchmark(benchmark_path)
    semantic_questions = [
        entry for entry in benchmark["questions"] if entry["type"] == "semantic"
    ]
    values = sorted(set(k_values))
    if not values:
        raise ValueError("At least one K value is required.")
    for k in values:
        _validate_k(k)
    max_k = max(values)
    upload_id = benchmark["source"]["original_upload_id"]
    results: list[dict[str, Any]] = []

    for entry in semantic_questions:
        failure = None
        matches: list[RetrievalMatch] = []
        try:
            matches = retriever(upload_id, entry["question"], max_k)
            if any(match.metadata.get("upload_id") != upload_id for match in matches):
                raise RetrievalError("Retrieval returned rows from another dataset.")
        except RetrievalError as error:
            failure = _safe_failure(error)
            matches = []

        retrieved_rows = _unique_ranked_rows(match.source_row for match in matches)
        metrics = calculate_question_metrics(
            retrieved_rows,
            entry["relevant_source_rows"],
            values,
        )
        results.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "relevant_source_rows": entry["relevant_source_rows"],
                "retrieved_source_rows": retrieved_rows,
                "matches": [
                    {
                        "rank": match.rank,
                        "source_row": match.source_row,
                        "similarity_score": match.similarity_score,
                        "chunk_id": match.chunk_id,
                    }
                    for match in matches
                ],
                "metrics": metrics,
                "failure": failure,
            }
        )

    aggregate: dict[str, float] = {}
    for k in values:
        aggregate[f"precision_at_{k}"] = _mean(
            [result["metrics"][f"precision_at_{k}"] for result in results]
        )
        aggregate[f"recall_at_{k}"] = _mean(
            [result["metrics"][f"recall_at_{k}"] for result in results]
        )
    aggregate["mrr"] = _mean(
        [result["metrics"]["reciprocal_rank"] for result in results]
    )
    failure_count = sum(result["failure"] is not None for result in results)

    return {
        "evaluation_id": "retailiq-phase5-round2-retrieval-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if failure_count == 0 else "completed_with_failures",
        "benchmark": {
            "id": benchmark["benchmark_id"],
            "sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            "total_questions": len(benchmark["questions"]),
            "semantic_questions_evaluated": len(semantic_questions),
            "numerical_questions_excluded": len(benchmark["questions"]) - len(semantic_questions),
        },
        "retrieval": {
            "upload_id": upload_id,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "vector_collection": COLLECTION_NAME,
            "k_values": values,
            "mrr_cutoff": max_k,
            "source_row_numbering": benchmark["source"]["row_numbering"],
            "precision_denominator": "fixed K; fewer than K returned rows are penalized",
            "duplicate_row_policy": "keep the first occurrence of each source row",
            "failed_questions_counted_as_zero": True,
        },
        "aggregate_metrics": aggregate,
        "failure_count": failure_count,
        "questions": results,
    }


def save_report(report: dict[str, Any], output_path: Path | str = DEFAULT_REPORT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RetailIQ semantic retrieval.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    arguments = parser.parse_args()
    report = evaluate_retrieval(arguments.benchmark, arguments.k_values)
    output = save_report(report, arguments.output)
    print(f"Saved retrieval evaluation to {output}")
    print(json.dumps(report["aggregate_metrics"], indent=2))
    if report["failure_count"]:
        print(f"Retrieval failures: {report['failure_count']}")


if __name__ == "__main__":
    main()
