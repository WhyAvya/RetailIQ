import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_PATH = EVALUATION_ROOT / "retailiq_12row_benchmark.json"
DEFAULT_SOURCE_PATH = EVALUATION_ROOT / "data" / "retailiq_sample_12_rows.csv"
ALLOWED_TYPES = {"semantic", "numerical"}
EXPECTED_COUNTS = {"semantic": 15, "numerical": 5}
REQUIRED_FIELDS = {
    "id",
    "question",
    "type",
    "expected_answer",
    "relevant_source_rows",
    "evaluation_notes",
}


class BenchmarkValidationError(ValueError):
    """Raised when the evaluation benchmark is structurally invalid."""


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_benchmark(
    benchmark_path: Path | str = DEFAULT_BENCHMARK_PATH,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
) -> dict[str, Any]:
    """Validate benchmark shape, labels, IDs, source rows, and source identity."""
    benchmark_path = Path(benchmark_path)
    source_path = Path(source_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    source_rows = _read_csv_rows(source_path)
    questions = benchmark.get("questions")
    errors: list[str] = []

    if not isinstance(questions, list):
        raise BenchmarkValidationError("questions must be a list")
    if len(questions) != 20:
        errors.append(f"expected 20 questions, found {len(questions)}")

    ids: set[str] = set()
    question_texts: set[str] = set()
    counts = {label: 0 for label in ALLOWED_TYPES}
    for position, entry in enumerate(questions, start=1):
        if not isinstance(entry, dict):
            errors.append(f"entry {position} must be an object")
            continue
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"entry {position} is missing fields: {sorted(missing)}")
            continue

        question_id = entry["id"]
        if not isinstance(question_id, str) or not question_id.strip():
            errors.append(f"entry {position} has an invalid ID")
        elif question_id in ids:
            errors.append(f"duplicate ID: {question_id}")
        else:
            ids.add(question_id)

        question_type = entry["type"]
        if question_type not in ALLOWED_TYPES:
            errors.append(f"{question_id} has an incorrect type label: {question_type}")
        else:
            counts[question_type] += 1
            expected_prefix = "SEM" if question_type == "semantic" else "NUM"
            if not re.fullmatch(rf"{expected_prefix}-\d{{3}}", question_id):
                errors.append(f"{question_id} does not match its type label")

        question = entry["question"]
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{question_id} has an empty question")
        elif question in question_texts:
            errors.append(f"duplicate question text: {question}")
        else:
            question_texts.add(question)

        if not isinstance(entry["expected_answer"], str) or not entry["expected_answer"].strip():
            errors.append(f"{question_id} has an empty expected answer")
        if not isinstance(entry["evaluation_notes"], str) or not entry["evaluation_notes"].strip():
            errors.append(f"{question_id} has empty evaluation notes")

        relevant_rows = entry["relevant_source_rows"]
        if not isinstance(relevant_rows, list):
            errors.append(f"{question_id} relevant_source_rows must be a list")
            continue
        if any(isinstance(row, bool) or not isinstance(row, int) for row in relevant_rows):
            errors.append(f"{question_id} contains a non-integer source row")
            continue
        if len(relevant_rows) != len(set(relevant_rows)):
            errors.append(f"{question_id} contains duplicate source rows")
        invalid_rows = [row for row in relevant_rows if row < 1 or row > len(source_rows)]
        if invalid_rows:
            errors.append(f"{question_id} contains invalid source rows: {invalid_rows}")
        if question_type == "semantic" and not relevant_rows:
            errors.append(f"{question_id} must identify at least one relevant source row")
        if question_type == "numerical" and relevant_rows:
            errors.append(f"{question_id} numerical questions must not list source rows")

    if counts != EXPECTED_COUNTS:
        errors.append(f"incorrect question type counts: {counts}")

    source = benchmark.get("source", {})
    if source.get("row_count") != len(source_rows):
        errors.append("source row count does not match the CSV")
    if source.get("sha256") != _sha256(source_path):
        errors.append("source SHA-256 does not match the CSV")
    if source.get("row_numbering") != "1-based CSV data rows; header excluded":
        errors.append("source row-numbering convention is incorrect")

    if errors:
        raise BenchmarkValidationError("; ".join(errors))
    return benchmark
