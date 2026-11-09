import copy
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from evaluation.benchmark_validation import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_SOURCE_PATH,
    BenchmarkValidationError,
    validate_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_UPLOAD = (
    PROJECT_ROOT
    / "backend"
    / "storage"
    / "uploads"
    / "6377aeff-8a64-47ac-9748-28e0b0523d56.csv"
)


def _questions_by_id(benchmark):
    return {entry["id"]: entry for entry in benchmark["questions"]}


def _write_changed_benchmark(tmp_path, benchmark):
    path = tmp_path / "changed-benchmark.json"
    path.write_text(json.dumps(benchmark), encoding="utf-8")
    return path


def test_benchmark_has_exact_required_counts_and_matches_original_sample():
    benchmark = validate_benchmark()

    assert len(benchmark["questions"]) == 20
    assert sum(item["type"] == "semantic" for item in benchmark["questions"]) == 15
    assert sum(item["type"] == "numerical" for item in benchmark["questions"]) == 5
    assert sha256(DEFAULT_SOURCE_PATH.read_bytes()).hexdigest() == benchmark["source"]["sha256"]

    # Local uploads are intentionally Git-ignored, so this provenance check runs
    # whenever the retained original is available without making CI depend on it.
    if ORIGINAL_UPLOAD.exists():
        original_hash = sha256(ORIGINAL_UPLOAD.read_bytes()).hexdigest()
        assert sha256(DEFAULT_SOURCE_PATH.read_bytes()).hexdigest() == original_hash
        pd.testing.assert_frame_equal(
            pd.read_csv(DEFAULT_SOURCE_PATH),
            pd.read_csv(ORIGINAL_UPLOAD),
        )
        assert sha256(ORIGINAL_UPLOAD.read_bytes()).hexdigest() == original_hash


def test_all_semantic_source_rows_are_derived_directly_from_csv():
    questions = _questions_by_id(validate_benchmark())
    frame = pd.read_csv(DEFAULT_SOURCE_PATH)
    frame.index = frame.index + 1

    expected_rows = {
        "SEM-001": frame.index[(frame["Category"] == "Technology") & (frame["Sub-Category"] == "Accessories") & (frame["Profit"] >= 300)].tolist(),
        "SEM-002": frame.index[(frame["Category"] == "Furniture") & (frame["Profit"] < 0)].tolist(),
        "SEM-003": frame.index[frame["Region"] == "West"].tolist(),
        "SEM-004": frame.index[frame["Customer Name"] == "Aisha Sharma"].tolist(),
        "SEM-005": frame.index[(frame["Region"] == "South") & (frame["Category"] == "Technology") & (frame["Sub-Category"] == "Phones")].tolist(),
        "SEM-006": frame.index[(frame["Category"] == "Office Supplies") & (frame["Sales"] < 700)].tolist(),
        "SEM-007": frame.index[frame["Profit"] >= 800].tolist(),
        "SEM-008": frame.index[frame["Region"] == "East"].tolist(),
        "SEM-009": frame.index[frame["Quantity"] >= 5].tolist(),
        "SEM-010": frame.index[pd.to_datetime(frame["Order Date"]).dt.to_period("M") == pd.Period("2026-05")].tolist(),
        "SEM-011": frame.index[frame["Customer ID"].duplicated(keep=False)].tolist(),
        "SEM-012": frame.index[frame["Sub-Category"] == "Chairs"].tolist(),
        "SEM-013": frame.index[frame["Sales"].between(800, 1500, inclusive="both")].tolist(),
        "SEM-014": frame.index[frame["Profit"].between(100, 200, inclusive="both")].tolist(),
        "SEM-015": frame.index[frame["Region"] == "North"].tolist(),
    }

    assert set(expected_rows) == {f"SEM-{number:03d}" for number in range(1, 16)}
    for question_id, source_rows in expected_rows.items():
        assert questions[question_id]["relevant_source_rows"] == source_rows


def test_semantic_answers_name_the_verified_csv_facts():
    questions = _questions_by_id(validate_benchmark())
    frame = pd.read_csv(DEFAULT_SOURCE_PATH)
    frame.index = frame.index + 1
    answer_columns = {
        "SEM-001": ("Product Name", "Profit"),
        "SEM-002": ("Product Name", "Profit"),
        "SEM-003": ("Product Name", "Category"),
        "SEM-004": ("Product Name",),
        "SEM-005": ("Product Name",),
        "SEM-006": ("Product Name", "Sales"),
        "SEM-007": ("Product Name", "Profit"),
        "SEM-008": ("Product Name", "Profit"),
        "SEM-009": ("Product Name", "Quantity"),
        "SEM-010": ("Product Name", "Order Date"),
        "SEM-011": ("Customer Name",),
        "SEM-012": ("Product Name", "Profit"),
        "SEM-013": ("Product Name", "Sales"),
        "SEM-014": ("Product Name", "Profit"),
        "SEM-015": ("Product Name", "Category"),
    }

    for question_id, columns in answer_columns.items():
        entry = questions[question_id]
        answer = entry["expected_answer"]
        for row_number in entry["relevant_source_rows"]:
            for column in columns:
                assert str(frame.loc[row_number, column]) in answer


def test_numerical_answers_match_independent_dataframe_calculations():
    questions = _questions_by_id(validate_benchmark())
    frame = pd.read_csv(DEFAULT_SOURCE_PATH)
    revenue = frame["Sales"].sum()
    profit = frame["Profit"].sum()
    orders = frame["Order ID"].nunique()

    assert questions["NUM-001"]["expected_answer"] == f"Total revenue is {revenue:,} sales units."
    assert questions["NUM-002"]["expected_answer"] == f"Total profit is {profit:,} sales units."
    assert questions["NUM-003"]["expected_answer"] == f"There are {orders} distinct orders."
    assert questions["NUM-004"]["expected_answer"] == f"Average order value is approximately {revenue / orders:,.2f} sales units per distinct order."
    assert questions["NUM-005"]["expected_answer"] == f"Overall profit margin is approximately {profit / revenue * 100:.2f}%."


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["questions"].pop(), "expected 20 questions"),
        (lambda data: data["questions"][1].update(id=data["questions"][0]["id"]), "duplicate ID"),
        (lambda data: data["questions"][0].update(relevant_source_rows=[13]), "invalid source rows"),
        (lambda data: data["questions"][0].update(type="retrieval"), "incorrect type label"),
    ],
)
def test_validation_rejects_invalid_benchmarks(tmp_path, change, message):
    benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(benchmark)
    change(changed)

    with pytest.raises(BenchmarkValidationError, match=message):
        validate_benchmark(_write_changed_benchmark(tmp_path, changed))
