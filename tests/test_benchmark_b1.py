import copy
import json
from collections import Counter
from hashlib import sha256

import pytest

from evaluation.benchmark_b1 import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_COVERAGE_REPORT_PATH,
    DEFAULT_SOURCE_PATH,
    EXPECTED_ROW_COUNT,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TYPE_COUNTS,
    FULL_DATASET_UPLOAD_ID,
    BenchmarkB1ValidationError,
    audit_index_coverage,
    expected_chunks,
    load_source_dataframe,
    validate_benchmark,
)


SOURCE_SHA256 = "58fb59c63089e564d835e4aabe19d371757de1137a5bf1888243ae386c4b7a97"
EXPECTED_SPLIT_TYPE_COUNTS = {
    "development": {
        "semantic_product_discovery": 13,
        "metadata_filtering": 10,
        "numerical": 10,
        "mixed": 7,
    },
    "held_out_test": {
        "semantic_product_discovery": 7,
        "metadata_filtering": 5,
        "numerical": 5,
        "mixed": 3,
    },
}


@pytest.fixture(scope="module")
def benchmark():
    return json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    return expected_chunks()


def test_full_source_identity_and_dataset_coverage(chunks):
    frame = load_source_dataframe()

    assert len(frame) == EXPECTED_ROW_COUNT
    assert sha256(DEFAULT_SOURCE_PATH.read_bytes()).hexdigest() == SOURCE_SHA256
    assert len(chunks) == EXPECTED_ROW_COUNT
    assert len({chunk.chunk_id for chunk in chunks}) == EXPECTED_ROW_COUNT
    assert {chunk.metadata["row_number"] for chunk in chunks} == set(
        range(1, EXPECTED_ROW_COUNT + 1)
    )
    assert all(
        chunk.metadata["upload_id"] == FULL_DATASET_UPLOAD_ID for chunk in chunks
    )


def test_recorded_chroma_audit_has_every_row_once_and_no_conflicts():
    report = json.loads(DEFAULT_COVERAGE_REPORT_PATH.read_text(encoding="utf-8"))
    coverage = report["coverage_after"]

    assert report["source_sha256"] == SOURCE_SHA256
    assert report["upload_id"] == FULL_DATASET_UPLOAD_ID
    assert report["indexing_action"] == "none"
    assert report["rows_indexed_during_run"] == 0
    assert coverage["complete"] is True
    assert coverage["expected_rows"] == EXPECTED_ROW_COUNT
    assert coverage["stored_chunks"] == EXPECTED_ROW_COUNT
    assert coverage["unique_stored_ids"] == EXPECTED_ROW_COUNT
    assert coverage["unique_stored_rows"] == EXPECTED_ROW_COUNT
    assert coverage["missing_rows"] == []
    assert coverage["duplicate_rows"] == []
    assert coverage["unexpected_chunk_ids"] == []
    assert coverage["row_id_conflicts"] == []


def test_coverage_audit_detects_missing_and_duplicate_rows(chunks):
    first, second = chunks[:2]
    stored = {
        "ids": [first.chunk_id, second.chunk_id, "duplicate-row-one"],
        "metadatas": [
            first.metadata,
            second.metadata,
            {**first.metadata, "document_id": "different"},
        ],
    }

    coverage = audit_index_coverage(stored, chunks)

    assert coverage["complete"] is False
    assert coverage["duplicate_rows"] == [1]
    assert coverage["missing_rows"][0] == 3
    assert len(coverage["missing_rows"]) == EXPECTED_ROW_COUNT - 2
    assert coverage["unexpected_chunk_ids"] == ["duplicate-row-one"]


def test_benchmark_counts_required_fields_and_split_integrity(benchmark):
    questions = benchmark["questions"]
    ids = [entry["id"] for entry in questions]

    assert len(questions) == 60
    assert len(ids) == len(set(ids))
    assert Counter(entry["type"] for entry in questions) == EXPECTED_TYPE_COUNTS
    assert Counter(entry["split"] for entry in questions) == EXPECTED_SPLIT_COUNTS
    for split, expected in EXPECTED_SPLIT_TYPE_COUNTS.items():
        assert Counter(
            entry["type"] for entry in questions if entry["split"] == split
        ) == expected
    development_ids = {
        entry["id"] for entry in questions if entry["split"] == "development"
    }
    held_out_ids = {
        entry["id"] for entry in questions if entry["split"] == "held_out_test"
    }
    assert development_ids.isdisjoint(held_out_ids)
    assert development_ids | held_out_ids == set(ids)


def test_all_ground_truth_and_relevance_are_recomputed_from_original_csv(benchmark):
    # validate_benchmark rebuilds every expected answer, structured result, and
    # relevance row set from data/sample_superstore.csv with Pandas.
    assert validate_benchmark(benchmark) == benchmark


def test_numerical_questions_declare_full_dataset_scope(benchmark):
    numerical = [
        entry for entry in benchmark["questions"] if entry["type"] == "numerical"
    ]

    assert all(entry["relevant_source_rows"] == [] for entry in numerical)
    assert all("all 9,994 CSV data rows" in entry["relevance_definition"] for entry in numerical)


def test_changed_ground_truth_is_rejected(benchmark):
    changed = copy.deepcopy(benchmark)
    changed["questions"][0]["ground_truth"]["records"][0]["Profit"] = 0

    with pytest.raises(
        BenchmarkB1ValidationError,
        match="independently recomputed Pandas ground truth",
    ):
        validate_benchmark(changed)
