import json
from uuid import uuid4

import pandas as pd
import pytest

from backend.dataset_selection import SelectedDataset
from backend.rag_pipeline import answer_question
from backend.rag_retrieval import RetrievalMatch


@pytest.fixture(scope="module")
def sample_context():
    benchmark = json.load(open("evaluation/retailiq_12row_benchmark.json", encoding="utf-8"))
    dataframe = pd.read_csv("evaluation/data/retailiq_sample_12_rows.csv")
    dataframe.index = pd.RangeIndex(1, len(dataframe) + 1)
    dataset = SelectedDataset(uuid4(), "retailiq_sample_12_rows.csv", dataframe)
    return benchmark, dataset


def _match(dataset, row, rank):
    return RetrievalMatch(
        rank,
        f"chunk-{row}",
        f"Source row: {row}",
        row,
        {"upload_id": str(dataset.upload_id), "row_number": row},
        1.0 - rank / 100,
    )


class GroundedStub:
    def generate_grounded_answer(self, _question, matches):
        return "Grounded in rows " + ", ".join(str(match.source_row) for match in matches)


def _answer(dataset, question, relevant_rows):
    return answer_question(
        dataset.upload_id,
        question,
        metadata_lookup=lambda _id: {"upload_id": str(_id)},
        dataset_loader=lambda _id: dataset,
        retriever=lambda _id, _question, _top_k: [
            _match(dataset, row, rank)
            for rank, row in enumerate(relevant_rows, 1)
        ],
        llm_service=GroundedStub(),
    )


def _calculation_rows(result):
    rows = set()
    for record in result.calculation.get("records", []):
        if "source_row" in record:
            rows.add(int(record["source_row"]))
        rows.update(int(row) for row in record.get("source_rows", []))
    return rows


def test_original_12_row_semantic_and_filter_questions_preserve_grounded_rows(sample_context):
    benchmark, dataset = sample_context
    semantic = [entry for entry in benchmark["questions"] if entry["type"] == "semantic"]

    for entry in semantic:
        result = _answer(dataset, entry["question"], entry["relevant_source_rows"])
        if result.mode == "analytics":
            assert _calculation_rows(result) == set(entry["relevant_source_rows"]), entry["id"]
        else:
            assert result.mode == "semantic_rag", entry["id"]
            assert {source.source_row for source in result.sources} == set(
                entry["relevant_source_rows"]
            )


def test_original_12_row_numerical_questions_remain_exact(sample_context):
    benchmark, dataset = sample_context
    expected = {
        "NUM-001": 45150.0,
        "NUM-002": 5750.0,
        "NUM-003": 11,
        "NUM-004": 45150 / 11,
        "NUM-005": 5750 / 45150 * 100,
    }

    for entry in benchmark["questions"]:
        if entry["id"] not in expected:
            continue
        result = _answer(dataset, entry["question"], [])
        assert result.mode == "analytics"
        assert result.calculation["value"] == pytest.approx(expected[entry["id"]])


def test_ambiguous_and_unsupported_queries_do_not_call_llm_or_retrieval(sample_context):
    _benchmark, dataset = sample_context

    result = answer_question(
        dataset.upload_id,
        "What are the top products?",
        metadata_lookup=lambda _id: {"upload_id": str(_id)},
        dataset_loader=lambda _id: dataset,
        retriever=lambda *_args: pytest.fail("retrieval must not run"),
        filtered_retriever=lambda *_args: pytest.fail("filtered retrieval must not run"),
        llm_service=pytest.fail,
    )

    assert result.mode == "unsupported"
    assert "ambiguous" in result.answer.lower() or "unsupported" in result.answer.lower()
