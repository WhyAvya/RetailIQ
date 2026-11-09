from uuid import uuid4

import pandas as pd
import pytest

from backend.analytics import calculate_kpis
from backend.dataset_selection import SelectedDataset
from backend.rag_pipeline import (
    PipelineDatasetNotFoundError,
    RAGPipelineError,
    answer_question,
    route_question,
)
from backend.rag_retrieval import RetrievalMatch


@pytest.fixture
def retail_dataset():
    upload_id = uuid4()
    dataframe = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B"],
            "Customer ID": ["C1", "C1", "C2"],
            "Region": ["West", "West", "East"],
            "Category": ["Technology", "Furniture", "Furniture"],
            "Sales": [100.0, 50.0, 200.0],
            "Profit": [20.0, -10.0, 30.0],
        }
    )
    return SelectedDataset(upload_id, "sales.csv", dataframe)


def pipeline(dataset, question, **kwargs):
    return answer_question(
        dataset.upload_id,
        question,
        metadata_lookup=lambda _id: {"upload_id": str(_id)},
        dataset_loader=lambda _id: dataset,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What is the total revenue?", "total_revenue"),
        ("What is the total profit?", "total_profit"),
        ("How many unique orders are there?", "number_of_orders"),
        ("How many unique customers are there?", "unique_customers"),
        ("What is the average order value?", "average_order_value"),
        ("What is the profit margin?", "profit_margin"),
        ("Which region generated the highest revenue?", "highest_revenue_region"),
        ("Which category has the lowest profit?", "lowest_profit_category"),
    ],
)
def test_supported_numerical_questions_route_to_analytics(question, intent):
    assert route_question(question) == intent


def test_kpi_answers_match_existing_analytics_and_never_call_llm(retail_dataset):
    expected = calculate_kpis(retail_dataset.dataframe)

    for question, intent in (
        ("What is the total revenue?", "total_revenue"),
        ("What is the total profit?", "total_profit"),
        ("How many unique orders are there?", "number_of_orders"),
        ("What is the average order value?", "average_order_value"),
        ("What is the profit margin?", "profit_margin"),
    ):
        result = pipeline(
            retail_dataset,
            question,
            retriever=lambda *_args: pytest.fail("retriever must not run"),
        )
        assert result.mode == "analytics"
        assert result.sources == []
        assert result.calculation["value"] == expected[intent]["value"]


def test_region_and_category_comparisons_use_full_dataframe(retail_dataset):
    region = pipeline(retail_dataset, "Which region generated the highest revenue?")
    category = pipeline(retail_dataset, "Which category has the lowest profit?")

    assert region.calculation == {
        "intent": "highest_revenue_region",
        "available": True,
        "region": "East",
        "revenue": 200.0,
    }
    assert category.calculation["category"] == "Furniture"
    assert category.calculation["profit"] == 20.0


def test_semantic_flow_returns_grounded_sources(retail_dataset):
    match = RetrievalMatch(
        1,
        "chunk-1",
        "Technology item produced profit of 20.",
        1,
        {"upload_id": str(retail_dataset.upload_id), "row_number": 1},
        0.72,
    )

    class FakeLLM:
        def generate_grounded_answer(self, question, matches):
            assert matches == [match]
            return "Source row 1 shows a profitable technology item."

    result = pipeline(
        retail_dataset,
        "Which items are useful for setting up a creative workspace?",
        retriever=lambda *_args: [match],
        llm_service=FakeLLM(),
    )

    assert result.mode == "semantic_rag"
    assert result.sources[0].chunk_id == "chunk-1"
    assert result.sources[0].source_row == 1
    assert result.sources[0].similarity_score == 0.72


def test_missing_evidence_and_unsupported_numbers_do_not_call_llm(retail_dataset):
    empty = pipeline(
        retail_dataset,
        "Describe products",
        retriever=lambda *_args: [],
    )
    unsupported = pipeline(retail_dataset, "Show the monthly revenue trend")
    filtered = pipeline(retail_dataset, "What is the total revenue in West?")

    assert "insufficient" in empty.answer.lower()
    assert empty.mode == "unsupported"
    assert empty.sources == []
    assert unsupported.mode == "unsupported"
    assert "not yet supported" in unsupported.answer
    assert filtered.mode == "analytics"
    assert filtered.calculation["value"] == 150.0


def test_exact_category_and_profit_filter_uses_structured_analytics(retail_dataset):
    result = pipeline(
        retail_dataset,
        "Which Technology line items were profitable?",
        retriever=lambda *_args: pytest.fail("semantic retriever must not run"),
    )

    assert result.mode == "analytics"
    assert result.calculation["rows_after_filtering"] == 1
    assert result.calculation["records"][0]["Sales"] == 100.0


def test_missing_dataset_and_cross_dataset_source_are_rejected(retail_dataset):
    with pytest.raises(PipelineDatasetNotFoundError):
        answer_question(
            retail_dataset.upload_id,
            "Describe products",
            metadata_lookup=lambda _id: None,
        )

    wrong = RetrievalMatch(
        1, "wrong", "text", 1, {"upload_id": str(uuid4())}, 0.9
    )
    with pytest.raises(RAGPipelineError):
        pipeline(
            retail_dataset,
            "Describe products",
            retriever=lambda *_args: [wrong],
        )
