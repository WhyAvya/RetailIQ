from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.chart_analytics import (
    calculate_category_performance,
    calculate_profit_analysis,
)
from backend.database.db import Base
from backend.dataset_selection import (
    calculate_selected_category_performance,
    calculate_selected_profit_analysis,
)
from backend.file_storage import store_original_csv
from backend.upload_metadata import persist_upload_metadata


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def _store_with_metadata(contents, filename, storage_directory, session_factory):
    stored_upload = store_original_csv(
        BytesIO(contents),
        filename,
        storage_directory=storage_directory,
    )
    persist_upload_metadata(
        stored_upload,
        len(contents),
        session_factory=session_factory,
    )
    return stored_upload


def test_sample_category_performance_matches_independent_pandas_calculation():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    original_hash = sha256(source_path.read_bytes()).hexdigest()
    dataframe = pd.read_csv(source_path, encoding="cp1252")

    result = calculate_category_performance(dataframe)
    expected = (
        dataframe.assign(
            revenue=pd.to_numeric(dataframe["Sales"], errors="coerce"),
            numeric_profit=pd.to_numeric(dataframe["Profit"], errors="coerce"),
        )
        .groupby("Category", sort=False)[["revenue", "numeric_profit"]]
        .sum(min_count=1)
        .sort_values("revenue", ascending=False, kind="stable")
    )

    assert result["available"]
    assert result["warnings"] == []
    assert [item["category"] for item in result["data"]] == expected.index.tolist()
    assert [item["revenue"] for item in result["data"]] == expected["revenue"].tolist()
    assert [item["profit"] for item in result["data"]] == expected["numeric_profit"].tolist()
    assert sha256(source_path.read_bytes()).hexdigest() == original_hash


def test_sample_profit_analysis_matches_independent_pandas_calculation():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    dataframe = pd.read_csv(source_path, encoding="cp1252")

    result = calculate_profit_analysis(dataframe)
    expected = (
        dataframe.assign(
            revenue=pd.to_numeric(dataframe["Sales"], errors="coerce"),
            numeric_profit=pd.to_numeric(dataframe["Profit"], errors="coerce"),
        )
        .groupby("Sub-Category", sort=False)[["revenue", "numeric_profit"]]
        .sum(min_count=1)
        .sort_values("numeric_profit", ascending=False, kind="stable")
    )

    assert result["available"]
    assert result["dimension"] == "Sub-Category"
    assert result["dimension_type"] == "sub_category"
    assert [item["segment"] for item in result["data"]] == expected.index.tolist()
    assert [item["profit"] for item in result["data"]] == expected["numeric_profit"].tolist()
    assert [item["profit_margin"] for item in result["data"]] == pytest.approx(
        (expected["numeric_profit"] / expected["revenue"] * 100).tolist()
    )
    assert any(item["status"] == "profitable" for item in result["data"])
    assert any(item["status"] == "loss-making" for item in result["data"])


def test_category_performance_handles_null_categories_and_invalid_values():
    dataframe = pd.DataFrame(
        {
            "Product Category": ["Furniture", "Furniture", None, "", "Office"],
            "Revenue": [100, "bad", 20, 30, 50],
            "Net Profit": [10, -2, 3, 4, "invalid"],
        }
    )

    result = calculate_category_performance(dataframe)

    assert result["available"]
    assert result["data"] == [
        {"category": "Furniture", "revenue": 100, "profit": 8.0},
        {"category": "Office", "revenue": 50, "profit": None},
    ]
    assert len(result["warnings"]) == 4


def test_profit_analysis_handles_losses_zero_revenue_and_category_fallback():
    dataframe = pd.DataFrame(
        {
            "Category": ["Gain", "Loss", "Zero", "Gain"],
            "Sales": [100, 50, 0, "invalid"],
            "Profit": [20, -10, -5, 5],
        }
    )

    result = calculate_profit_analysis(dataframe)
    by_segment = {item["segment"]: item for item in result["data"]}

    assert result["dimension_type"] == "category"
    assert by_segment["Gain"]["profit"] == 25
    assert by_segment["Gain"]["status"] == "profitable"
    assert by_segment["Loss"]["profit_margin"] == -20
    assert by_segment["Loss"]["status"] == "loss-making"
    assert by_segment["Zero"]["profit_margin"] is None
    assert by_segment["Zero"]["status"] == "loss-making"
    assert any("Profit margin is unavailable" in warning for warning in result["warnings"])


def test_profit_analysis_falls_back_when_sub_category_is_empty():
    dataframe = pd.DataFrame(
        {
            "Sub-Category": [None, ""],
            "Category": ["Furniture", "Office"],
            "Sales": [100, 50],
            "Profit": [20, -5],
        }
    )

    result = calculate_profit_analysis(dataframe)

    assert result["available"]
    assert result["dimension_type"] == "category"
    assert {item["segment"] for item in result["data"]} == {"Furniture", "Office"}
    assert "Skipped empty dimensions: Sub-Category." in result["warnings"]


@pytest.mark.parametrize(
    ("dataframe", "category_reason", "profit_reason"),
    [
        (
            pd.DataFrame({"Product": ["Tea"]}),
            "Required columns were not found: category, sales or revenue, profit.",
            "Required columns were not found: sub-category, category, or region, sales or revenue, profit.",
        ),
        (
            pd.DataFrame(columns=["Category", "Sub-Category", "Sales", "Profit"]),
            "No valid category performance data is available.",
            "No valid profit analysis data is available.",
        ),
    ],
)
def test_round4_missing_columns_and_empty_results_are_unavailable(
    dataframe,
    category_reason,
    profit_reason,
):
    category = calculate_category_performance(dataframe)
    profit = calculate_profit_analysis(dataframe)

    assert not category["available"]
    assert category["reason"] == category_reason
    assert category["data"] == []
    assert not profit["available"]
    assert profit["reason"] == profit_reason
    assert profit["data"] == []


def test_round4_selected_data_changes_with_upload_id(session_factory, tmp_path):
    first = _store_with_metadata(
        b"Category,Sub-Category,Sales,Profit\nFurniture,Chair,100,20\n",
        "first.csv",
        tmp_path,
        session_factory,
    )
    second = _store_with_metadata(
        b"Category,Sub-Category,Sales,Profit\nOffice,Paper,40,-5\n",
        "second.csv",
        tmp_path,
        session_factory,
    )

    first_category = calculate_selected_category_performance(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    second_category = calculate_selected_category_performance(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    first_profit = calculate_selected_profit_analysis(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    second_profit = calculate_selected_profit_analysis(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )

    assert first_category["upload_id"] == str(first.upload_id)
    assert first_category["data"][0]["category"] == "Furniture"
    assert second_category["upload_id"] == str(second.upload_id)
    assert second_category["data"][0]["category"] == "Office"
    assert first_profit["data"][0]["status"] == "profitable"
    assert second_profit["data"][0]["status"] == "loss-making"
