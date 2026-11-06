from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.chart_analytics import calculate_monthly_revenue, calculate_sales_by_region
from backend.database.db import Base
from backend.dataset_selection import (
    calculate_selected_monthly_revenue,
    calculate_selected_sales_by_region,
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


def test_sample_monthly_revenue_matches_independent_pandas_calculation():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    original_hash = sha256(source_path.read_bytes()).hexdigest()
    dataframe = pd.read_csv(source_path, encoding="cp1252")

    result = calculate_monthly_revenue(dataframe)

    dates = pd.to_datetime(dataframe["Order Date"], format="%m/%d/%Y", errors="coerce")
    sales = pd.to_numeric(dataframe["Sales"], errors="coerce")
    expected = (
        pd.DataFrame({"month": dates.dt.to_period("M"), "revenue": sales})
        .dropna()
        .groupby("month", sort=True)["revenue"]
        .sum(min_count=1)
    )

    assert result["available"]
    assert result["warnings"] == []
    assert [item["month"] for item in result["data"]] == [
        month.strftime("%Y-%m") for month in expected.index
    ]
    assert [item["revenue"] for item in result["data"]] == expected.tolist()
    assert sha256(source_path.read_bytes()).hexdigest() == original_hash


def test_sample_regional_sales_matches_independent_pandas_calculation():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    dataframe = pd.read_csv(source_path, encoding="cp1252")

    result = calculate_sales_by_region(dataframe)

    expected = (
        dataframe.assign(revenue=pd.to_numeric(dataframe["Sales"], errors="coerce"))
        .dropna(subset=["Region", "revenue"])
        .groupby("Region", sort=False)["revenue"]
        .sum(min_count=1)
        .sort_values(ascending=False, kind="stable")
    )

    assert result["available"]
    assert result["warnings"] == []
    assert [item["region"] for item in result["data"]] == expected.index.tolist()
    assert [item["revenue"] for item in result["data"]] == expected.tolist()


def test_monthly_revenue_ignores_invalid_dates_and_sales():
    dataframe = pd.DataFrame(
        {
            "Transaction Date": ["2024-01-05", "invalid", None, "2024-01-20"],
            "Revenue": [10, 20, 30, "5.5"],
        }
    )

    result = calculate_monthly_revenue(dataframe)

    assert result["available"]
    assert result["data"] == [
        {"month": "2024-01", "label": "Jan 2024", "revenue": 15.5}
    ]
    assert result["warnings"] == ["Ignored 2 rows with missing or invalid dates."]


def test_regional_sales_ignores_null_regions_and_invalid_sales():
    dataframe = pd.DataFrame(
        {
            "Territory": ["West", None, "", "East", "West"],
            "Revenue": [10, 20, 30, "invalid", 5],
        }
    )

    result = calculate_sales_by_region(dataframe)

    assert result["available"]
    assert result["data"] == [{"region": "West", "revenue": 15}]
    assert result["warnings"] == [
        "Ignored 2 rows with missing or empty regions.",
        "Ignored 1 rows with missing, invalid, or non-finite sales values.",
    ]


@pytest.mark.parametrize(
    ("dataframe", "monthly_reason", "regional_reason"),
    [
        (
            pd.DataFrame({"Product": ["Tea"]}),
            "Date and sales columns were not found.",
            "Region and sales columns were not found.",
        ),
        (
            pd.DataFrame(columns=["Order Date", "Sales", "Region"]),
            "No valid monthly revenue data is available.",
            "No valid regional revenue data is available.",
        ),
    ],
)
def test_missing_columns_and_empty_results_are_unavailable(
    dataframe,
    monthly_reason,
    regional_reason,
):
    monthly = calculate_monthly_revenue(dataframe)
    regional = calculate_sales_by_region(dataframe)

    assert not monthly["available"]
    assert monthly["reason"] == monthly_reason
    assert monthly["data"] == []
    assert not regional["available"]
    assert regional["reason"] == regional_reason
    assert regional["data"] == []


def test_individual_required_columns_are_reported_when_missing():
    without_sales = pd.DataFrame(
        {"Order Date": ["2024-01-01"], "Region": ["West"]}
    )
    sales_only = pd.DataFrame({"Sales": [10]})

    assert calculate_monthly_revenue(without_sales)["reason"] == (
        "A sales or revenue column was not found."
    )
    assert calculate_sales_by_region(without_sales)["reason"] == (
        "A sales or revenue column was not found."
    )
    assert calculate_monthly_revenue(sales_only)["reason"] == (
        "A recognizable date column was not found."
    )
    assert calculate_sales_by_region(sales_only)["reason"] == (
        "A region column was not found."
    )


def test_selected_chart_data_changes_with_upload_id(
    session_factory,
    tmp_path,
):
    first = _store_with_metadata(
        b"Order Date,Sales,Region\n2024-01-01,10,West\n",
        "first.csv",
        tmp_path,
        session_factory,
    )
    second = _store_with_metadata(
        b"Order Date,Sales,Region\n2024-02-01,25,East\n2024-02-02,5,East\n",
        "second.csv",
        tmp_path,
        session_factory,
    )

    first_monthly = calculate_selected_monthly_revenue(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    second_monthly = calculate_selected_monthly_revenue(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    first_region = calculate_selected_sales_by_region(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    second_region = calculate_selected_sales_by_region(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )

    assert first_monthly["upload_id"] == str(first.upload_id)
    assert first_monthly["data"][0]["revenue"] == 10
    assert second_monthly["upload_id"] == str(second.upload_id)
    assert second_monthly["data"][0]["revenue"] == 30
    assert first_region["data"] == [{"region": "West", "revenue": 10}]
    assert second_region["data"] == [{"region": "East", "revenue": 30}]
