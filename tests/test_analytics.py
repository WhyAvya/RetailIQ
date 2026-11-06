import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.analytics import build_kpi_warnings, calculate_kpis
from backend.database.db import Base
from backend.dataset_selection import (
    DatasetInvalidError,
    DatasetNotFoundError,
    DatasetUnavailableError,
    calculate_selected_dataset_kpis,
    load_selected_dataset,
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


def test_sample_superstore_kpis_match_independent_pandas_calculations():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    original_hash = sha256(source_path.read_bytes()).hexdigest()
    dataframe = pd.read_csv(source_path, encoding="cp1252")

    result = calculate_kpis(dataframe)

    expected_revenue = pd.to_numeric(dataframe["Sales"], errors="coerce").sum(min_count=1)
    expected_profit = pd.to_numeric(dataframe["Profit"], errors="coerce").sum(min_count=1)
    expected_orders = dataframe["Order ID"].dropna().nunique()
    expected_customers = dataframe["Customer ID"].dropna().nunique()

    assert result["total_revenue"]["value"] == expected_revenue
    assert result["total_profit"]["value"] == expected_profit
    assert result["number_of_orders"]["value"] == expected_orders
    assert result["unique_customers"]["value"] == expected_customers
    assert result["average_order_value"]["value"] == expected_revenue / expected_orders
    assert result["profit_margin"]["value"] == (expected_profit / expected_revenue) * 100
    assert all(metric["available"] for metric in result.values())
    assert sha256(source_path.read_bytes()).hexdigest() == original_hash


def test_missing_retail_columns_are_marked_unavailable():
    result = calculate_kpis(pd.DataFrame({"Product": ["Tea"], "Units": [4]}))

    assert set(result) == {
        "total_revenue",
        "total_profit",
        "number_of_orders",
        "unique_customers",
        "average_order_value",
        "profit_margin",
    }
    assert all(metric["value"] is None for metric in result.values())
    assert all(not metric["available"] for metric in result.values())
    assert all(metric["reason"] for metric in result.values())


def test_invalid_numeric_values_are_ignored_and_reported():
    dataframe = pd.DataFrame(
        {
            "Sales": ["100.25", "invalid", None, float("inf")],
            "Profit": ["10.5", "invalid", None, float("-inf")],
            "Order ID": ["A", "A", "B", "B"],
            "Customer ID": ["C1", "C1", "C2", "C2"],
        }
    )

    result = calculate_kpis(dataframe)

    assert result["total_revenue"]["value"] == 100.25
    assert result["total_revenue"]["ignored_values"] == 3
    assert result["total_profit"]["value"] == 10.5
    assert result["total_profit"]["ignored_values"] == 3
    assert result["number_of_orders"]["value"] == 2
    assert result["unique_customers"]["value"] == 2
    assert result["average_order_value"]["value"] == 100.25 / 2
    assert result["profit_margin"]["value"] == (10.5 / 100.25) * 100
    assert build_kpi_warnings(result) == [
        "Total revenue: ignored 3 null, invalid, or non-finite values.",
        "Total profit: ignored 3 null, invalid, or non-finite values.",
    ]
    json.dumps(result, allow_nan=False)


def test_all_invalid_numbers_and_null_identifiers_are_unavailable():
    dataframe = pd.DataFrame(
        {
            "Sales": ["invalid", None],
            "Profit": [None, "invalid"],
            "Order ID": [None, ""],
            "Customer ID": [None, "  "],
        }
    )

    result = calculate_kpis(dataframe)

    assert all(not metric["available"] for metric in result.values())
    assert result["total_revenue"]["ignored_values"] == 2
    assert result["total_profit"]["ignored_values"] == 2
    assert result["number_of_orders"]["ignored_values"] == 2
    assert result["unique_customers"]["ignored_values"] == 2


def test_overflow_is_marked_unavailable_and_remains_valid_json():
    dataframe = pd.DataFrame(
        {
            "Sales": [1e308, 1e308],
            "Profit": [1, 1],
            "Order ID": ["A", "B"],
            "Customer ID": ["C1", "C2"],
        }
    )

    with pytest.warns(RuntimeWarning):
        result = calculate_kpis(dataframe)

    assert not result["total_revenue"]["available"]
    assert result["total_revenue"]["value"] is None
    assert not result["average_order_value"]["available"]
    assert not result["profit_margin"]["available"]
    json.dumps(result, allow_nan=False)


def test_duplicate_orders_are_counted_once():
    dataframe = pd.DataFrame(
        {
            "Sales": [10, 20, 30],
            "Profit": [1, 2, 3],
            "Order ID": ["A", "A", "B"],
            "Customer ID": ["C1", "C1", "C2"],
        }
    )

    result = calculate_kpis(dataframe)

    assert result["number_of_orders"]["value"] == 2
    assert result["average_order_value"]["value"] == 30


def test_zero_revenue_marks_profit_margin_unavailable():
    dataframe = pd.DataFrame(
        {
            "Sales": [0, 0],
            "Profit": [-5, 5],
            "Order ID": ["A", "B"],
            "Customer ID": ["C1", "C2"],
        }
    )

    result = calculate_kpis(dataframe)

    assert result["total_revenue"]["available"]
    assert result["total_revenue"]["value"] == 0
    assert result["average_order_value"]["value"] == 0
    assert not result["profit_margin"]["available"]
    assert result["profit_margin"]["value"] is None
    assert "revenue is zero" in result["profit_margin"]["reason"]


def test_column_variations_and_two_selected_datasets(
    session_factory,
    tmp_path,
):
    first_contents = (
        b"Total Revenue,Net Profit,order_id,customer-id\n"
        b"100.5,20.25,A,C1\n50.25,5.5,B,C2\n"
    )
    second_contents = (
        b"Sales,Profit,Order ID,Customer ID\n"
        b"10,1,X,U1\n20,2,X,U1\n30,3,Y,U2\n"
    )
    first = _store_with_metadata(first_contents, "first.csv", tmp_path, session_factory)
    second = _store_with_metadata(second_contents, "second.csv", tmp_path, session_factory)
    first_hash = sha256(first.stored_path.read_bytes()).hexdigest()
    second_hash = sha256(second.stored_path.read_bytes()).hexdigest()

    first_result = calculate_selected_dataset_kpis(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    second_result = calculate_selected_dataset_kpis(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )

    assert first_result["upload_id"] == str(first.upload_id)
    assert first_result["kpis"]["total_revenue"]["value"] == 150.75
    assert first_result["kpis"]["number_of_orders"]["value"] == 2
    assert second_result["upload_id"] == str(second.upload_id)
    assert second_result["kpis"]["total_revenue"]["value"] == 60
    assert second_result["kpis"]["number_of_orders"]["value"] == 2
    assert sha256(first.stored_path.read_bytes()).hexdigest() == first_hash
    assert sha256(second.stored_path.read_bytes()).hexdigest() == second_hash


def test_missing_and_deleted_selected_datasets_are_reported(
    session_factory,
    tmp_path,
):
    with pytest.raises(DatasetNotFoundError, match="not found"):
        load_selected_dataset(
            uuid4(),
            session_factory=session_factory,
            storage_directory=tmp_path,
        )

    stored = _store_with_metadata(
        b"Sales,Profit,Order ID,Customer ID\n10,1,A,C1\n",
        "deleted.csv",
        tmp_path,
        session_factory,
    )
    stored.stored_path.unlink()

    with pytest.raises(DatasetUnavailableError, match="unavailable"):
        load_selected_dataset(
            stored.upload_id,
            session_factory=session_factory,
            storage_directory=tmp_path,
        )


def test_invalid_selected_csv_is_reported(session_factory, tmp_path):
    stored = _store_with_metadata(
        b'Sales,Profit\n"unterminated,10\n',
        "invalid.csv",
        tmp_path,
        session_factory,
    )

    with pytest.raises(DatasetInvalidError, match="not a readable CSV"):
        load_selected_dataset(
            stored.upload_id,
            session_factory=session_factory,
            storage_directory=tmp_path,
        )
