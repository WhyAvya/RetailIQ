from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.analytics import calculate_kpis
from backend.analytics_filters import (
    AnalyticsFilterError,
    DashboardFilters,
    apply_dashboard_filters,
    build_filter_options,
)
from backend.chart_analytics import (
    calculate_category_performance,
    calculate_monthly_revenue,
    calculate_profit_analysis,
    calculate_sales_by_region,
)
from backend.database.db import Base
from backend.dataset_selection import (
    calculate_selected_dashboard,
    calculate_selected_filter_options,
)
from backend.file_storage import store_original_csv
from backend.upload_metadata import persist_upload_metadata


@pytest.fixture(scope="module")
def sample_dataframe():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    return pd.read_csv(source_path, encoding="cp1252")


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


def _independent_sample_filter(dataframe, filters):
    dates = pd.to_datetime(dataframe["Order Date"], format="%m/%d/%Y", errors="coerce")
    mask = pd.Series(True, index=dataframe.index)
    if filters.region:
        mask &= dataframe["Region"].eq(filters.region)
    if filters.category:
        mask &= dataframe["Category"].eq(filters.category)
    if filters.start_date:
        mask &= dates.dt.date.ge(filters.start_date)
    if filters.end_date:
        mask &= dates.dt.date.le(filters.end_date)
    return dataframe.loc[mask].copy()


def test_sample_filter_options_are_discovered_from_csv(sample_dataframe):
    options = build_filter_options(sample_dataframe)
    dates = pd.to_datetime(
        sample_dataframe["Order Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    assert options["regions"]["values"] == sorted(
        sample_dataframe["Region"].dropna().unique().tolist(), key=str.casefold
    )
    assert options["categories"]["values"] == sorted(
        sample_dataframe["Category"].dropna().unique().tolist(), key=str.casefold
    )
    assert options["date_range"]["min"] == dates.min().date().isoformat()
    assert options["date_range"]["max"] == dates.max().date().isoformat()


@pytest.mark.parametrize(
    "filters",
    [
        DashboardFilters(region="West"),
        DashboardFilters(category="Furniture"),
        DashboardFilters(start_date=date(2017, 1, 1), end_date=date(2017, 12, 31)),
        DashboardFilters(
            region="West",
            category="Technology",
            start_date=date(2016, 1, 1),
            end_date=date(2016, 12, 31),
        ),
    ],
)
def test_individual_and_combined_filters_match_independent_pandas(
    sample_dataframe,
    filters,
):
    original = sample_dataframe.copy(deep=True)

    result = apply_dashboard_filters(sample_dataframe, filters)
    expected = _independent_sample_filter(sample_dataframe, filters)

    assert_frame_equal(result.dataframe, expected)
    assert result.rows_before == len(sample_dataframe)
    assert result.rows_after == len(expected)
    assert_frame_equal(sample_dataframe, original)


def test_date_boundaries_are_inclusive():
    dataframe = pd.DataFrame(
        {
            "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Sales": [10, 20, 30],
        }
    )

    result = apply_dashboard_filters(
        dataframe,
        DashboardFilters(start_date=date(2024, 1, 2), end_date=date(2024, 1, 3)),
    )

    assert result.dataframe["Sales"].tolist() == [20, 30]


def test_empty_combined_filter_result_is_valid():
    dataframe = pd.DataFrame(
        {
            "Region": ["West", "East"],
            "Category": ["Furniture", "Technology"],
            "Order Date": ["2024-01-01", "2024-01-02"],
        }
    )

    result = apply_dashboard_filters(
        dataframe,
        DashboardFilters(region="West", category="Technology"),
    )

    assert result.rows_after == 0
    assert result.dataframe.empty


def test_invalid_dates_are_excluded_and_reported():
    dataframe = pd.DataFrame(
        {
            "Order Date": ["2024-01-01", "invalid", None],
            "Sales": [10, 20, 30],
        }
    )

    result = apply_dashboard_filters(
        dataframe,
        DashboardFilters(start_date=date(2024, 1, 1)),
    )

    assert result.dataframe["Sales"].tolist() == [10]
    assert result.warnings == [
        "Date filtering excluded 2 rows with missing or invalid dates."
    ]


@pytest.mark.parametrize(
    ("dataframe", "filters", "message"),
    [
        (
            pd.DataFrame({"Region": ["West"]}),
            DashboardFilters(region="Unknown"),
            "region filter value is not available",
        ),
        (
            pd.DataFrame({"Sales": [10]}),
            DashboardFilters(category="Furniture"),
            "no category column",
        ),
        (
            pd.DataFrame({"Sales": [10]}),
            DashboardFilters(start_date=date(2024, 1, 1)),
            "no recognizable date column",
        ),
        (
            pd.DataFrame({"Order Date": ["2024-01-01"]}),
            DashboardFilters(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 1, 1),
            ),
            "Start date must be on or before end date",
        ),
    ],
)
def test_invalid_filter_inputs_raise_clear_errors(dataframe, filters, message):
    with pytest.raises(AnalyticsFilterError, match=message):
        apply_dashboard_filters(dataframe, filters)


def test_all_filtered_kpis_and_charts_match_independent_pandas(sample_dataframe):
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    original_hash = sha256(source_path.read_bytes()).hexdigest()
    filters = DashboardFilters(
        region="West",
        category="Technology",
        start_date=date(2016, 1, 1),
        end_date=date(2016, 12, 31),
    )
    filtered = apply_dashboard_filters(sample_dataframe, filters).dataframe
    independent = _independent_sample_filter(sample_dataframe, filters)

    kpis = calculate_kpis(filtered)
    expected_revenue = pd.to_numeric(independent["Sales"], errors="coerce").sum()
    expected_profit = pd.to_numeric(independent["Profit"], errors="coerce").sum()
    expected_orders = independent["Order ID"].nunique()
    expected_customers = independent["Customer ID"].nunique()
    assert kpis["total_revenue"]["value"] == expected_revenue
    assert kpis["total_profit"]["value"] == expected_profit
    assert kpis["number_of_orders"]["value"] == expected_orders
    assert kpis["unique_customers"]["value"] == expected_customers
    assert kpis["average_order_value"]["value"] == expected_revenue / expected_orders
    assert kpis["profit_margin"]["value"] == expected_profit / expected_revenue * 100

    monthly = calculate_monthly_revenue(filtered)
    expected_monthly = (
        independent.assign(
            month=pd.to_datetime(
                independent["Order Date"], format="%m/%d/%Y", errors="coerce"
            ).dt.to_period("M"),
            revenue=pd.to_numeric(independent["Sales"], errors="coerce"),
        )
        .groupby("month")["revenue"]
        .sum()
    )
    assert [item["month"] for item in monthly["data"]] == [
        month.strftime("%Y-%m") for month in expected_monthly.index
    ]
    assert [item["revenue"] for item in monthly["data"]] == expected_monthly.tolist()

    regional = calculate_sales_by_region(filtered)
    assert regional["data"] == [{"region": "West", "revenue": expected_revenue}]

    category = calculate_category_performance(filtered)
    assert category["data"] == [
        {
            "category": "Technology",
            "revenue": expected_revenue,
            "profit": expected_profit,
        }
    ]

    profit = calculate_profit_analysis(filtered)
    expected_segments = (
        independent.assign(
            revenue=pd.to_numeric(independent["Sales"], errors="coerce"),
            numeric_profit=pd.to_numeric(independent["Profit"], errors="coerce"),
        )
        .groupby("Sub-Category", sort=False)[["revenue", "numeric_profit"]]
        .sum()
        .sort_values("numeric_profit", ascending=False, kind="stable")
    )
    assert [item["segment"] for item in profit["data"]] == expected_segments.index.tolist()
    assert [item["profit"] for item in profit["data"]] == expected_segments[
        "numeric_profit"
    ].tolist()
    assert [item["profit_margin"] for item in profit["data"]] == pytest.approx(
        (expected_segments["numeric_profit"] / expected_segments["revenue"] * 100).tolist()
    )
    assert sha256(source_path.read_bytes()).hexdigest() == original_hash


def test_filtered_dashboard_and_options_switch_with_upload_id(
    session_factory,
    tmp_path,
):
    first_contents = (
        b"Region,Category,Order Date,Sales,Profit,Order ID,Customer ID,Sub-Category\n"
        b"West,Technology,2024-01-01,100,20,A,C1,Phones\n"
    )
    second_contents = (
        b"Region,Category,Order Date,Sales,Profit,Order ID,Customer ID,Sub-Category\n"
        b"East,Furniture,2024-02-01,50,-5,B,C2,Chairs\n"
    )
    first = _store_with_metadata(
        first_contents, "first.csv", tmp_path, session_factory
    )
    second = _store_with_metadata(
        second_contents, "second.csv", tmp_path, session_factory
    )
    first_hash = sha256(first.stored_path.read_bytes()).hexdigest()
    second_hash = sha256(second.stored_path.read_bytes()).hexdigest()

    first_dashboard = calculate_selected_dashboard(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
        filters=DashboardFilters(region="West"),
    )
    second_dashboard = calculate_selected_dashboard(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
        filters=DashboardFilters(region="East"),
    )
    first_options = calculate_selected_filter_options(
        first.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    second_options = calculate_selected_filter_options(
        second.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )

    assert first_dashboard["upload_id"] == str(first.upload_id)
    assert first_dashboard["summary"]["kpis"]["total_revenue"]["value"] == 100
    assert second_dashboard["upload_id"] == str(second.upload_id)
    assert second_dashboard["summary"]["kpis"]["total_revenue"]["value"] == 50
    assert first_options["options"]["regions"]["values"] == ["West"]
    assert second_options["options"]["regions"]["values"] == ["East"]
    assert sha256(first.stored_path.read_bytes()).hexdigest() == first_hash
    assert sha256(second.stored_path.read_bytes()).hexdigest() == second_hash
