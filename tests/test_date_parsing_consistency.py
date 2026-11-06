from datetime import date
from hashlib import sha256
from pathlib import Path

import pandas as pd

from backend.analytics_filters import (
    DashboardFilters,
    apply_dashboard_filters,
    build_filter_options,
)
from backend.chart_analytics import calculate_monthly_revenue
from backend.csv_utils import read_csv_dataframe
from backend.data_processing import preprocess_dataframe


RETAILIQ_SAMPLE_DATES = [
    "2026-01-05",
    "2026-01-05",
    "2026-01-18",
    "2026-02-03",
    "2026-02-14",
    "2026-03-02",
    "2026-03-11",
    "2026-03-21",
    "2026-04-04",
    "2026-04-17",
    "2026-05-06",
    "2026-05-22",
]


def _date_conversion(summary):
    return next(
        operation
        for operation in summary["operations"]
        if operation["operation"] == "convert_dates"
        and operation["column"] == "Order Date"
    )


def test_retailiq_sample_iso_dates_are_consistent_across_date_consumers():
    dataframe = pd.DataFrame(
        {
            "Order ID": [f"R-{index}" for index in range(12)],
            "Order Date": RETAILIQ_SAMPLE_DATES,
            "Sales": [1] * len(RETAILIQ_SAMPLE_DATES),
        }
    )
    original = dataframe.copy(deep=True)

    cleaned, summary = preprocess_dataframe(dataframe)
    chart = calculate_monthly_revenue(dataframe)
    options = build_filter_options(dataframe)["date_range"]
    filtered = apply_dashboard_filters(
        dataframe,
        DashboardFilters(
            start_date=date(2026, 1, 5),
            end_date=date(2026, 5, 22),
        ),
    )

    operation = _date_conversion(summary)
    assert operation["affected"] == 12
    assert operation["description"] == (
        "Converted 12 values to datetime; 0 invalid values became missing."
    )
    assert cleaned["Order Date"].dt.strftime("%Y-%m-%d").tolist() == (
        RETAILIQ_SAMPLE_DATES
    )
    assert chart["warnings"] == []
    assert [item["month"] for item in chart["data"]] == [
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
    ]
    assert options == {
        "available": True,
        "min": "2026-01-05",
        "max": "2026-05-22",
        "reason": None,
    }
    assert filtered.rows_after == 12
    assert filtered.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_ambiguous_and_invalid_dates_are_interpreted_consistently():
    dataframe = pd.DataFrame(
        {
            "Order ID": [f"M-{index}" for index in range(6)],
            "Order Date": [
                "01/02/2026",
                "13/02/2026",
                "2026-03-04",
                "2026-02-30",
                "not-a-date",
                None,
            ],
            "Sales": [1, 1, 1, 1, 1, 1],
        }
    )

    cleaned, summary = preprocess_dataframe(dataframe)
    chart = calculate_monthly_revenue(dataframe)
    options = build_filter_options(dataframe)["date_range"]
    filtered = apply_dashboard_filters(
        dataframe,
        DashboardFilters(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 3, 31),
        ),
    )

    operation = _date_conversion(summary)
    assert operation["affected"] == 3
    assert operation["description"] == (
        "Converted 3 values to datetime; 2 invalid values became missing."
    )
    assert (
        cleaned["Order Date"]
        .dt.strftime("%Y-%m-%d")
        .astype("string")
        .fillna("<missing>")
        .tolist()
    ) == [
        "2026-01-02",
        "2026-02-13",
        "2026-03-04",
        "<missing>",
        "<missing>",
        "<missing>",
    ]
    assert [item["month"] for item in chart["data"]] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]
    assert chart["warnings"] == [
        "Ignored 3 rows with missing or invalid dates."
    ]
    assert options["min"] == "2026-01-02"
    assert options["max"] == "2026-03-04"
    assert filtered.rows_after == 3
    assert filtered.warnings == [
        "Date filtering excluded 3 rows with missing or invalid dates."
    ]


def test_sample_superstore_dates_remain_fully_valid_and_unchanged():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    original_hash = sha256(source_path.read_bytes()).hexdigest()
    with source_path.open("rb") as source_file:
        dataframe = read_csv_dataframe(source_file)

    _, summary = preprocess_dataframe(dataframe)
    chart = calculate_monthly_revenue(dataframe)
    options = build_filter_options(dataframe)["date_range"]
    filtered = apply_dashboard_filters(
        dataframe,
        DashboardFilters(
            start_date=date(2014, 1, 3),
            end_date=date(2017, 12, 30),
        ),
    )

    operation = _date_conversion(summary)
    assert operation["affected"] == 9994
    assert operation["description"] == (
        "Converted 9994 values to datetime; 0 invalid values became missing."
    )
    assert chart["warnings"] == []
    assert len(chart["data"]) == 48
    assert options["min"] == "2014-01-03"
    assert options["max"] == "2017-12-30"
    assert filtered.rows_after == 9994
    assert filtered.warnings == []
    assert sha256(source_path.read_bytes()).hexdigest() == original_hash
