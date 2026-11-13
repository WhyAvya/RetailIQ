import asyncio
import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException

import backend.main as main_module
from backend.analytics import build_kpi_warnings, calculate_kpis
from backend.analytics_filters import AnalyticsFilterError
from backend.dataset_selection import (
    DatasetInvalidError,
    DatasetNotFoundError,
    DatasetSelectionError,
    DatasetUnavailableError,
)


def _analytics_result(upload_id, filename, dataframe):
    kpis = calculate_kpis(dataframe)
    return {
        "upload_id": str(upload_id),
        "filename": filename,
        "kpis": kpis,
        "warnings": build_kpi_warnings(kpis),
    }


def test_analytics_summary_returns_all_sample_superstore_kpis(monkeypatch):
    upload_id = uuid4()
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    dataframe = pd.read_csv(source_path, encoding="cp1252")
    expected = _analytics_result(upload_id, source_path.name, dataframe)
    monkeypatch.setattr(
        main_module,
        "calculate_selected_dataset_kpis",
        lambda selected_id: expected if selected_id == upload_id else None,
    )

    response = asyncio.run(main_module.analytics_summary(upload_id))

    assert response["upload_id"] == str(upload_id)
    assert response["filename"] == "sample_superstore.csv"
    assert set(response["kpis"]) == {
        "total_revenue",
        "total_profit",
        "number_of_orders",
        "unique_customers",
        "average_order_value",
        "profit_margin",
    }
    assert response["kpis"]["total_revenue"]["value"] == 2297200.8603000003
    assert response["kpis"]["total_profit"]["value"] == 286397.0217
    assert response["kpis"]["number_of_orders"]["value"] == 5009
    assert response["kpis"]["unique_customers"]["value"] == 793
    assert response["kpis"]["average_order_value"]["value"] == 458.61466566180883
    assert response["kpis"]["profit_margin"]["value"] == 12.467217240315604
    assert response["warnings"] == []
    json.dumps(response, allow_nan=False)


def test_analytics_summary_returns_unavailable_metrics_and_warnings(monkeypatch):
    upload_id = uuid4()
    expected = _analytics_result(
        upload_id,
        "inventory.csv",
        pd.DataFrame({"Product": ["Tea"], "Units": [4]}),
    )
    monkeypatch.setattr(
        main_module,
        "calculate_selected_dataset_kpis",
        lambda _selected_id: expected,
    )

    response = asyncio.run(main_module.analytics_summary(upload_id))

    assert all(not metric["available"] for metric in response["kpis"].values())
    assert all(metric["value"] is None for metric in response["kpis"].values())
    assert response["warnings"]
    assert all("stored_file_path" not in warning for warning in response["warnings"])


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (DatasetNotFoundError("internal"), 404, "The selected dataset was not found."),
        (DatasetUnavailableError("internal"), 404, "The selected dataset file is unavailable."),
        (DatasetInvalidError("internal"), 422, "The selected dataset is not a readable CSV."),
        (DatasetSelectionError("internal"), 503, "Analytics are temporarily unavailable."),
    ],
)
def test_analytics_summary_maps_backend_errors_to_safe_http_responses(
    monkeypatch,
    error,
    status_code,
    detail,
):
    def fail_analytics(_upload_id):
        raise error

    monkeypatch.setattr(main_module, "calculate_selected_dataset_kpis", fail_analytics)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.analytics_summary(uuid4()))

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail
    assert "internal" not in exc_info.value.detail


def test_chart_endpoints_return_selected_dataset_results(monkeypatch):
    upload_id = uuid4()
    monthly_result = {
        "upload_id": str(upload_id),
        "filename": "sales.csv",
        "available": True,
        "reason": None,
        "warnings": [],
        "data": [{"month": "2024-01", "label": "Jan 2024", "revenue": 25}],
    }
    regional_result = {
        "upload_id": str(upload_id),
        "filename": "sales.csv",
        "available": True,
        "reason": None,
        "warnings": [],
        "data": [{"region": "West", "revenue": 25}],
    }
    category_result = {
        "upload_id": str(upload_id),
        "filename": "sales.csv",
        "available": True,
        "reason": None,
        "warnings": [],
        "data": [{"category": "Furniture", "revenue": 25, "profit": 5}],
    }
    profit_result = {
        "upload_id": str(upload_id),
        "filename": "sales.csv",
        "available": True,
        "reason": None,
        "warnings": [],
        "dimension": "Sub-Category",
        "dimension_type": "sub_category",
        "data": [
            {
                "segment": "Chairs",
                "revenue": 25,
                "profit": 5,
                "profit_margin": 20,
                "status": "profitable",
            }
        ],
    }
    monkeypatch.setattr(
        main_module,
        "calculate_selected_monthly_revenue",
        lambda _upload_id: monthly_result,
    )
    monkeypatch.setattr(
        main_module,
        "calculate_selected_sales_by_region",
        lambda _upload_id: regional_result,
    )
    monkeypatch.setattr(
        main_module,
        "calculate_selected_category_performance",
        lambda _upload_id: category_result,
    )
    monkeypatch.setattr(
        main_module,
        "calculate_selected_profit_analysis",
        lambda _upload_id: profit_result,
    )

    assert asyncio.run(main_module.monthly_revenue_trend(upload_id)) == monthly_result
    assert asyncio.run(main_module.regional_sales(upload_id)) == regional_result
    assert asyncio.run(main_module.category_performance(upload_id)) == category_result
    assert asyncio.run(main_module.profit_analysis(upload_id)) == profit_result


def test_currency_endpoint_returns_backward_compatible_dataset_metadata(monkeypatch):
    upload_id = uuid4()
    expected = {
        "upload_id": str(upload_id),
        "filename": "usd-sales.csv",
        "currency": {
            "status": "detected",
            "column": "Currency Code",
            "detected_code": "USD",
            "detected_codes": ["USD"],
            "unsupported_values": [],
        },
    }
    monkeypatch.setattr(main_module, "calculate_selected_currency_metadata", lambda _id: expected)

    assert asyncio.run(main_module.analytics_currency(upload_id)) == expected


@pytest.mark.parametrize(
    ("endpoint", "function_name"),
    [
        ("monthly_revenue_trend", "calculate_selected_monthly_revenue"),
        ("regional_sales", "calculate_selected_sales_by_region"),
        ("category_performance", "calculate_selected_category_performance"),
        ("profit_analysis", "calculate_selected_profit_analysis"),
    ],
)
def test_chart_endpoints_handle_backend_failures(monkeypatch, endpoint, function_name):
    def fail_analytics(_upload_id):
        raise DatasetSelectionError("internal database detail")

    monkeypatch.setattr(main_module, function_name, fail_analytics)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(getattr(main_module, endpoint)(uuid4()))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Analytics are temporarily unavailable."


@pytest.mark.parametrize(
    ("endpoint", "function_name"),
    [
        ("analytics_summary", "calculate_selected_dataset_kpis"),
        ("monthly_revenue_trend", "calculate_selected_monthly_revenue"),
        ("regional_sales", "calculate_selected_sales_by_region"),
        ("category_performance", "calculate_selected_category_performance"),
        ("profit_analysis", "calculate_selected_profit_analysis"),
    ],
)
def test_existing_analytics_endpoints_forward_optional_filters(
    monkeypatch,
    endpoint,
    function_name,
):
    upload_id = uuid4()
    captured = {}
    expected = {"upload_id": str(upload_id), "filtered": True}

    def calculate(_upload_id, *, filters):
        captured["filters"] = filters
        return expected

    monkeypatch.setattr(main_module, function_name, calculate)

    response = asyncio.run(
        getattr(main_module, endpoint)(
            upload_id,
            region="West",
            category="Technology",
            start_date=date(2016, 1, 1),
            end_date=date(2016, 12, 31),
        )
    )

    assert response == expected
    assert captured["filters"].region == "West"
    assert captured["filters"].category == "Technology"
    assert captured["filters"].start_date == date(2016, 1, 1)
    assert captured["filters"].end_date == date(2016, 12, 31)


def test_dashboard_endpoint_forwards_one_shared_filter_set(monkeypatch):
    upload_id = uuid4()
    expected = {
        "upload_id": str(upload_id),
        "filename": "sales.csv",
        "filter_context": {"rows_before": 10, "rows_after": 2},
        "summary": {"kpis": {}, "warnings": []},
        "charts": {},
    }

    def calculate(_upload_id, *, filters):
        assert filters.region == "West"
        return expected

    monkeypatch.setattr(main_module, "calculate_selected_dashboard", calculate)

    response = asyncio.run(main_module.analytics_dashboard(upload_id, region="West"))

    assert response == expected
    assert "stored_file_path" not in json.dumps(response)


def test_filter_options_endpoint_returns_dynamic_options(monkeypatch):
    upload_id = uuid4()
    expected = {
        "upload_id": str(upload_id),
        "filename": "sales.csv",
        "options": {
            "regions": {"available": True, "values": ["East", "West"]},
            "categories": {"available": True, "values": ["Technology"]},
            "date_range": {"available": True, "min": "2024-01-01", "max": "2024-12-31"},
        },
    }
    monkeypatch.setattr(
        main_module,
        "calculate_selected_filter_options",
        lambda _upload_id: expected,
    )

    response = asyncio.run(main_module.analytics_filter_options(upload_id))

    assert response == expected


def test_invalid_filter_error_is_safe_http_400(monkeypatch):
    def fail_filter(_upload_id, *, filters):
        raise AnalyticsFilterError("The requested region filter value is not available.")

    monkeypatch.setattr(main_module, "calculate_selected_dashboard", fail_filter)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.analytics_dashboard(uuid4(), region="Unknown"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "The requested region filter value is not available."
    )


