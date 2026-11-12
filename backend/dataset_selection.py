from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from backend.analytics import build_kpi_warnings, calculate_kpis
from backend.analytics_filters import (
    DashboardFilters,
    FilteredData,
    apply_dashboard_filters,
    build_filter_options,
)
from backend.chart_analytics import (
    calculate_category_performance,
    calculate_monthly_revenue,
    calculate_profit_analysis,
    calculate_sales_by_region,
)
from backend.csv_utils import CSVValidationError, read_csv_dataframe
from backend.file_storage import resolve_stored_upload_path
from backend.upload_metadata import DatabaseOperationError, get_upload_metadata

SessionFactory = Callable[[], Session]


class DatasetSelectionError(RuntimeError):
    """Base error for loading a dataset selected by upload ID."""


class DatasetNotFoundError(DatasetSelectionError):
    """Raised when an upload ID has no metadata record."""


class DatasetUnavailableError(DatasetSelectionError):
    """Raised when metadata exists but the original file is unavailable."""


class DatasetInvalidError(DatasetSelectionError):
    """Raised when the stored original can no longer be parsed as CSV."""


@dataclass(frozen=True)
class SelectedDataset:
    upload_id: UUID
    original_filename: str
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class FilteredSelectedDataset:
    upload_id: UUID
    original_filename: str
    filtered: FilteredData


def load_selected_dataset(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
) -> SelectedDataset:
    """Load an original CSV by its metadata ID without exposing its stored path."""
    try:
        metadata = get_upload_metadata(upload_id, session_factory=session_factory)
    except DatabaseOperationError as exc:
        raise DatasetSelectionError("Dataset metadata is unavailable.") from exc

    if metadata is None:
        raise DatasetNotFoundError("The selected dataset was not found.")

    try:
        stored_path = resolve_stored_upload_path(
            metadata["stored_file_path"],
            storage_directory=storage_directory,
        )
        with stored_path.open("rb") as stored_file:
            dataframe = read_csv_dataframe(stored_file)
    except OSError as exc:
        raise DatasetUnavailableError("The selected dataset file is unavailable.") from exc
    except CSVValidationError as exc:
        raise DatasetInvalidError("The selected dataset is not a readable CSV.") from exc

    dataframe.index = pd.RangeIndex(1, len(dataframe) + 1, name="source_row")
    return SelectedDataset(
        upload_id=upload_id,
        original_filename=metadata["filename"],
        dataframe=dataframe,
    )


def load_filtered_selected_dataset(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> FilteredSelectedDataset:
    selected = load_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
    )
    return FilteredSelectedDataset(
        upload_id=selected.upload_id,
        original_filename=selected.original_filename,
        filtered=apply_dashboard_filters(selected.dataframe, filters),
    )


def _filtered_response(
    selected: FilteredSelectedDataset,
    result: dict[str, Any],
) -> dict[str, Any]:
    result["warnings"] = selected.filtered.warnings + result.get("warnings", [])
    return {
        "upload_id": str(selected.upload_id),
        "filename": selected.original_filename,
        "filter_context": selected.filtered.context(),
        **result,
    }


def calculate_selected_dataset_kpis(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    selected = load_filtered_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
        filters=filters,
    )
    kpis = calculate_kpis(selected.filtered.dataframe)
    return {
        "upload_id": str(selected.upload_id),
        "filename": selected.original_filename,
        "filter_context": selected.filtered.context(),
        "kpis": kpis,
        "warnings": selected.filtered.warnings + build_kpi_warnings(kpis),
    }


def calculate_selected_monthly_revenue(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    selected = load_filtered_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
        filters=filters,
    )
    return _filtered_response(
        selected,
        calculate_monthly_revenue(selected.filtered.dataframe),
    )


def calculate_selected_sales_by_region(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    selected = load_filtered_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
        filters=filters,
    )
    return _filtered_response(
        selected,
        calculate_sales_by_region(selected.filtered.dataframe),
    )


def calculate_selected_category_performance(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    selected = load_filtered_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
        filters=filters,
    )
    return _filtered_response(
        selected,
        calculate_category_performance(selected.filtered.dataframe),
    )


def calculate_selected_profit_analysis(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    selected = load_filtered_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
        filters=filters,
    )
    return _filtered_response(
        selected,
        calculate_profit_analysis(selected.filtered.dataframe),
    )


def calculate_selected_filter_options(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
) -> dict[str, Any]:
    selected = load_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
    )
    return {
        "upload_id": str(selected.upload_id),
        "filename": selected.original_filename,
        "options": build_filter_options(selected.dataframe),
    }


def calculate_selected_dashboard(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    """Calculate every dashboard view from one filtered DataFrame."""
    selected = load_filtered_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
        filters=filters,
    )
    dataframe = selected.filtered.dataframe
    kpis = calculate_kpis(dataframe)
    return {
        "upload_id": str(selected.upload_id),
        "filename": selected.original_filename,
        "filter_context": selected.filtered.context(),
        "summary": {
            "kpis": kpis,
            "warnings": build_kpi_warnings(kpis),
        },
        "charts": {
            "monthly_revenue": calculate_monthly_revenue(dataframe),
            "sales_by_region": calculate_sales_by_region(dataframe),
            "category_performance": calculate_category_performance(dataframe),
            "profit_analysis": calculate_profit_analysis(dataframe),
        },
    }
