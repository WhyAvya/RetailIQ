from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from backend.analytics import find_column


class AnalyticsFilterError(ValueError):
    """Raised when requested dashboard filters cannot be applied safely."""


@dataclass(frozen=True)
class DashboardFilters:
    region: str | None = None
    category: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    def __post_init__(self) -> None:
        for field in ("region", "category"):
            value = getattr(self, field)
            if value is not None:
                normalized = value.strip()
                object.__setattr__(self, field, normalized or None)

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.region, self.category, self.start_date, self.end_date)
        )

    def as_dict(self) -> dict[str, str | None]:
        return {
            "region": self.region,
            "category": self.category,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


@dataclass(frozen=True)
class FilteredData:
    dataframe: pd.DataFrame
    filters: DashboardFilters
    rows_before: int
    rows_after: int
    warnings: list[str]

    def context(self) -> dict[str, Any]:
        return {
            "applied": self.filters.as_dict(),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "warnings": self.warnings,
        }


def _clean_dimension(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


def _dimension_options(dataframe: pd.DataFrame, field: str, label: str) -> dict[str, Any]:
    column = find_column(dataframe, field)
    if column is None:
        return {
            "available": False,
            "values": [],
            "reason": f"A {label} column was not found.",
        }

    values = _clean_dimension(dataframe[column])
    values = values.loc[values.notna() & values.ne("")]
    options = sorted(values.unique().tolist(), key=str.casefold)
    return {
        "available": bool(options),
        "values": options,
        "reason": None if options else f"The {column} column has no usable values.",
    }


def build_filter_options(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Discover available dashboard filters without modifying the source data."""
    date_column = find_column(dataframe, "order_date")
    if date_column is None:
        date_range = {
            "available": False,
            "min": None,
            "max": None,
            "reason": "A recognizable date column was not found.",
        }
    else:
        parsed_dates = pd.to_datetime(
            dataframe[date_column],
            errors="coerce",
            format="mixed",
            utc=True,
        )
        valid_dates = parsed_dates.dropna()
        date_range = {
            "available": not valid_dates.empty,
            "min": valid_dates.min().date().isoformat() if not valid_dates.empty else None,
            "max": valid_dates.max().date().isoformat() if not valid_dates.empty else None,
            "reason": (
                None
                if not valid_dates.empty
                else f"The {date_column} column has no recognizable dates."
            ),
        }

    return {
        "regions": _dimension_options(dataframe, "region", "region"),
        "categories": _dimension_options(dataframe, "category", "category"),
        "date_range": date_range,
    }


def _match_dimension(
    dataframe: pd.DataFrame,
    mask: pd.Series,
    field: str,
    requested: str,
    label: str,
) -> pd.Series:
    column = find_column(dataframe, field)
    if column is None:
        raise AnalyticsFilterError(
            f"{label} filtering is unavailable because the dataset has no {label.lower()} column."
        )

    values = _clean_dimension(dataframe[column])
    available = {
        value.casefold(): value
        for value in values.loc[values.notna() & values.ne("")].unique().tolist()
    }
    requested_key = requested.casefold()
    if requested_key not in available:
        raise AnalyticsFilterError(
            f"The requested {label.lower()} filter value is not available in this dataset."
        )
    return mask & values.str.casefold().eq(requested_key).fillna(False)


def apply_dashboard_filters(
    dataframe: pd.DataFrame,
    filters: DashboardFilters | None = None,
) -> FilteredData:
    """Apply inclusive dashboard filters and return a filtered copy."""
    filters = filters or DashboardFilters()
    if filters.start_date and filters.end_date and filters.start_date > filters.end_date:
        raise AnalyticsFilterError("Start date must be on or before end date.")

    mask = pd.Series(True, index=dataframe.index, dtype=bool)
    warnings = []
    if filters.region:
        mask = _match_dimension(dataframe, mask, "region", filters.region, "Region")
    if filters.category:
        mask = _match_dimension(dataframe, mask, "category", filters.category, "Category")

    if filters.start_date or filters.end_date:
        date_column = find_column(dataframe, "order_date")
        if date_column is None:
            raise AnalyticsFilterError(
                "Date filtering is unavailable because the dataset has no recognizable date column."
            )
        parsed_dates = pd.to_datetime(
            dataframe[date_column],
            errors="coerce",
            format="mixed",
            utc=True,
        )
        invalid_dates = int(len(parsed_dates) - parsed_dates.notna().sum())
        if invalid_dates:
            warnings.append(
                f"Date filtering excluded {invalid_dates} rows with missing or invalid dates."
            )
        valid_dates = parsed_dates.notna()
        mask &= valid_dates
        normalized_dates = parsed_dates.dt.date
        if filters.start_date:
            mask &= normalized_dates.ge(filters.start_date).fillna(False)
        if filters.end_date:
            mask &= normalized_dates.le(filters.end_date).fillna(False)

    filtered = dataframe.loc[mask].copy()
    return FilteredData(
        dataframe=filtered,
        filters=filters,
        rows_before=int(len(dataframe)),
        rows_after=int(len(filtered)),
        warnings=warnings,
    )
