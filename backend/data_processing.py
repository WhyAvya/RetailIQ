import re
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

DATE_PARSE_THRESHOLD = 0.5


def _json_number(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _display_value(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _standardized_column_names(columns: pd.Index) -> list[str]:
    names: list[str] = []
    used_names: dict[str, int] = {}

    for position, column in enumerate(columns, start=1):
        base_name = str(column).strip() or f"unnamed_column_{position}"
        occurrence = used_names.get(base_name, 0) + 1
        used_names[base_name] = occurrence
        unique_name = base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        names.append(unique_name)

    return names


def _looks_like_date_column(column_name: str) -> bool:
    normalized = re.sub(r"[\s-]+", "_", column_name.strip().lower())
    return normalized.endswith(("date", "time", "datetime", "timestamp", "_at"))


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    row_count, column_count = dataframe.shape
    column_profiles = []
    numeric_statistics = []

    for position, column in enumerate(dataframe.columns):
        series = dataframe.iloc[:, position]
        missing_count = int(series.isna().sum())
        missing_percentage = (
            round((missing_count / row_count) * 100, 2) if row_count else 0.0
        )

        column_profiles.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            }
        )

        if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
            numeric_statistics.append(
                {
                    "column": str(column),
                    "mean": _json_number(series.mean()),
                    "minimum": _json_number(series.min()),
                    "maximum": _json_number(series.max()),
                    "standard_deviation": _json_number(series.std()),
                }
            )

    return {
        "rows": int(row_count),
        "columns": int(column_count),
        "column_profiles": column_profiles,
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numeric_statistics": numeric_statistics,
    }


def preprocess_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = dataframe.copy(deep=True)
    operations: list[dict[str, Any]] = []
    missing_before = int(cleaned.isna().sum().sum())

    original_names = [str(column) for column in cleaned.columns]
    cleaned.columns = _standardized_column_names(cleaned.columns)
    renamed_columns = [
        {"from": original, "to": cleaned_name}
        for original, cleaned_name in zip(original_names, cleaned.columns)
        if original != cleaned_name
    ]
    operations.append(
        {
            "operation": "standardize_column_names",
            "affected": len(renamed_columns),
            "description": (
                f"Trimmed or disambiguated {len(renamed_columns)} column names."
            ),
        }
    )

    for column in cleaned.columns:
        series = cleaned[column]
        if not _looks_like_date_column(column) or is_datetime64_any_dtype(series.dtype):
            continue

        non_null_count = int(series.notna().sum())
        if non_null_count == 0:
            continue

        converted = pd.to_datetime(series, errors="coerce", format="mixed")
        parsed_count = int(converted[series.notna()].notna().sum())
        invalid_count = non_null_count - parsed_count

        if parsed_count / non_null_count >= DATE_PARSE_THRESHOLD:
            cleaned[column] = converted
            operations.append(
                {
                    "operation": "convert_dates",
                    "column": column,
                    "affected": parsed_count,
                    "description": (
                        f"Converted {parsed_count} values to datetime; "
                        f"{invalid_count} invalid values became missing."
                    ),
                }
            )

    for column in cleaned.columns:
        series = cleaned[column]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        if is_datetime64_any_dtype(series.dtype):
            operations.append(
                {
                    "operation": "preserve_missing_dates",
                    "column": column,
                    "affected": missing_count,
                    "description": (
                        f"Left {missing_count} missing date values unchanged to avoid "
                        "inventing timestamps."
                    ),
                }
            )
            continue

        if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
            fill_value = series.median()
            method = "median"
        else:
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else None
            method = "mode"

        if fill_value is None or pd.isna(fill_value):
            operations.append(
                {
                    "operation": "preserve_all_missing_column",
                    "column": column,
                    "affected": missing_count,
                    "description": (
                        f"Left {missing_count} values missing because the column has "
                        "no usable fill value."
                    ),
                }
            )
            continue

        cleaned[column] = series.fillna(fill_value)
        operations.append(
            {
                "operation": "fill_missing_values",
                "column": column,
                "affected": missing_count,
                "description": (
                    f"Filled {missing_count} missing values using the {method} "
                    f"({_display_value(fill_value)})."
                ),
            }
        )

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    operations.append(
        {
            "operation": "remove_duplicates",
            "affected": duplicate_count,
            "description": f"Removed {duplicate_count} exact duplicate rows.",
        }
    )

    summary = {
        "original_rows": int(dataframe.shape[0]),
        "original_columns": int(dataframe.shape[1]),
        "cleaned_rows": int(cleaned.shape[0]),
        "cleaned_columns": int(cleaned.shape[1]),
        "missing_values_before": missing_before,
        "missing_values_after": int(cleaned.isna().sum().sum()),
        "duplicate_rows_removed": duplicate_count,
        "operations": operations,
    }

    return cleaned, summary
