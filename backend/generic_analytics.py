"""Schema-aware analytics for non-retail CSV files."""

from __future__ import annotations

from typing import Any

import pandas as pd


RETAIL_COLUMNS = {"Sales", "Profit", "Order ID", "Customer ID"}


def is_retail_dataset(dataframe: pd.DataFrame) -> bool:
    return RETAIL_COLUMNS.issubset({str(column) for column in dataframe.columns})


def _safe_number(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def infer_schema(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        kind = "text"
        if pd.api.types.is_bool_dtype(series):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "datetime"
        elif len(non_null):
            date_named = any(word in str(column).casefold() for word in ("date", "time", "year"))
            parsed = pd.to_datetime(non_null, errors="coerce") if date_named else None
            if date_named and parsed is not None and parsed.notna().mean() >= 0.9:
                kind = "datetime"
            elif non_null.nunique() <= min(50, max(10, len(non_null) // 5)):
                kind = "categorical"
        schema.append({
            "name": str(column),
            "type": kind,
            "missing_count": int(series.isna().sum()),
            "unique_count": int(non_null.nunique()),
        })
    return schema


def generic_dashboard(dataframe: pd.DataFrame) -> dict[str, Any]:
    schema = infer_schema(dataframe)
    numeric = [item["name"] for item in schema if item["type"] == "numeric"]
    categorical = [item["name"] for item in schema if item["type"] in {"categorical", "boolean"}]
    numeric_summaries = []
    for column in numeric[:8]:
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            continue
        numeric_summaries.append({
            "column": column,
            "count": int(values.count()),
            "sum": _safe_number(values.sum()),
            "mean": _safe_number(values.mean()),
            "minimum": _safe_number(values.min()),
            "maximum": _safe_number(values.max()),
        })
    distributions = []
    for column in categorical[:4]:
        counts = dataframe[column].fillna("Missing").astype(str).value_counts().head(10)
        distributions.append({
            "column": column,
            "values": [{"label": str(label), "count": int(count)} for label, count in counts.items()],
        })
    suggested = ["How many rows are in this dataset?"]
    if numeric:
        suggested.extend([f"What is the average {numeric[0]}?", f"What is the highest {numeric[0]}?"])
    if numeric and categorical:
        suggested.append(f"What is the average {numeric[0]} by {categorical[0]}?")
    return {
        "kind": "generic",
        "schema": schema,
        "summary": {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "missing_values": int(dataframe.isna().sum().sum()),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "numeric": numeric_summaries,
        },
        "charts": {"categorical_distributions": distributions},
        "suggested_questions": suggested[:5],
    }

