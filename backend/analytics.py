import math
import re
from typing import Any

import pandas as pd

COLUMN_ALIASES = {
    "sales": (
        "sales",
        "revenue",
        "total sales",
        "total revenue",
        "sales amount",
        "revenue amount",
    ),
    "profit": (
        "profit",
        "net profit",
        "total profit",
        "profit amount",
    ),
    "order_id": (
        "order id",
        "order number",
        "order no",
        "order identifier",
    ),
    "customer_id": (
        "customer id",
        "customer number",
        "customer no",
        "customer identifier",
    ),
    "order_date": (
        "order date",
        "transaction date",
        "sales date",
        "date",
    ),
    "region": (
        "region",
        "sales region",
        "market region",
        "territory",
        "area",
    ),
    "category": (
        "category",
        "product category",
        "item category",
        "department",
    ),
    "sub_category": (
        "sub category",
        "subcategory",
        "product sub category",
        "product subcategory",
        "item sub category",
    ),
}

METRIC_LABELS = {
    "total_revenue": "Total revenue",
    "total_profit": "Total profit",
    "number_of_orders": "Number of orders",
    "unique_customers": "Unique customers",
    "average_order_value": "Average order value",
    "profit_margin": "Profit margin",
}


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).strip().lower())


def find_column(dataframe: pd.DataFrame, field: str) -> object | None:
    columns_by_name: dict[str, object] = {}
    for column in dataframe.columns:
        columns_by_name.setdefault(_normalize_column_name(column), column)

    for alias in COLUMN_ALIASES[field]:
        column = columns_by_name.get(_normalize_column_name(alias))
        if column is not None:
            return column
    return None


def coerce_numeric_values(values: pd.Series) -> tuple[pd.Series, int]:
    """Coerce a Series to finite numbers and report ignored values."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    finite_values = numeric_values.map(
        lambda value: pd.notna(value) and math.isfinite(value)
    )
    numeric_values = numeric_values.where(finite_values)
    ignored_values = int(len(numeric_values) - numeric_values.notna().sum())
    return numeric_values, ignored_values


def _metric(
    value: int | float | None,
    *,
    available: bool,
    reason: str | None = None,
    source_column: object | None = None,
    ignored_values: int = 0,
) -> dict[str, Any]:
    if hasattr(value, "item"):
        value = value.item()
    if value is not None and not math.isfinite(value):
        value = None
        available = False
        reason = reason or "The calculation did not produce a finite result."
    return {
        "value": value,
        "available": available,
        "reason": reason,
        "source_column": str(source_column) if source_column is not None else None,
        "ignored_values": ignored_values,
    }


def _numeric_total(
    dataframe: pd.DataFrame,
    field: str,
    display_name: str,
) -> tuple[dict[str, Any], int | float | None]:
    column = find_column(dataframe, field)
    if column is None:
        return (
            _metric(
                None,
                available=False,
                reason=f"A {display_name} column was not found.",
            ),
            None,
        )

    numeric_values, ignored_values = coerce_numeric_values(dataframe[column])
    valid_count = int(numeric_values.notna().sum())
    if valid_count == 0:
        return (
            _metric(
                None,
                available=False,
                reason=f"The {column} column has no valid numeric values.",
                source_column=column,
                ignored_values=ignored_values,
            ),
            None,
        )

    value = numeric_values.sum(min_count=1)
    result = value.item() if hasattr(value, "item") else value
    if not math.isfinite(result):
        return (
            _metric(
                None,
                available=False,
                reason=f"The {column} total is outside the supported numeric range.",
                source_column=column,
                ignored_values=ignored_values,
            ),
            None,
        )
    return (
        _metric(
            result,
            available=True,
            source_column=column,
            ignored_values=ignored_values,
        ),
        result,
    )


def _distinct_count(
    dataframe: pd.DataFrame,
    field: str,
    display_name: str,
) -> tuple[dict[str, Any], int | None]:
    column = find_column(dataframe, field)
    if column is None:
        return (
            _metric(
                None,
                available=False,
                reason=f"An {display_name} column was not found.",
            ),
            None,
        )

    values = dataframe[column]
    valid_mask = values.notna() & values.astype("string").str.strip().ne("")
    valid_values = values.loc[valid_mask]
    ignored_values = int(len(values) - len(valid_values))
    if valid_values.empty:
        return (
            _metric(
                None,
                available=False,
                reason=f"The {column} column has no valid identifiers.",
                source_column=column,
                ignored_values=ignored_values,
            ),
            None,
        )

    value = int(valid_values.nunique(dropna=True))
    return (
        _metric(
            value,
            available=True,
            source_column=column,
            ignored_values=ignored_values,
        ),
        value,
    )


def calculate_kpis(dataframe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate retail KPIs without modifying or rounding the source DataFrame."""
    revenue_metric, revenue = _numeric_total(dataframe, "sales", "sales or revenue")
    profit_metric, profit = _numeric_total(dataframe, "profit", "profit")
    orders_metric, order_count = _distinct_count(dataframe, "order_id", "order ID")
    customers_metric, _ = _distinct_count(dataframe, "customer_id", "customer ID")

    if revenue is None or order_count is None:
        average_order_value = _metric(
            None,
            available=False,
            reason="Revenue and a valid order count are required.",
        )
    elif order_count == 0:
        average_order_value = _metric(
            None,
            available=False,
            reason="Average order value is unavailable because the order count is zero.",
        )
    else:
        average_order_value = _metric(revenue / order_count, available=True)

    if revenue is None or profit is None:
        profit_margin = _metric(
            None,
            available=False,
            reason="Revenue and profit are required.",
        )
    elif revenue == 0:
        profit_margin = _metric(
            None,
            available=False,
            reason="Profit margin is unavailable because revenue is zero.",
        )
    else:
        profit_margin = _metric((profit / revenue) * 100, available=True)

    return {
        "total_revenue": revenue_metric,
        "total_profit": profit_metric,
        "number_of_orders": orders_metric,
        "unique_customers": customers_metric,
        "average_order_value": average_order_value,
        "profit_margin": profit_margin,
    }


def build_kpi_warnings(kpis: dict[str, dict[str, Any]]) -> list[str]:
    """Create user-safe warnings for unavailable metrics and ignored values."""
    warnings = []
    for metric_name, metric in kpis.items():
        label = METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())
        ignored_values = int(metric.get("ignored_values", 0))
        if ignored_values:
            warnings.append(
                f"{label}: ignored {ignored_values} null, invalid, or non-finite values."
            )
        if not metric.get("available") and metric.get("reason"):
            warnings.append(f"{label}: {metric['reason']}")
    return warnings
