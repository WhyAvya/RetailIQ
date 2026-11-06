import math
from typing import Any

import pandas as pd

from backend.analytics import coerce_numeric_values, find_column


def _unavailable(reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "warnings": warnings or [],
        "data": [],
    }


def calculate_monthly_revenue(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Aggregate finite sales values by calendar month in chronological order."""
    date_column = find_column(dataframe, "order_date")
    sales_column = find_column(dataframe, "sales")
    if date_column is None and sales_column is None:
        return _unavailable("Date and sales columns were not found.")
    if date_column is None:
        return _unavailable("A recognizable date column was not found.")
    if sales_column is None:
        return _unavailable("A sales or revenue column was not found.")

    parsed_dates = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
        format="mixed",
    )
    sales_values, invalid_sales = coerce_numeric_values(dataframe[sales_column])
    invalid_dates = int(len(parsed_dates) - parsed_dates.notna().sum())
    warnings = []
    if invalid_dates:
        warnings.append(f"Ignored {invalid_dates} rows with missing or invalid dates.")
    if invalid_sales:
        warnings.append(
            f"Ignored {invalid_sales} rows with missing, invalid, or non-finite sales values."
        )

    valid_rows = parsed_dates.notna() & sales_values.notna()
    if not valid_rows.any():
        return _unavailable("No valid monthly revenue data is available.", warnings)

    monthly_data = pd.DataFrame(
        {
            "month": parsed_dates.loc[valid_rows].dt.to_period("M"),
            "revenue": sales_values.loc[valid_rows],
        }
    )
    grouped = monthly_data.groupby("month", sort=True)["revenue"].sum(min_count=1)

    data = []
    overflow_months = 0
    for month, revenue in grouped.items():
        value = revenue.item() if hasattr(revenue, "item") else revenue
        if not math.isfinite(value):
            overflow_months += 1
            continue
        data.append(
            {
                "month": month.strftime("%Y-%m"),
                "label": month.strftime("%b %Y"),
                "revenue": value,
            }
        )

    if overflow_months:
        warnings.append(
            f"Ignored {overflow_months} months whose totals exceeded the supported numeric range."
        )
    if not data:
        return _unavailable("No valid monthly revenue data is available.", warnings)

    return {
        "available": True,
        "reason": None,
        "warnings": warnings,
        "data": data,
    }


def calculate_sales_by_region(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Aggregate finite sales values by non-empty region, highest revenue first."""
    region_column = find_column(dataframe, "region")
    sales_column = find_column(dataframe, "sales")
    if region_column is None and sales_column is None:
        return _unavailable("Region and sales columns were not found.")
    if region_column is None:
        return _unavailable("A region column was not found.")
    if sales_column is None:
        return _unavailable("A sales or revenue column was not found.")

    regions = dataframe[region_column].astype("string").str.strip()
    valid_regions = regions.notna() & regions.ne("")
    sales_values, invalid_sales = coerce_numeric_values(dataframe[sales_column])
    invalid_regions = int(len(regions) - valid_regions.sum())
    warnings = []
    if invalid_regions:
        warnings.append(f"Ignored {invalid_regions} rows with missing or empty regions.")
    if invalid_sales:
        warnings.append(
            f"Ignored {invalid_sales} rows with missing, invalid, or non-finite sales values."
        )

    valid_rows = valid_regions & sales_values.notna()
    if not valid_rows.any():
        return _unavailable("No valid regional revenue data is available.", warnings)

    regional_data = pd.DataFrame(
        {
            "region": regions.loc[valid_rows],
            "revenue": sales_values.loc[valid_rows],
        }
    )
    grouped = regional_data.groupby("region", sort=False)["revenue"].sum(min_count=1)
    grouped = grouped.sort_values(ascending=False, kind="stable")

    data = []
    overflow_regions = 0
    for region, revenue in grouped.items():
        value = revenue.item() if hasattr(revenue, "item") else revenue
        if not math.isfinite(value):
            overflow_regions += 1
            continue
        data.append({"region": str(region), "revenue": value})

    if overflow_regions:
        warnings.append(
            f"Ignored {overflow_regions} regions whose totals exceeded the supported numeric range."
        )
    if not data:
        return _unavailable("No valid regional revenue data is available.", warnings)

    return {
        "available": True,
        "reason": None,
        "warnings": warnings,
        "data": data,
    }


def _dimension_values(values: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    dimensions = values.astype("string").str.strip()
    valid_dimensions = dimensions.notna() & dimensions.ne("")
    invalid_dimensions = int(len(dimensions) - valid_dimensions.sum())
    return dimensions, valid_dimensions, invalid_dimensions


def _finite_value(value: object) -> int | float | None:
    if pd.isna(value):
        return None
    result = value.item() if hasattr(value, "item") else value
    return result if math.isfinite(result) else None


def calculate_category_performance(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Aggregate revenue and profit by non-empty product category."""
    category_column = find_column(dataframe, "category")
    sales_column = find_column(dataframe, "sales")
    profit_column = find_column(dataframe, "profit")
    missing = []
    if category_column is None:
        missing.append("category")
    if sales_column is None:
        missing.append("sales or revenue")
    if profit_column is None:
        missing.append("profit")
    if missing:
        return _unavailable(f"Required columns were not found: {', '.join(missing)}.")

    categories, valid_categories, invalid_categories = _dimension_values(
        dataframe[category_column]
    )
    revenue, invalid_sales = coerce_numeric_values(dataframe[sales_column])
    profit, invalid_profit = coerce_numeric_values(dataframe[profit_column])
    warnings = []
    if invalid_categories:
        warnings.append(
            f"Ignored {invalid_categories} rows with missing or empty categories."
        )
    if invalid_sales:
        warnings.append(
            f"Ignored {invalid_sales} missing, invalid, or non-finite sales values."
        )
    if invalid_profit:
        warnings.append(
            f"Ignored {invalid_profit} missing, invalid, or non-finite profit values."
        )

    valid_rows = valid_categories & (revenue.notna() | profit.notna())
    if not valid_rows.any():
        return _unavailable("No valid category performance data is available.", warnings)

    grouped = (
        pd.DataFrame(
            {
                "category": categories.loc[valid_rows],
                "revenue": revenue.loc[valid_rows],
                "profit": profit.loc[valid_rows],
            }
        )
        .groupby("category", sort=False)[["revenue", "profit"]]
        .sum(min_count=1)
        .sort_values("revenue", ascending=False, kind="stable", na_position="last")
    )

    data = []
    incomplete_categories = 0
    for category, row in grouped.iterrows():
        revenue_value = _finite_value(row["revenue"])
        profit_value = _finite_value(row["profit"])
        if revenue_value is None or profit_value is None:
            incomplete_categories += 1
        data.append(
            {
                "category": str(category),
                "revenue": revenue_value,
                "profit": profit_value,
            }
        )

    if incomplete_categories:
        warnings.append(
            f"{incomplete_categories} categories have an unavailable revenue or profit total."
        )
    return {
        "available": True,
        "reason": None,
        "warnings": warnings,
        "data": data,
    }


def calculate_profit_analysis(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Aggregate profit and profit margin using the best available dimension."""
    dimension_type = None
    dimension_column = None
    skipped_dimensions = []
    found_dimension_column = False
    for field in ("sub_category", "category", "region"):
        column = find_column(dataframe, field)
        if column is None:
            continue
        found_dimension_column = True
        _, valid_values, _ = _dimension_values(dataframe[column])
        if valid_values.any():
            dimension_type = field
            dimension_column = column
            break
        skipped_dimensions.append(str(column))

    sales_column = find_column(dataframe, "sales")
    profit_column = find_column(dataframe, "profit")
    missing = []
    if not found_dimension_column:
        missing.append("sub-category, category, or region")
    if sales_column is None:
        missing.append("sales or revenue")
    if profit_column is None:
        missing.append("profit")
    if missing:
        return _unavailable(f"Required columns were not found: {', '.join(missing)}.")
    if dimension_column is None:
        warnings = []
        if skipped_dimensions:
            warnings.append(
                f"Skipped empty dimensions: {', '.join(skipped_dimensions)}."
            )
        return _unavailable("No valid profit analysis data is available.", warnings)

    dimensions, valid_dimensions, invalid_dimensions = _dimension_values(
        dataframe[dimension_column]
    )
    revenue, invalid_sales = coerce_numeric_values(dataframe[sales_column])
    profit, invalid_profit = coerce_numeric_values(dataframe[profit_column])
    warnings = []
    if skipped_dimensions:
        warnings.append(
            f"Skipped empty dimensions: {', '.join(skipped_dimensions)}."
        )
    if invalid_dimensions:
        warnings.append(
            f"Ignored {invalid_dimensions} rows with a missing or empty {dimension_column} value."
        )
    if invalid_sales:
        warnings.append(
            f"Ignored {invalid_sales} missing, invalid, or non-finite sales values."
        )
    if invalid_profit:
        warnings.append(
            f"Ignored {invalid_profit} missing, invalid, or non-finite profit values."
        )

    valid_rows = valid_dimensions & (revenue.notna() | profit.notna())
    if not valid_rows.any():
        return _unavailable("No valid profit analysis data is available.", warnings)

    grouped = (
        pd.DataFrame(
            {
                "segment": dimensions.loc[valid_rows],
                "revenue": revenue.loc[valid_rows],
                "profit": profit.loc[valid_rows],
            }
        )
        .groupby("segment", sort=False)[["revenue", "profit"]]
        .sum(min_count=1)
        .sort_values("profit", ascending=False, kind="stable", na_position="last")
    )

    data = []
    unavailable_margins = 0
    for segment, row in grouped.iterrows():
        revenue_value = _finite_value(row["revenue"])
        profit_value = _finite_value(row["profit"])
        if revenue_value in (None, 0) or profit_value is None:
            profit_margin = None
            unavailable_margins += 1
        else:
            profit_margin = _finite_value((profit_value / revenue_value) * 100)
            if profit_margin is None:
                unavailable_margins += 1

        if profit_value is None:
            status = "unavailable"
        elif profit_value < 0:
            status = "loss-making"
        elif profit_value > 0:
            status = "profitable"
        else:
            status = "break-even"
        data.append(
            {
                "segment": str(segment),
                "revenue": revenue_value,
                "profit": profit_value,
                "profit_margin": profit_margin,
                "status": status,
            }
        )

    if unavailable_margins:
        warnings.append(
            f"Profit margin is unavailable for {unavailable_margins} segments with zero or unavailable revenue/profit."
        )
    return {
        "available": True,
        "reason": None,
        "warnings": warnings,
        "dimension": str(dimension_column),
        "dimension_type": dimension_type,
        "data": data,
    }
