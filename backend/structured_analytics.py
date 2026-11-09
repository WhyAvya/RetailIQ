"""Validated Pandas execution for deterministic RetailIQ query plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.query_planner import NumericConstraint, QueryPlan


class StructuredExecutionError(ValueError):
    """Raised when a plan is invalid or cannot be executed safely."""


DIMENSION_COLUMNS = {
    "region": "Region",
    "category": "Category",
    "sub_category": "Sub-Category",
    "state": "State",
    "city": "City",
    "segment": "Segment",
    "ship_mode": "Ship Mode",
    "customer_name": "Customer Name",
}
METRIC_COLUMNS = {
    "sales": "Sales", "profit": "Profit", "quantity": "Quantity", "discount": "Discount",
    "shipping_days": "shipping_days",
}
ALLOWED_GROUP_COLUMNS = {
    "Region", "State", "Category", "Sub-Category", "order_year", "Customer ID", "Order ID"
}


@dataclass(frozen=True)
class StructuredResult:
    plan: QueryPlan
    rows_before: int
    rows_after_filtering: int
    entity_count: int
    value: int | float | None = None
    records: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "rows_before": self.rows_before,
            "rows_after_filtering": self.rows_after_filtering,
            "entity_count": self.entity_count,
            "value": self.value,
            "records": list(self.records),
        }


def _apply_numeric_constraint(
    values: pd.Series, constraint: NumericConstraint
) -> pd.Series:
    if constraint.operator == "eq":
        return values.eq(constraint.value)
    if constraint.operator == "gt":
        return values > constraint.value
    if constraint.operator == "gte":
        return values >= constraint.value
    if constraint.operator == "lt":
        return values < constraint.value
    if constraint.operator == "lte":
        return values <= constraint.value
    if constraint.operator == "between" and constraint.upper_value is not None:
        return values.between(constraint.value, constraint.upper_value, inclusive="both")
    raise StructuredExecutionError("The numeric constraint is invalid.")


def apply_plan_filters(dataframe: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    """Apply only allowlisted filters from a validated plan."""
    if plan.status != "valid":
        raise StructuredExecutionError(
            "Only valid query plans can be executed: " + "; ".join(plan.issues)
        )
    mask = pd.Series(True, index=dataframe.index, dtype=bool)
    filters = plan.filters
    for field, expected in (
        ("region", filters.region),
        ("category", filters.category),
        ("sub_category", filters.sub_category),
        ("state", filters.state),
        ("city", filters.city),
        ("segment", filters.segment),
        ("ship_mode", filters.ship_mode),
        ("customer_name", filters.customer_name),
    ):
        if expected is not None:
            column = DIMENSION_COLUMNS[field]
            if column not in dataframe:
                raise StructuredExecutionError(f"Required column {column} is unavailable.")
            mask &= dataframe[column].astype("string").str.casefold().eq(expected.casefold()).fillna(False)
    if filters.start_date or filters.end_date:
        if "Order Date" not in dataframe:
            raise StructuredExecutionError("Required column Order Date is unavailable.")
        dates = pd.to_datetime(dataframe["Order Date"], errors="coerce", format="mixed").dt.date
        mask &= dates.notna()
        if filters.start_date:
            mask &= dates.ge(filters.start_date).fillna(False)
        if filters.end_date:
            mask &= dates.le(filters.end_date).fillna(False)
    for constraint in filters.numeric:
        column = METRIC_COLUMNS[constraint.metric]
        if constraint.metric == "shipping_days":
            missing = {"Order Date", "Ship Date"} - set(dataframe.columns)
            if missing:
                raise StructuredExecutionError(
                    "Required shipping date columns are unavailable: " + ", ".join(sorted(missing))
                )
            order_dates = pd.to_datetime(dataframe["Order Date"], errors="coerce", format="mixed")
            ship_dates = pd.to_datetime(dataframe["Ship Date"], errors="coerce", format="mixed")
            values = (ship_dates - order_dates).dt.days
            mask &= values.notna() & _apply_numeric_constraint(values, constraint)
            continue
        if column not in dataframe:
            raise StructuredExecutionError(f"Required column {column} is unavailable.")
        values = pd.to_numeric(dataframe[column], errors="coerce")
        mask &= values.notna() & _apply_numeric_constraint(values, constraint)
    if filters.product_name_query:
        if "Product Name" not in dataframe:
            raise StructuredExecutionError("Required column Product Name is unavailable.")
        mask &= dataframe["Product Name"].astype("string").str.contains(
            filters.product_name_query, case=False, regex=False, na=False
        )
    return dataframe.loc[mask].copy()


def _entity_frame(filtered: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    frame = filtered.copy()
    frame["source_row"] = frame.index.astype(int)
    if plan.grain in (None, "row"):
        return frame
    aggregations: dict[str, tuple[str, Any]] = {
        column: (column, "sum")
        for column in ("Sales", "Profit", "Quantity")
        if column in frame
    }
    aggregations["source_rows"] = (
        "source_row",
        lambda values: [int(value) for value in values],
    )
    if plan.grain == "order":
        return (
            frame.groupby("Order ID", as_index=False, sort=False)
            .agg(**aggregations)
        )
    if plan.grain == "product":
        product_keys = (
            ["Product ID", "Product Name"]
            if "Product ID" in frame
            else ["Product Name"]
        )
        return (
            frame.groupby(product_keys, as_index=False, sort=False)
            .agg(**aggregations)
        )
    if plan.grain == "customer":
        return (
            frame.groupby("Customer ID", as_index=False, sort=False)
            .agg(**aggregations)
        )
    raise StructuredExecutionError("The requested analysis grain is unsupported.")


def _safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    return tuple(
        {str(key): _safe_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    )


def _entities_within_group(
    filtered: pd.DataFrame, plan: QueryPlan, group_column: str
) -> pd.DataFrame:
    if plan.grain in (None, "row"):
        return filtered.copy()
    if plan.grain == "order":
        keys = list(dict.fromkeys((group_column, "Order ID")))
    elif plan.grain == "product":
        keys = [
            group_column,
            "Product ID" if "Product ID" in filtered else "Product Name",
        ]
    elif plan.grain == "customer":
        keys = list(dict.fromkeys((group_column, "Customer ID")))
    else:
        raise StructuredExecutionError("The requested grouped grain is unsupported.")
    aggregations = {
        column: (column, "sum")
        for column in ("Sales", "Profit", "Quantity")
        if column in filtered
    }
    return (
        filtered.groupby(keys, as_index=False, sort=False)
        .agg(**aggregations)
    )


def execute_plan(dataframe: pd.DataFrame, plan: QueryPlan) -> StructuredResult:
    """Execute an allowlisted plan; never evaluate code or SQL text."""
    filtered = apply_plan_filters(dataframe, plan)
    entities = _entity_frame(filtered, plan)
    metric_column = METRIC_COLUMNS.get(plan.metric) if plan.metric else None
    common = {
        "plan": plan,
        "rows_before": int(len(dataframe)),
        "rows_after_filtering": int(len(filtered)),
        "entity_count": int(len(entities)),
    }
    if plan.operation == "semantic":
        raise StructuredExecutionError("Pure semantic plans must use semantic retrieval.")
    if plan.operation == "records":
        return StructuredResult(**common, records=_records(entities))
    if plan.operation == "ranking":
        ranking_column = METRIC_COLUMNS.get(plan.ranking_metric or plan.metric)
        if plan.ranking_group_by:
            ranked = (
                filtered.groupby(plan.ranking_group_by, as_index=False)[ranking_column]
                .sum()
                .sort_values(ranking_column, ascending=not plan.descending, kind="stable")
                .head(plan.limit or 1)
            )
            if plan.include_distinct_order_count:
                if "Order ID" not in filtered:
                    raise StructuredExecutionError("Required column Order ID is unavailable.")
                order_counts = filtered.groupby(plan.ranking_group_by)["Order ID"].nunique()
                ranked["distinct_order_count"] = ranked[plan.ranking_group_by].map(order_counts)
                if plan.ranking_group_by == "Customer ID" and "Customer Name" in filtered:
                    customer_names = (
                        filtered.drop_duplicates("Customer ID")
                        .set_index("Customer ID")["Customer Name"]
                    )
                    ranked["Customer Name"] = ranked["Customer ID"].map(customer_names)
            if plan.include_row_ranking:
                group_records = [
                    {**record, "result_kind": "group_ranking"}
                    for record in _records(ranked)
                ]
                row_ranked = entities.sort_values(
                    ranking_column, ascending=not plan.descending, kind="stable"
                ).head(plan.limit or 1)
                row_records = [
                    {**record, "result_kind": "row_ranking"}
                    for record in _records(row_ranked)
                ]
                return StructuredResult(
                    **common, records=tuple(group_records + row_records)
                )
            return StructuredResult(**common, records=_records(ranked))
        if ranking_column is None or ranking_column not in entities:
            raise StructuredExecutionError("Ranking metric is unavailable.")
        ranked = entities.sort_values(
            [ranking_column], ascending=[not plan.descending], kind="stable"
        ).head(plan.limit or 1)
        return StructuredResult(**common, records=_records(ranked))
    if plan.group_by:
        if plan.group_by not in ALLOWED_GROUP_COLUMNS:
            raise StructuredExecutionError("The grouping field is not allowlisted.")
        grouped_source = filtered.copy()
        if plan.group_by == "order_year":
            grouped_source["order_year"] = pd.to_datetime(
                grouped_source["Order Date"], format="mixed", errors="coerce"
            ).dt.year
        grouped_entities = _entities_within_group(grouped_source, plan, plan.group_by)
        if plan.operation == "count":
            grouped = grouped_entities.groupby(plan.group_by, dropna=False).size().reset_index(name="value")
        elif plan.operation == "profit_margin":
            grouped = grouped_source.groupby(plan.group_by, dropna=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
            grouped["value"] = grouped["Profit"] / grouped["Sales"] * 100
        else:
            if metric_column is None:
                raise StructuredExecutionError("An aggregate metric is required.")
            function = {"sum": "sum", "mean": "mean", "median": "median"}[plan.operation]
            grouped = grouped_entities.groupby(plan.group_by, dropna=False)[metric_column].agg(function).reset_index(name="value")
        return StructuredResult(**common, records=_records(grouped))
    ranked_records: tuple[dict[str, Any], ...] = ()
    if plan.ranking_metric:
        ranking_column = METRIC_COLUMNS[plan.ranking_metric]
        if plan.ranking_group_by:
            ranked_frame = (
                filtered.groupby(plan.ranking_group_by, as_index=False)[ranking_column]
                .sum()
                .sort_values(ranking_column, ascending=not plan.descending, kind="stable")
                .head(plan.limit or 1)
            )
        else:
            ranked_frame = entities.sort_values(
                ranking_column, ascending=not plan.descending, kind="stable"
            ).head(plan.limit or 1)
        ranked_records = _records(ranked_frame)
    if plan.operation == "count":
        return StructuredResult(**common, value=int(len(entities)), records=ranked_records)
    if plan.operation == "profit_margin":
        sales = float(filtered["Sales"].sum())
        value = float(filtered["Profit"].sum() / sales * 100) if sales else None
        return StructuredResult(**common, value=value, records=ranked_records)
    if plan.operation in {"sum", "mean", "median"}:
        if metric_column is None or metric_column not in entities:
            raise StructuredExecutionError("Aggregate metric is unavailable.")
        function = {"sum": "sum", "mean": "mean", "median": "median"}[plan.operation]
        if plan.aggregate_ranked_group and ranked_records:
            value = ranked_records[0][metric_column]
        else:
            value = getattr(entities[metric_column], function)()
        return StructuredResult(**common, value=float(value), records=ranked_records)
    raise StructuredExecutionError("The planned operation is unsupported.")
