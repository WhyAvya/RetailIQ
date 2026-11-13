"""Small allowlisted deterministic query planner for arbitrary CSV schemas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


Operation = Literal["count", "sum", "mean", "min", "max", "ranking", "records"]


@dataclass(frozen=True)
class GenericPlan:
    operation: Operation
    metric: str | None = None
    group_by: str | None = None
    filter_column: str | None = None
    filter_operator: str | None = None
    filter_value: str | float | None = None
    limit: int = 5
    descending: bool = True


def _mentioned_columns(question: str, columns: list[str]) -> list[str]:
    lower = question.casefold()
    return sorted(
        [column for column in columns if re.search(rf"(?<!\w){re.escape(column.casefold())}(?!\w)", lower)],
        key=len,
        reverse=True,
    )


def plan_generic_query(dataframe: pd.DataFrame, question: str) -> tuple[GenericPlan | None, str | None]:
    columns = [str(column) for column in dataframe.columns]
    mentioned = _mentioned_columns(question, columns)
    lower = question.casefold()
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    metric = next((column for column in mentioned if column in numeric), None)
    group_match = re.search(r"\bby\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    group_by = None
    if group_match:
        group_candidates = _mentioned_columns(group_match.group(1), columns)
        group_by = group_candidates[0] if group_candidates else None

    operation: Operation | None = None
    if re.search(r"\b(how many|count|number of rows|total rows)\b", lower): operation = "count"
    elif re.search(r"\b(average|mean)\b", lower): operation = "mean"
    elif re.search(r"\b(sum|total)\b", lower): operation = "sum"
    elif re.search(r"\b(highest|maximum|max|top\b)", lower): operation = "ranking" if re.search(r"\b(top|which|who)\b", lower) else "max"
    elif re.search(r"\b(lowest|minimum|min|bottom)\b", lower): operation = "ranking" if re.search(r"\b(bottom|which|who)\b", lower) else "min"
    elif re.search(r"\b(show|list|which)\b", lower): operation = "records"
    if operation is None:
        return None, None
    if operation in {"sum", "mean", "min", "max", "ranking"} and metric is None:
        return None, "Please name the numeric column to calculate."

    filter_column = filter_operator = None
    filter_value: str | float | None = None
    comparator = re.search(r"(.+?)\s+(below|under|less than|above|over|greater than|at least|at most)\s+(-?[\d,.]+)", question, re.IGNORECASE)
    if comparator:
        candidates = _mentioned_columns(comparator.group(1), numeric)
        if candidates:
            filter_column = candidates[0]
            filter_operator = {"below": "lt", "under": "lt", "less than": "lt", "above": "gt", "over": "gt", "greater than": "gt", "at least": "gte", "at most": "lte"}[comparator.group(2).casefold()]
            filter_value = float(comparator.group(3).replace(",", ""))
    if filter_column is None:
        for column in mentioned:
            if column in numeric:
                continue
            values = dataframe[column].dropna().astype(str).unique()
            matches = [value for value in values if re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", lower)]
            if len(matches) == 1:
                filter_column, filter_operator, filter_value = column, "eq", matches[0]
                break
    if operation == "ranking":
        group_by = next((column for column in mentioned if column not in numeric), group_by)
    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    limit_match = re.search(r"\b(?:top|bottom)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", lower)
    limit = min(int(limit_match.group(1)) if limit_match and limit_match.group(1).isdigit() else number_words.get(limit_match.group(1), 5) if limit_match else (1 if "highest" in lower or "lowest" in lower else 5), 50)
    return GenericPlan(operation, metric, group_by, filter_column, filter_operator, filter_value, limit, "lowest" not in lower and "bottom" not in lower), None


def execute_generic_query(dataframe: pd.DataFrame, plan: GenericPlan) -> tuple[str, dict[str, Any]]:
    frame = dataframe.copy()
    if plan.filter_column and plan.filter_operator and isinstance(plan.filter_value, float):
        values = pd.to_numeric(frame[plan.filter_column], errors="coerce")
        operators = {"lt": values.lt, "lte": values.le, "gt": values.gt, "gte": values.ge}
        frame = frame[operators[plan.filter_operator](plan.filter_value)]
    elif plan.filter_column and plan.filter_operator == "eq":
        frame = frame[frame[plan.filter_column].astype(str).str.casefold() == str(plan.filter_value).casefold()]
    if plan.operation == "count":
        value = int(len(frame))
        return f"The matching row count is {value:,}.", {"operation": "count", "value": value, "plan": plan.__dict__}
    if plan.operation == "records":
        records = frame.head(plan.limit).where(pd.notna(frame), None).to_dict(orient="records")
        return f"Found {len(frame):,} matching rows; showing {len(records):,}.", {"operation": "records", "records": records, "plan": plan.__dict__}
    metric = plan.metric
    assert metric is not None
    values = pd.to_numeric(frame[metric], errors="coerce")
    if plan.group_by:
        work = frame.assign(_metric=values).dropna(subset=["_metric"])
        grouped = work.groupby(plan.group_by, dropna=False)["_metric"].agg(plan.operation if plan.operation != "ranking" else "sum")
        if plan.operation == "ranking": grouped = grouped.sort_values(ascending=not plan.descending).head(plan.limit)
        records = [{plan.group_by: str(key), metric: float(value)} for key, value in grouped.items()]
        return f"Results for {metric} by {plan.group_by}: " + "; ".join(f"{row[plan.group_by]}: {row[metric]:,.2f}" for row in records), {"operation": plan.operation, "records": records, "plan": plan.__dict__}
    if plan.operation == "ranking":
        ranked = frame.assign(_metric=values).dropna(subset=["_metric"]).sort_values("_metric", ascending=not plan.descending).head(plan.limit)
        records = ranked.drop(columns=["_metric"]).to_dict(orient="records")
        return f"The {'highest' if plan.descending else 'lowest'} {metric} value is {float(ranked.iloc[0]['_metric']):,.2f}." if len(ranked) else "No matching rows were found.", {"operation": "ranking", "records": records, "plan": plan.__dict__}
    clean = values.dropna()
    if clean.empty: return f"{metric} has no numeric values to calculate.", {"operation": plan.operation, "value": None, "plan": plan.__dict__}
    value = {"sum": clean.sum, "mean": clean.mean, "min": clean.min, "max": clean.max}[plan.operation]()
    return f"The {plan.operation} {metric} is {float(value):,.2f}.", {"operation": plan.operation, "value": float(value), "column": metric, "plan": plan.__dict__}

