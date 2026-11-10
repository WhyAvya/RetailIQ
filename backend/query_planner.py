"""Deterministic allowlisted planning for RetailIQ questions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal, Mapping, Sequence


PlanStatus = Literal["valid", "ambiguous", "unsupported"]
QueryType = Literal["semantic", "filter", "numerical", "ranking", "mixed"]
Grain = Literal["row", "order", "product", "customer"]
Operation = Literal["semantic", "records", "count", "sum", "mean", "median", "profit_margin", "ranking"]

REGIONS = ("Central", "East", "North", "South", "West")
CATEGORIES = ("Furniture", "Office Supplies", "Technology")
SUB_CATEGORIES = (
    "Accessories", "Appliances", "Art", "Binders", "Bookcases", "Chairs",
    "Copiers", "Envelopes", "Fasteners", "Furnishings", "Labels", "Machines",
    "Paper", "Phones", "Storage", "Supplies", "Tables",
)
SUB_CATEGORY_ALIASES = {
    "accessory": "Accessories", "appliance": "Appliances", "binder": "Binders",
    "bookcase": "Bookcases", "chair": "Chairs", "copier": "Copiers",
    "envelope": "Envelopes", "fastener": "Fasteners", "furnishing": "Furnishings",
    "label": "Labels", "machine": "Machines", "phone": "Phones", "table": "Tables",
}
SEGMENTS = ("Consumer", "Corporate", "Home Office")
SHIP_MODES = ("Standard Class", "Second Class", "First Class", "Same Day")
GROUP_FIELDS = {
    "region": "Region",
    "state": "State",
    "category": "Category",
    "categories": "Category",
    "product category": "Category",
    "product categories": "Category",
    "sub-category": "Sub-Category",
    "sub-categories": "Sub-Category",
    "subcategory": "Sub-Category",
    "subcategories": "Sub-Category",
    "product sub-category": "Sub-Category",
    "product sub-categories": "Sub-Category",
    "year": "order_year",
    "order year": "order_year",
    "customer": "Customer ID",
    "order": "Order ID",
}


@dataclass(frozen=True)
class NumericConstraint:
    metric: Literal["sales", "profit", "quantity", "discount", "shipping_days"]
    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value: float
    upper_value: float | None = None


@dataclass(frozen=True)
class QueryFilters:
    region: str | None = None
    category: str | None = None
    sub_category: str | None = None
    state: str | None = None
    city: str | None = None
    segment: str | None = None
    ship_mode: str | None = None
    customer_name: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    numeric: tuple[NumericConstraint, ...] = ()
    product_name_query: str | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.region,
                self.category,
                self.sub_category,
                self.state,
                self.city,
                self.segment,
                self.ship_mode,
                self.customer_name,
                self.start_date,
                self.end_date,
                self.numeric,
                self.product_name_query,
            )
        )


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    status: PlanStatus
    query_type: QueryType
    operation: Operation
    grain: Grain | None
    filters: QueryFilters = field(default_factory=QueryFilters)
    metric: Literal["sales", "profit", "quantity"] | None = None
    group_by: str | None = None
    limit: int | None = None
    descending: bool = True
    semantic_query: str | None = None
    ranking_metric: Literal["sales", "profit", "quantity"] | None = None
    ranking_group_by: str | None = None
    aggregate_ranked_group: bool = False
    include_row_ranking: bool = False
    include_distinct_order_count: bool = False
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        filters = value["filters"]
        for key in ("start_date", "end_date"):
            if filters[key] is not None:
                filters[key] = filters[key].isoformat()
        return value


def _find_value(question: str, values: tuple[str, ...]) -> str | None:
    normalized = question.casefold()
    matches = [value for value in values if re.search(rf"\b{re.escape(value.casefold())}\b", normalized)]
    return matches[0] if len(matches) == 1 else None


def _find_dynamic_value(question: str, values: Sequence[str]) -> str | None:
    normalized = question.casefold()
    matches = sorted(
        {
            str(value)
            for value in values
            if re.search(rf"\b{re.escape(str(value).casefold())}\b", normalized)
        },
        key=len,
        reverse=True,
    )
    return matches[0] if matches else None


def _occurs_only_within(question: str, shorter: str, longer: str) -> bool:
    """Return true when every shorter location match is part of the longer match."""
    normalized = question.casefold()
    shorter_spans = [
        match.span()
        for match in re.finditer(rf"\b{re.escape(shorter.casefold())}\b", normalized)
    ]
    longer_spans = [
        match.span()
        for match in re.finditer(rf"\b{re.escape(longer.casefold())}\b", normalized)
    ]
    return bool(shorter_spans and longer_spans) and all(
        any(long_start <= start and end <= long_end for long_start, long_end in longer_spans)
        for start, end in shorter_spans
    )


def _date_range(question: str) -> tuple[date | None, date | None, list[str]]:
    issues: list[str] = []
    iso = re.search(
        r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b",
        question,
        re.IGNORECASE,
    )
    if iso:
        try:
            start, end = date.fromisoformat(iso.group(1)), date.fromisoformat(iso.group(2))
        except ValueError:
            return None, None, ["The requested date range is invalid."]
        if start > end:
            issues.append("The start date must not be after the end date.")
        return start, end, issues
    quarter = re.search(r"\bq([1-4])\s+(20\d{2})\b", question, re.IGNORECASE)
    if not quarter:
        ordinal_quarter = re.search(
            r"\b(first|second|third|fourth)\s+quarter(?:\s+of)?\s+(20\d{2})\b",
            question,
            re.IGNORECASE,
        )
        if ordinal_quarter:
            quarter_number = {"first": 1, "second": 2, "third": 3, "fourth": 4}[
                ordinal_quarter.group(1).lower()
            ]
            quarter = (quarter_number, int(ordinal_quarter.group(2)))
    if quarter:
        if isinstance(quarter, tuple):
            quarter_number, year = quarter
        else:
            quarter_number, year = int(quarter.group(1)), int(quarter.group(2))
        start_month = (quarter_number - 1) * 3 + 1
        end_month = start_month + 2
        next_month = date(year + (end_month == 12), 1 if end_month == 12 else end_month + 1, 1)
        return date(year, start_month, 1), date.fromordinal(next_month.toordinal() - 1), issues
    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
        question,
        re.IGNORECASE,
    )
    if month_match:
        months = {
            name: number
            for number, name in enumerate(
                ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"),
                1,
            )
        }
        month, year = months[month_match.group(1).lower()], int(month_match.group(2))
        next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        return date(year, month, 1), date.fromordinal(next_month.toordinal() - 1), issues
    year_match = re.search(r"\b(20\d{2})\b", question)
    if year_match:
        year = int(year_match.group(1))
        return date(year, 1, 1), date(year, 12, 31), issues
    return None, None, issues


def _numeric_constraints(question: str) -> tuple[NumericConstraint, ...]:
    constraints: list[NumericConstraint] = []
    for match in re.finditer(
        r"\b(sales|profit|quantity|discount)\s+(?:is\s+|of\s+)?between\s+\$?([\d,]+(?:\.\d+)?)(%|\s+percent)?\s+and\s+\$?([\d,]+(?:\.\d+)?)(%|\s+percent)?",
        question,
        re.IGNORECASE,
    ):
        constraints.append(
            NumericConstraint(
                match.group(1).lower(),
                "between",
                float(match.group(2).replace(",", "")) / (100 if match.group(1).lower() == "discount" and match.group(3) else 1),
                float(match.group(4).replace(",", "")) / (100 if match.group(1).lower() == "discount" and match.group(5) else 1),
            )
        )
    comparator_map = {
        "at least": "gte", ">=": "gte", "more than": "gt", "greater than": "gt",
        "over": "gt", "above": "gt", "at most": "lte", "<=": "lte",
        "less than": "lt", "under": "lt", "below": "lt",
    }
    for match in re.finditer(
        r"\b(sales|profit|quantity|discount)\s+(?:is\s+|of\s+|(?:rate\s+)?(?:was\s+|were\s+)?)?(at least|more than|greater than|over|above|at most|less than|under|below|>=|<=)\s*\$?([\d,]+(?:\.\d+)?)(%|\s+percent)?",
        question,
        re.IGNORECASE,
    ):
        constraints.append(
            NumericConstraint(
                match.group(1).lower(),
                comparator_map[match.group(2).lower()],
                float(match.group(3).replace(",", "")) / (100 if match.group(1).lower() == "discount" and match.group(4) else 1),
            )
        )
    lower = question.casefold()
    discounted_threshold = re.search(
        r"\bdiscounted\s+(?:by\s+)?(at least|more than|greater than|over|above|at most|less than|under|below|>=|<=)\s*"
        r"([\d,]+(?:\.\d+)?)(%|\s+percent)?",
        lower,
    )
    if discounted_threshold and not any(item.metric == "discount" for item in constraints):
        value = float(discounted_threshold.group(2).replace(",", ""))
        if discounted_threshold.group(3):
            value /= 100
        constraints.append(
            NumericConstraint(
                "discount", comparator_map[discounted_threshold.group(1)], value
            )
        )
    if re.search(r"\b(?:no|zero)\s+discount\b|\bwithout\s+(?:a\s+)?discount\b", lower):
        constraints.append(NumericConstraint("discount", "eq", 0.0))
    elif re.search(r"\bdiscounted\b|\bwith\s+(?:a\s+)?discount\b", lower) and not any(
        item.metric == "discount" for item in constraints
    ):
        constraints.append(NumericConstraint("discount", "gt", 0.0))
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    shipping_match = re.search(
        r"\b(?:shipping|ship(?:ping)? duration|took|taking)\b.*?"
        r"(at least|more than|greater than|over|above|at most|less than|under|below|>=|<=)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:calendar\s+)?days?\b",
        lower,
    )
    if shipping_match:
        token = shipping_match.group(2)
        constraints.append(
            NumericConstraint(
                "shipping_days",
                comparator_map[shipping_match.group(1)],
                float(token) if token.isdigit() else float(number_words[token]),
            )
        )
    if (
        re.search(r"\bprofitable\b", lower)
        and not re.search(r"\bwere they profitable\b", lower)
        and not any(item.metric == "profit" for item in constraints)
    ):
        constraints.append(NumericConstraint("profit", "gt", 0.0))
    if re.search(r"\b(loss-making|loss making|made a loss)\b", lower) and not any(item.metric == "profit" for item in constraints):
        constraints.append(NumericConstraint("profit", "lt", 0.0))
    return tuple(constraints)


def plan_query(
    question: str,
    dimension_values: Mapping[str, Sequence[str]] | None = None,
) -> QueryPlan:
    if not isinstance(question, str) or not question.strip():
        return QueryPlan(str(question), "unsupported", "semantic", "semantic", None, issues=("A non-empty question is required.",))
    text = re.sub(r"\s+", " ", question.strip())
    lower = text.casefold()
    if re.search(r"\b(forecast|predict|prediction|correlation|caused|why|trend)\b", lower):
        return QueryPlan(text, "unsupported", "semantic", "semantic", None, issues=("Predictive, causal, and correlation queries are not supported.",))

    repeated_customer_semantic = bool(
        re.search(r"\bcustomers?\b.*\bmore than one\b.*\brows?\b", lower)
    )
    secondary_customer_order_count = bool(
        re.search(r"\b(?:how many|number of)\s+(?:distinct|unique)?\s*orders?\b", lower)
        and re.search(r"\b(?:who|customer)\b.*\b(?:highest|largest|most|lowest|least)\b", lower)
    )
    explicit_order = bool(
        re.search(r"\b(?:distinct |unique )?orders?\b|\border value\b|\bfor each order\b", lower)
    ) and not re.search(r"\border (?:date|year)\b", lower) and not secondary_customer_order_count
    explicit_row = bool(re.search(r"\b(rows?|records?|line items?)\b", lower)) and not explicit_order and not repeated_customer_semantic
    explicit_product = bool(
        re.search(
            r"\bproducts?\b(?!\s+(?:categor(?:y|ies)|sub-categor(?:y|ies)|subcategor(?:y|ies)))",
            lower,
        )
    ) and not explicit_row
    explicit_customer = bool(
        re.search(
            r"\b(?:distinct|unique)\s+customers?\b|\b(?:how many|count of)\s+customers?\b"
            r"|\b(?:which|what|top)\s+(?:\d+\s+)?customers?\b|\bwho\b.*\bcustomers?\b",
            lower,
        )
    )
    grains = [name for name, present in (("row", explicit_row), ("order", explicit_order), ("product", explicit_product), ("customer", explicit_customer)) if present]
    issues: list[str] = []
    grain: Grain | None = grains[0] if len(grains) == 1 else None
    if len(grains) > 1:
        issues.append("The question specifies more than one analysis grain.")

    dimension_values = dimension_values or {}
    region = _find_value(text, REGIONS) or _find_dynamic_value(
        text, dimension_values.get("region", ())
    )
    category = _find_value(text, CATEGORIES)
    sub_category = _find_value(text, SUB_CATEGORIES)
    if sub_category is None:
        for alias, canonical in SUB_CATEGORY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                sub_category = canonical
                break
    if (
        category == "Office Supplies"
        and sub_category == "Supplies"
        and not re.search(r"\bsub[ -]?category\s+supplies\b", lower)
    ):
        sub_category = None
    state = _find_dynamic_value(text, dimension_values.get("state", ()))
    city = _find_dynamic_value(text, dimension_values.get("city", ()))
    if state and city:
        if len(state) > len(city) and _occurs_only_within(text, city, state):
            city = None
        elif len(city) > len(state) and _occurs_only_within(text, state, city):
            state = None
    segment = _find_value(text, SEGMENTS)
    ship_mode = _find_value(text, SHIP_MODES)
    customer_name = _find_dynamic_value(text, dimension_values.get("customer_name", ()))
    customer_candidate = re.search(
        r"\bcustomer\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text
    )
    if customer_candidate and customer_name is None:
        issues.append("The customer name must be validated against the selected dataset.")
    structural_context = bool(
        re.search(
            r"\b(segment|ship(?:ping)?|line items?|rows?|records?|how many|count|top|highest|lowest|most)\b",
            lower,
        )
    )
    if not structural_context:
        segment = None
        ship_mode = None
    start_date, end_date, date_issues = _date_range(text)
    issues.extend(date_issues)
    numeric = _numeric_constraints(text)
    product_query_match = re.search(
        r"\b(?:containing|contains|matching)\s+['\"]([^'\"]+)['\"]",
        text,
        re.IGNORECASE,
    )
    product_name_query = product_query_match.group(1).strip() if product_query_match else None
    if product_name_query is None:
        descriptor_match = re.search(
            r"\bwhich\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z][a-z-]*)\s+product\s+line items?\b",
            lower,
        )
        if descriptor_match and descriptor_match.group(1) not in {
            "profitable", "technology", "furniture", "office",
        }:
            product_name_query = descriptor_match.group(1)
    filters = QueryFilters(
        region=region,
        category=category,
        sub_category=sub_category,
        state=state,
        city=city,
        segment=segment,
        ship_mode=ship_mode,
        customer_name=customer_name,
        start_date=start_date,
        end_date=end_date,
        numeric=numeric,
        product_name_query=product_name_query,
    )

    metric: Literal["sales", "profit", "quantity"] | None = None
    for candidate in ("sales", "profit", "quantity"):
        if re.search(rf"\b{candidate}\b", lower):
            metric = candidate  # type: ignore[assignment]
            break
    if metric is None and re.search(r"\brevenue\b", lower):
        metric = "sales"
    if metric is None and "order value" in lower:
        metric = "sales"
    if re.search(r"\bunits?\s+(?:were\s+)?sold\b", lower):
        metric = "quantity"
    mentioned_metric = metric
    group_by = None
    for phrase, column in GROUP_FIELDS.items():
        if re.search(rf"\b(?:by|for each|in each|per)\s+(?:product\s+)?{re.escape(phrase)}s?\b", lower):
            if column == "Order ID" and "order value" in lower:
                continue
            group_by = column
            break

    ranking = bool(
        re.search(r"\b(top|highest|largest|most|lowest|worst)\b", lower)
        or re.search(r"(?<!at )\bleast\b", lower)
    )
    loss_ranking = bool(re.search(r"\b(largest|most|worst)\s+loss(?:es)?\b", lower))
    ranking_metric: Literal["sales", "profit", "quantity"] | None = None
    if ranking:
        ranking_metric_match = re.search(
            r"\b(?:highest|largest|most|lowest|least|worst)\s+(sales|revenue|profit|quantity)\b"
            r"|\bby\s+(sales|revenue|profit|quantity)\b",
            lower,
        )
        if ranking_metric_match:
            ranking_name = next(
                value for value in ranking_metric_match.groups() if value is not None
            )
            ranking_metric = "sales" if ranking_name == "revenue" else ranking_name  # type: ignore[assignment]
        elif loss_ranking:
            ranking_metric = "profit"
        else:
            ranking_metric = mentioned_metric
    ranking_group_by = None
    ranking_group_terms = "|".join(
        re.escape(phrase) for phrase in sorted(GROUP_FIELDS, key=len, reverse=True)
    )
    ranking_group_match = re.search(
        rf"\bwhich\s+({ranking_group_terms})s?\b.*\b(?:most|highest|largest|lowest|least)\b"
        rf"|\btop\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+({ranking_group_terms})s?\b",
        lower,
    )
    if ranking_group_match:
        ranking_phrase = next(value for value in ranking_group_match.groups() if value)
        ranking_group_by = GROUP_FIELDS[ranking_phrase]
    elif re.search(r"\bwho\b.*\b(?:highest|largest|most|lowest|least)\b", lower) and "customer" in lower:
        ranking_group_by = "Customer ID"
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    limit_match = re.search(
        r"\b(?:top|which)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        lower,
    )
    if limit_match:
        token = limit_match.group(1)
        limit = int(token) if token.isdigit() else number_words[token]
    else:
        limit = 1 if ranking else None
    descending = not bool(
        re.search(r"\b(lowest|worst)\b", lower)
        or re.search(r"(?<!at )\bleast\b", lower)
    ) and not loss_ranking
    if limit is not None and not 1 <= limit <= 100:
        issues.append("Top-N must be between 1 and 100.")
    if ranking and ranking_metric is None:
        issues.append("A ranking metric such as sales, profit, or quantity is required.")
    if ranking and grain is None and ranking_group_by is None:
        issues.append("A ranking must specify rows, orders, or products, or request a customer or grouped-dimension ranking.")

    compound_total = bool(
        ranking
        and re.search(
            r"\band\s+(?:(?:what (?:was|is|were|are)\s+)?(?:the\s+)?(?:aggregate\s+)?total\s+"
            r"(?:sales|revenue|profit|quantity|loss(?:es)?)\b|what (?:was|is|were|are)\s+its\s+"
            r"(?:sales|revenue|profit|quantity)\s+total\b)",
            lower,
        )
    )
    if re.search(r"\bprofit margin\b", lower):
        operation: Operation = "profit_margin"
    elif secondary_customer_order_count and ranking:
        operation = "ranking"
    elif re.search(r"\b(how many|count(?: of)?)\b", lower) and not re.search(
        r"\bhow many\s+units?\b", lower
    ):
        operation = "count"
    elif re.search(r"\bmedian\b", lower):
        operation = "median"
    elif re.search(r"\b(average|avg|mean)\b", lower):
        operation = "mean"
    elif compound_total:
        operation = "sum"
    elif ranking:
        operation = "ranking"
    elif re.search(r"\b(total|sum of)\b", lower):
        operation = "sum"
    elif not filters.is_empty:
        operation = "records"
    else:
        operation = "semantic"

    aggregate_metric_match = None
    if operation == "sum":
        aggregate_metric_match = re.search(
            r"\b(?:total|sum of)\s+(sales|revenue|profit|quantity|loss(?:es)?)\b",
            lower,
        )
    elif operation == "mean":
        aggregate_metric_match = re.search(
            r"\b(?:average|avg|mean)\s+(sales|profit|quantity)\b",
            lower,
        )
    elif operation == "median":
        aggregate_metric_match = re.search(
            r"\bmedian\s+(sales|profit|quantity)\b",
            lower,
        )
    if operation == "ranking":
        metric = ranking_metric
    elif aggregate_metric_match:
        aggregate_name = aggregate_metric_match.group(1)
        metric = (
            "profit"
            if aggregate_name.startswith("loss")
            else "sales"
            if aggregate_name == "revenue"
            else aggregate_name
        )  # type: ignore[assignment]
    else:
        metric = mentioned_metric

    if operation in {"sum", "mean", "median"} and metric is None:
        issues.append("The requested calculation must specify sales, profit, or quantity.")
    if operation == "count" and grain is None:
        issues.append("A count must specify rows, line items, orders, products, or customers.")
    if group_by and operation not in {"sum", "mean", "median", "count", "profit_margin"}:
        issues.append("Grouping requires an aggregate calculation.")

    if issues:
        status: PlanStatus = "ambiguous" if all("not supported" not in issue for issue in issues) else "unsupported"
    else:
        status = "valid"
    has_structure = operation != "semantic" or not filters.is_empty or group_by is not None
    if (product_name_query and has_structure) or (ranking and operation != "ranking"):
        query_type: QueryType = "mixed"
    elif operation == "semantic":
        query_type = "semantic"
    elif operation == "ranking":
        query_type = "ranking"
    elif operation in {"sum", "mean", "median", "count", "profit_margin"}:
        query_type = "numerical"
    else:
        query_type = "filter"
    return QueryPlan(
        original_question=text,
        status=status,
        query_type=query_type,
        operation=operation,
        grain=grain,
        filters=filters,
        metric=metric,
        group_by=group_by,
        limit=limit,
        descending=descending,
        semantic_query=product_name_query or (text if query_type == "semantic" else None),
        ranking_metric=ranking_metric,
        ranking_group_by=ranking_group_by,
        aggregate_ranked_group=bool(re.search(r"\bits\s+(?:sales|profit|quantity)\s+total\b", lower)),
        include_row_ranking=bool(
            ranking_group_by
            and re.search(r"\b(?:matching\s+)?line item\b.*\b(?:highest|most|largest|lowest)\b", lower)
        ),
        include_distinct_order_count=bool(
            ranking_group_by == "Customer ID"
            and re.search(r"\b(?:how many|number of)\s+(?:distinct|unique)?\s*orders?\b", lower)
        ),
        issues=tuple(issues),
    )
