import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from backend.analytics import calculate_kpis
from backend.chart_analytics import (
    calculate_category_performance,
    calculate_sales_by_region,
)
from backend.dataset_selection import SelectedDataset, load_selected_dataset
from backend.llm_service import OpenRouterLLMService
from backend.metadata_filtered_retrieval import retrieve_metadata_filtered_matches
from backend.query_planner import QueryPlan, plan_query
from backend.rag_preparation import MetadataValue
from backend.rag_retrieval import RetrievalMatch, retrieve_semantic_matches
from backend.structured_analytics import StructuredExecutionError, StructuredResult, execute_plan
from backend.upload_metadata import get_upload_metadata

ProcessingMode = Literal["analytics", "semantic_rag", "unsupported"]


class RAGPipelineError(RuntimeError):
    pass


class PipelineDatasetNotFoundError(RAGPipelineError):
    pass


class InvalidPipelineRequestError(RAGPipelineError):
    pass


@dataclass(frozen=True)
class AnswerSource:
    chunk_id: str
    source_row: int | None
    similarity_score: float
    metadata: dict[str, MetadataValue]
    document_excerpt: str


@dataclass(frozen=True)
class RAGAnswer:
    upload_id: str
    question: str
    answer: str
    mode: ProcessingMode
    sources: list[AnswerSource]
    calculation: dict[str, Any] | None = None


INTENT_PATTERNS = (
    ("profit_margin", r"\bprofit\s+margin\b"),
    ("average_order_value", r"\b(?:average|avg)\s+order\s+value\b"),
    ("number_of_orders", r"\b(?:how many|number of|count of|unique|distinct)\s+orders?\b"),
    ("unique_customers", r"\b(?:how many|number of|count of|unique|distinct)\s+customers?\b"),
    ("total_revenue", r"\btotal\s+(?:revenue|sales)\b|\bsum\s+of\s+(?:revenue|sales)\b"),
    ("total_profit", r"\btotal\s+profit\b|\bsum\s+of\s+profit\b"),
    ("highest_revenue_region", r"\b(?:highest|top|most)\b.*\bregion\b.*\b(?:revenue|sales)\b|\bregion\b.*\b(?:highest|top|most)\b.*\b(?:revenue|sales)\b"),
    ("lowest_profit_category", r"\b(?:lowest|least|worst)\b.*\bcategory\b.*\bprofit\b|\bcategory\b.*\b(?:lowest|least|worst)\b.*\bprofit\b"),
)
NUMERICAL_CUE = re.compile(
    r"\b(total|sum|average|avg|how many|number of|count|highest|lowest|least|most|margin|percentage|by region|by category|trend)\b",
    re.IGNORECASE,
)


def route_question(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, normalized):
            if intent in {
                "profit_margin",
                "average_order_value",
                "number_of_orders",
                "unique_customers",
                "total_revenue",
                "total_profit",
            } and re.search(r"\b(in|for|during|between|from|by)\b", normalized):
                return "unsupported_numerical"
            return intent
    return "unsupported_numerical" if NUMERICAL_CUE.search(normalized) else "semantic_rag"


def _format_number(value: int | float, percent: bool = False) -> str:
    if isinstance(value, int):
        formatted = f"{value:,}"
    else:
        formatted = f"{value:,.2f}"
    return f"{formatted}%" if percent else formatted


def _kpi_answer(dataset: SelectedDataset, intent: str) -> RAGAnswer:
    metric = calculate_kpis(dataset.dataframe)[intent]
    if not metric["available"]:
        answer = f"This metric is unavailable: {metric['reason']}"
    else:
        labels = {
            "total_revenue": "Total revenue",
            "total_profit": "Total profit",
            "number_of_orders": "Number of orders",
            "unique_customers": "Unique customers",
            "average_order_value": "Average order value",
            "profit_margin": "Profit margin",
        }
        answer = (
            f"{labels[intent]} is "
            f"{_format_number(metric['value'], intent == 'profit_margin')}."
        )
    return RAGAnswer(
        str(dataset.upload_id),
        "",
        answer,
        "analytics",
        [],
        {"intent": intent, **metric},
    )


def _comparison_answer(dataset: SelectedDataset, intent: str) -> RAGAnswer:
    if intent == "highest_revenue_region":
        result = calculate_sales_by_region(dataset.dataframe)
        if not result["available"] or not result["data"]:
            answer = f"This calculation is unavailable: {result['reason']}"
            calculation = {"intent": intent, "available": False, "reason": result["reason"]}
        else:
            item = result["data"][0]
            answer = f"{item['region']} generated the highest revenue: {_format_number(item['revenue'])}."
            calculation = {"intent": intent, "available": True, **item}
    else:
        result = calculate_category_performance(dataset.dataframe)
        valid = [item for item in result.get("data", []) if item["profit"] is not None]
        if not result["available"] or not valid:
            answer = f"This calculation is unavailable: {result['reason']}"
            calculation = {"intent": intent, "available": False, "reason": result["reason"]}
        else:
            item = min(valid, key=lambda value: value["profit"])
            answer = f"{item['category']} has the lowest profit: {_format_number(item['profit'])}."
            calculation = {"intent": intent, "available": True, **item}
    return RAGAnswer(str(dataset.upload_id), "", answer, "analytics", [], calculation)


def _planner_dimensions(dataset: SelectedDataset) -> dict[str, list[str]]:
    return {
        field: sorted(
            dataset.dataframe[column].dropna().astype(str).unique().tolist(),
            key=str.casefold,
        )
        for field, column in (
            ("region", "Region"),
            ("state", "State"),
            ("city", "City"),
            ("customer_name", "Customer Name"),
        )
        if column in dataset.dataframe
    }


def _number(value: int | float | None) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.2f}"


def _record_summary(record: dict[str, Any]) -> str:
    label = next(
        (
            str(record[key])
            for key in (
                "Product Name", "Customer Name", "Order ID", "Customer ID", "State", "Region", "Category",
                "Sub-Category", "source_row",
            )
            if key in record
        ),
        "result",
    )
    details = [
        f"{key}={_number(value)}"
        for key, value in record.items()
        if key in {"Sales", "Profit", "Quantity", "value", "distinct_order_count"}
        and isinstance(value, (int, float))
    ]
    return f"{label}{' (' + ', '.join(details) + ')' if details else ''}"


def _structured_answer(plan: QueryPlan, result: StructuredResult) -> str:
    labels = {"sales": "sales", "profit": "profit", "quantity": "quantity"}
    if plan.operation == "count":
        base = f"Exact dataset calculation: the matching {plan.grain or 'row'} count is {_number(result.value)}."
    elif plan.operation == "sum":
        if plan.group_by:
            base = f"Exact dataset calculation: grouped total {labels.get(plan.metric, plan.metric or 'value')}."
        else:
            base = f"Exact dataset calculation: total {labels.get(plan.metric, plan.metric or 'value')} is {_number(result.value)}."
    elif plan.operation == "mean":
        base = f"Exact dataset calculation: average {labels.get(plan.metric, plan.metric or 'value')} is {_number(result.value)}."
    elif plan.operation == "median":
        base = f"Exact dataset calculation: median {labels.get(plan.metric, plan.metric or 'value')} is {_number(result.value)}."
    elif plan.operation == "profit_margin":
        base = f"Exact dataset calculation: profit margin is {_number(result.value)}%."
    elif plan.operation == "records":
        base = f"Exact dataset filter: found {result.entity_count:,} matching {plan.grain or 'row'} records."
    else:
        base = "Exact dataset calculation: the requested ranking was calculated."
    if result.records:
        summaries = "; ".join(_record_summary(record) for record in result.records[:10])
        base = f"{base} Results: {summaries}."
    return base


def _legacy_intent(plan: QueryPlan) -> str:
    if plan.operation == "sum" and plan.metric == "sales" and plan.filters.is_empty:
        return "total_revenue"
    if plan.operation == "sum" and plan.metric == "profit" and plan.filters.is_empty:
        return "total_profit"
    if plan.operation == "mean" and plan.grain == "order" and plan.metric == "sales":
        return "average_order_value"
    if plan.operation == "count" and plan.grain == "order" and plan.filters.is_empty:
        return "number_of_orders"
    if plan.operation == "count" and plan.grain == "customer" and plan.filters.is_empty:
        return "unique_customers"
    if plan.operation == "profit_margin" and plan.filters.is_empty:
        return "profit_margin"
    if (
        plan.operation == "ranking"
        and plan.ranking_group_by == "Region"
        and plan.metric == "sales"
        and plan.filters.is_empty
        and not plan.include_row_ranking
    ):
        return "highest_revenue_region"
    if (
        plan.operation == "ranking"
        and plan.ranking_group_by == "Category"
        and plan.metric == "profit"
        and not plan.descending
        and plan.filters.is_empty
        and not plan.include_row_ranking
    ):
        return "lowest_profit_category"
    return "structured_query"


def _structured_calculation(plan: QueryPlan, result: StructuredResult) -> dict[str, Any]:
    records = list(result.records)
    intent = _legacy_intent(plan)
    if intent == "highest_revenue_region" and records:
        return {
            "intent": intent,
            "available": True,
            "region": records[0]["Region"],
            "revenue": records[0]["Sales"],
        }
    if intent == "lowest_profit_category" and records:
        return {
            "intent": intent,
            "available": True,
            "category": records[0]["Category"],
            "profit": records[0]["Profit"],
        }
    calculation: dict[str, Any] = {
        "intent": intent,
        "available": True,
        "plan": plan.as_dict(),
        "rows_before": result.rows_before,
        "rows_after_filtering": result.rows_after_filtering,
        "entity_count": result.entity_count,
        "value": result.value,
        "records": records[:100],
        "records_truncated": len(records) > 100,
    }
    return calculation


def _answer_sources(matches: list[RetrievalMatch]) -> list[AnswerSource]:
    return [
        AnswerSource(
            match.chunk_id,
            match.source_row,
            match.similarity_score,
            match.metadata,
            match.document[:600],
        )
        for match in matches
    ]


def answer_question(
    upload_id: UUID,
    question: str,
    top_k: int = 5,
    metadata_lookup: Callable[..., dict[str, Any] | None] = get_upload_metadata,
    dataset_loader: Callable[[UUID], SelectedDataset] = load_selected_dataset,
    retriever: Callable[..., list[RetrievalMatch]] = retrieve_semantic_matches,
    filtered_retriever: Callable[..., list[RetrievalMatch]] = retrieve_metadata_filtered_matches,
    llm_service: OpenRouterLLMService | None = None,
) -> RAGAnswer:
    if not isinstance(question, str) or not question.strip():
        raise InvalidPipelineRequestError("The question cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise InvalidPipelineRequestError("top_k must be between 1 and 50.")
    metadata = metadata_lookup(upload_id)
    if metadata is None:
        raise PipelineDatasetNotFoundError("The selected dataset was not found.")
    initial_plan = plan_query(question)
    if initial_plan.status == "unsupported":
        return RAGAnswer(
            str(upload_id), question, "This question is not yet supported: " + "; ".join(initial_plan.issues), "unsupported", [], {"issues": list(initial_plan.issues)}
        )
    if initial_plan.operation != "semantic" or initial_plan.status != "valid":
        dataset = dataset_loader(upload_id)
        plan = plan_query(question, _planner_dimensions(dataset))
        if plan.status != "valid":
            return RAGAnswer(
                str(upload_id),
                question,
                "This question is ambiguous or unsupported: " + "; ".join(plan.issues),
                "unsupported",
                [],
                {"plan": plan.as_dict(), "issues": list(plan.issues)},
            )
        try:
            structured = execute_plan(dataset.dataframe, plan)
        except StructuredExecutionError as exc:
            raise RAGPipelineError("Structured analytics could not be completed.") from exc
        matches: list[RetrievalMatch] = []
        has_derived_shipping_filter = any(
            constraint.metric == "shipping_days" for constraint in plan.filters.numeric
        )
        if plan.query_type == "mixed" and plan.semantic_query and not has_derived_shipping_filter:
            matches = filtered_retriever(str(upload_id), plan, top_k)
            if any(match.metadata.get("upload_id") != str(upload_id) for match in matches):
                raise RAGPipelineError("Retrieved sources did not match the selected dataset.")
        return RAGAnswer(
            str(upload_id),
            question,
            _structured_answer(plan, structured),
            "analytics",
            _answer_sources(matches),
            _structured_calculation(plan, structured),
        )

    matches = retriever(upload_id, question, top_k)
    if not matches:
        return RAGAnswer(
            str(upload_id), question, "The retrieved evidence is insufficient to answer this question.", "unsupported", [], None
        )
    if any(match.metadata.get("upload_id") != str(upload_id) for match in matches):
        raise RAGPipelineError("Retrieved sources did not match the selected dataset.")
    service = llm_service or OpenRouterLLMService()
    answer = service.generate_grounded_answer(question, matches)
    sources = _answer_sources(matches)
    return RAGAnswer(str(upload_id), question, answer, "semantic_rag", sources, None)
