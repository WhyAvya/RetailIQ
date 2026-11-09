"""Separate metadata-filtered Chroma retrieval for Batch 1 C3."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from typing import Any

from backend.embeddings import EmbeddingError, generate_query_embedding
from backend.query_planner import QueryPlan
from backend.rag_retrieval import (
    InvalidRetrievalRequestError,
    RetrievalMatch,
    RetrievalUnavailableError,
)
from backend.vector_store import ChromaVectorStore, VectorStoreUnavailableError


QueryEmbedder = Callable[[str], list[float]]


def _epoch_day(value: date) -> int:
    return (value - date(1970, 1, 1)).days


def build_chroma_where(upload_id: str, plan: QueryPlan) -> dict[str, Any]:
    """Translate only validated allowlisted constraints into Chroma operators."""
    if plan.status != "valid":
        raise InvalidRetrievalRequestError("A valid query plan is required.")
    clauses: list[dict[str, Any]] = [{"upload_id": upload_id}]
    filters = plan.filters
    for key, value in (
        ("region", filters.region),
        ("category", filters.category),
        ("sub_category", filters.sub_category),
        ("state", filters.state),
        ("city", filters.city),
        ("segment", filters.segment),
        ("ship_mode", filters.ship_mode),
        ("customer_name", filters.customer_name),
    ):
        if value is not None:
            clauses.append({key: value})
    if filters.start_date:
        clauses.append({"order_date_epoch_day": {"$gte": _epoch_day(filters.start_date)}})
    if filters.end_date:
        clauses.append({"order_date_epoch_day": {"$lte": _epoch_day(filters.end_date)}})
    for constraint in filters.numeric:
        if constraint.metric == "shipping_days":
            raise InvalidRetrievalRequestError(
                "Shipping duration is derived from two dates and cannot be expressed as a Chroma metadata filter."
            )
        if constraint.operator == "between":
            clauses.extend(
                [
                    {constraint.metric: {"$gte": constraint.value}},
                    {constraint.metric: {"$lte": constraint.upper_value}},
                ]
            )
        else:
            operator = {"eq": "$eq", "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte"}[
                constraint.operator
            ]
            clauses.append({constraint.metric: {operator: constraint.value}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _first(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key)
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")
    return values[0]


def retrieve_metadata_filtered_matches(
    upload_id: str,
    plan: QueryPlan,
    top_k: int = 5,
    vector_store: ChromaVectorStore | None = None,
    embedding_function: QueryEmbedder = generate_query_embedding,
) -> list[RetrievalMatch]:
    """Rank semantic similarity only among rows passing plan metadata filters."""
    upload_id = str(upload_id).strip()
    if not upload_id:
        raise InvalidRetrievalRequestError("upload_id is required.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise InvalidRetrievalRequestError("top_k must be between 1 and 50.")
    where = build_chroma_where(upload_id, plan)
    query = plan.semantic_query or plan.original_question
    try:
        store = vector_store or ChromaVectorStore()
        embedding = embedding_function(query)
        result = store._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except (EmbeddingError, VectorStoreUnavailableError) as exc:
        raise RetrievalUnavailableError("Filtered semantic retrieval is unavailable.") from exc
    ids = _first(result, "ids")
    documents = _first(result, "documents")
    metadatas = _first(result, "metadatas")
    distances = _first(result, "distances")
    if not len(ids) == len(documents) == len(metadatas) == len(distances):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")
    matches = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        if not isinstance(metadata, dict) or metadata.get("upload_id") != upload_id:
            raise RetrievalUnavailableError("Filtered retrieval crossed dataset boundaries.")
        similarity = 1.0 - float(distance)
        if not math.isfinite(similarity):
            raise RetrievalUnavailableError("Vector search returned an invalid score.")
        row = metadata.get("row_number")
        matches.append(
            RetrievalMatch(
                rank=0,
                chunk_id=str(chunk_id),
                document=str(document),
                source_row=row if isinstance(row, int) and not isinstance(row, bool) else None,
                metadata={
                    str(key): value
                    for key, value in metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
                similarity_score=max(-1.0, min(1.0, similarity)),
            )
        )
    matches.sort(key=lambda match: match.similarity_score, reverse=True)
    return [
        RetrievalMatch(
            rank=rank,
            chunk_id=match.chunk_id,
            document=match.document,
            source_row=match.source_row,
            metadata=match.metadata,
            similarity_score=match.similarity_score,
        )
        for rank, match in enumerate(matches, 1)
    ]
