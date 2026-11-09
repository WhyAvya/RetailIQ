import pytest

from backend.metadata_filtered_retrieval import (
    build_chroma_where,
    retrieve_metadata_filtered_matches,
)
from backend.query_planner import plan_query
from backend.rag_retrieval import InvalidRetrievalRequestError


UPLOAD_ID = "11111111-1111-1111-1111-111111111111"


class FakeCollection:
    def __init__(self):
        self.query_kwargs = None

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "ids": [["chunk-2", "chunk-1"]],
            "documents": [["row 2", "row 1"]],
            "metadatas": [[
                {"upload_id": UPLOAD_ID, "row_number": 2, "region": "West", "sales": 2000.0},
                {"upload_id": UPLOAD_ID, "row_number": 1, "region": "West", "sales": 1500.0},
            ]],
            "distances": [[0.1, 0.3]],
        }


class FakeStore:
    def __init__(self):
        self._collection = FakeCollection()


def test_builds_upload_isolated_dimension_date_and_numeric_where_clause():
    plan = plan_query(
        "Show Chairs records in the West during 2017 with sales at least 1000"
    )
    where = build_chroma_where(UPLOAD_ID, plan)

    assert where["$and"][0] == {"upload_id": UPLOAD_ID}
    assert {"region": "West"} in where["$and"]
    assert {"sub_category": "Chairs"} in where["$and"]
    assert {"sales": {"$gte": 1000.0}} in where["$and"]
    assert any("order_date_epoch_day" in clause for clause in where["$and"])


def test_pure_semantic_plan_preserves_upload_only_filter():
    plan = plan_query("Find products suitable for a home office")

    assert build_chroma_where(UPLOAD_ID, plan) == {"upload_id": UPLOAD_ID}


def test_filtered_retrieval_ranks_within_where_clause_and_preserves_source_rows():
    store = FakeStore()
    plan = plan_query(
        "Among line items containing 'wireless' in the West, which five had highest sales?"
    )

    matches = retrieve_metadata_filtered_matches(
        UPLOAD_ID,
        plan,
        top_k=2,
        vector_store=store,
        embedding_function=lambda query: [0.5, 0.25] if query == "wireless" else [],
    )

    assert [match.source_row for match in matches] == [2, 1]
    assert [match.rank for match in matches] == [1, 2]
    assert matches[0].similarity_score == pytest.approx(0.9)
    assert store._collection.query_kwargs["where"]["$and"] == [
        {"upload_id": UPLOAD_ID},
        {"region": "West"},
    ]
    assert store._collection.query_kwargs["query_embeddings"] == [[0.5, 0.25]]


def test_rejects_ambiguous_plan_before_querying_chroma():
    with pytest.raises(InvalidRetrievalRequestError, match="valid query plan"):
        build_chroma_where(UPLOAD_ID, plan_query("What are the top products?"))


def test_builds_discount_equality_and_rejects_derived_shipping_filter():
    no_discount = build_chroma_where(
        UPLOAD_ID, plan_query("Show line items with no discount")
    )
    assert {"discount": {"$eq": 0.0}} in no_discount["$and"]

    with pytest.raises(InvalidRetrievalRequestError, match="derived from two dates"):
        build_chroma_where(
            UPLOAD_ID,
            plan_query("Which orders took more than seven calendar days to ship?"),
        )
