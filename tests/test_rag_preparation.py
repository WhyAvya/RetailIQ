from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.csv_utils import read_csv_dataframe
from backend.database.db import Base
from backend.file_storage import store_original_csv
from backend.rag_preparation import (
    MISSING_VALUE_TEXT,
    chunk_documents,
    dataframe_to_chunks,
    dataframe_to_documents,
    prepare_selected_dataset_chunks,
    selected_dataset_to_documents,
)
from backend.upload_metadata import persist_upload_metadata


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def test_rows_become_context_rich_documents_without_mutating_source():
    upload_id = uuid4()
    dataframe = pd.DataFrame(
        {
            "Order ID": ["ORDER-1", "ORDER-1", "ORDER-2"],
            "Order Date": ["2026-01-05", None, "not-a-date"],
            "Region": ["West", pd.NA, "East"],
            "Category": ["Technology", "Technology", "Office"],
            "Sub-Category": ["Phones", "Phones", "Paper"],
            "Customer Note": [
                'Café, "priority"\nsecond line',
                None,
                "contains <xml> & symbols",
            ],
            "Score": [10.5, float("inf"), 3],
        }
    )
    snapshot = dataframe.copy(deep=True)

    documents = dataframe_to_documents(
        dataframe,
        upload_id,
        r"C:\private\retail sample.csv",
    )

    assert len(documents) == 3
    assert len({document.document_id for document in documents}) == 3
    assert [document.metadata["row_number"] for document in documents] == [1, 2, 3]
    assert documents[0].metadata["order_id"] == "ORDER-1"
    assert documents[1].metadata["order_id"] == "ORDER-1"
    assert documents[0].metadata["order_date"] == "2026-01-05"
    assert "order_date" not in documents[1].metadata
    assert "order_date" not in documents[2].metadata
    assert documents[0].metadata["region"] == "West"
    assert "region" not in documents[1].metadata
    assert documents[0].metadata["source_filename"] == "retail sample.csv"
    assert "C:\\private" not in documents[0].text
    assert "Customer Note: Café, \"priority\" second line" in documents[0].text
    assert f"Order Date: {MISSING_VALUE_TEXT}" in documents[1].text
    assert "Score: [invalid numeric value]" in documents[1].text
    assert "contains <xml> & symbols" in documents[2].text
    assert all(
        isinstance(value, (str, int, float, bool))
        for document in documents
        for value in document.metadata.values()
    )
    pd.testing.assert_frame_equal(dataframe, snapshot)


def test_sample_superstore_keeps_every_row_and_repeated_order_id():
    source_path = Path(__file__).resolve().parents[1] / "data" / "sample_superstore.csv"
    original_hash = sha256(source_path.read_bytes()).hexdigest()
    with source_path.open("rb") as source_file:
        dataframe = read_csv_dataframe(source_file)
    upload_id = UUID("11111111-1111-1111-1111-111111111111")

    documents = dataframe_to_documents(dataframe, upload_id, source_path.name)
    chunks = chunk_documents(documents)

    assert len(documents) == len(dataframe) == 9994
    assert len(chunks) == 9994
    assert len({document.document_id for document in documents}) == 9994
    assert len({chunk.chunk_id for chunk in chunks}) == 9994
    order_counts = Counter(
        document.metadata["order_id"]
        for document in documents
        if "order_id" in document.metadata
    )
    repeated_order = next(order_id for order_id, count in order_counts.items() if count > 1)
    repeated_rows = [
        document.metadata["row_number"]
        for document in documents
        if document.metadata.get("order_id") == repeated_order
    ]
    assert len(repeated_rows) > 1
    assert len(set(repeated_rows)) == len(repeated_rows)
    assert documents[0].metadata["upload_id"] == str(upload_id)
    assert documents[0].metadata["source_filename"] == "sample_superstore.csv"
    assert documents[0].metadata["region"]
    assert documents[0].metadata["category"]
    assert documents[0].metadata["sub_category"]
    assert documents[0].metadata["order_date"]
    assert documents[0].metadata["ship_date"]
    assert sha256(source_path.read_bytes()).hexdigest() == original_hash


def test_long_row_splits_without_losing_content_and_ids_are_stable():
    upload_id = UUID("22222222-2222-2222-2222-222222222222")
    long_value = "§" * 5000
    dataframe = pd.DataFrame(
        {
            "Ticket": ["T-1"],
            "Observed At": ["2026-11-07T09:30:00+05:30"],
            "Description": [long_value],
        }
    )
    documents = dataframe_to_documents(dataframe, upload_id, "tickets.csv")

    first_chunks = chunk_documents(documents, max_chars=300)
    second_chunks = chunk_documents(documents, max_chars=300)

    assert len(first_chunks) > 1
    assert [chunk.chunk_id for chunk in first_chunks] == [
        chunk.chunk_id for chunk in second_chunks
    ]
    assert sum(chunk.text.count("§") for chunk in first_chunks) == len(long_value)
    assert all(len(chunk.text) <= 300 for chunk in first_chunks)
    assert all(
        chunk.metadata["chunk_count"] == len(first_chunks)
        for chunk in first_chunks
    )
    assert [chunk.metadata["chunk_index"] for chunk in first_chunks] == list(
        range(1, len(first_chunks) + 1)
    )
    assert all(chunk.metadata["observed_at"].startswith("2026-11-07T04:00:00") for chunk in first_chunks)


def test_non_retail_schema_and_dataset_isolation():
    dataframe = pd.DataFrame(
        {
            "Sensor": ["A-1"],
            "Reading": [42.5],
            "Captured Date": ["2026-10-01"],
            "Comment": [None],
        }
    )
    first_upload = UUID("33333333-3333-3333-3333-333333333333")
    second_upload = UUID("44444444-4444-4444-4444-444444444444")

    first = dataframe_to_chunks(dataframe, first_upload, "sensors.csv")
    first_again = dataframe_to_chunks(dataframe, first_upload, "sensors.csv")
    second = dataframe_to_chunks(dataframe, second_upload, "sensors.csv")

    assert first[0].chunk_id == first_again[0].chunk_id
    assert first[0].chunk_id != second[0].chunk_id
    assert first[0].metadata["document_id"] != second[0].metadata["document_id"]
    assert first[0].metadata["upload_id"] == str(first_upload)
    assert second[0].metadata["upload_id"] == str(second_upload)
    assert first[0].metadata["captured_date"] == "2026-10-01"
    assert "region" not in first[0].metadata
    assert f"Comment: {MISSING_VALUE_TEXT}" in first[0].text


def test_selected_dataset_wrappers_reuse_secure_loader_and_preserve_file(
    session_factory,
    tmp_path,
):
    contents = (
        b"Order ID,Order Date,Region,Category,Sub-Category,Sales\n"
        b"A,2026-01-05,West,Technology,Phones,100\n"
        b"A,2026-01-06,West,Technology,Phones,150\n"
    )
    stored = store_original_csv(
        BytesIO(contents),
        "small-retail.csv",
        storage_directory=tmp_path,
    )
    persist_upload_metadata(
        stored,
        len(contents),
        session_factory=session_factory,
    )
    original_hash = sha256(stored.stored_path.read_bytes()).hexdigest()

    documents = selected_dataset_to_documents(
        stored.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )
    chunks = prepare_selected_dataset_chunks(
        stored.upload_id,
        session_factory=session_factory,
        storage_directory=tmp_path,
    )

    assert len(documents) == len(chunks) == 2
    assert [document.metadata["row_number"] for document in documents] == [1, 2]
    assert documents[0].metadata["order_id"] == documents[1].metadata["order_id"]
    assert chunks[0].metadata["source_filename"] == "small-retail.csv"
    assert stored.stored_path.name not in chunks[0].text
    assert "stored_file_path" not in chunks[0].metadata
    assert sha256(stored.stored_path.read_bytes()).hexdigest() == original_hash


def test_chunk_size_has_safe_minimum():
    documents = dataframe_to_documents(
        pd.DataFrame({"value": ["short"]}),
        uuid4(),
        "sample.csv",
    )

    with pytest.raises(ValueError, match="at least 256"):
        chunk_documents(documents, max_chars=100)
