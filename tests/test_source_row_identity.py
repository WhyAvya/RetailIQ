from io import BytesIO

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.db import Base
from backend.dataset_selection import load_selected_dataset
from backend.file_storage import store_original_csv
from backend.query_planner import plan_query
from backend.rag_pipeline import answer_question
from backend.rag_preparation import prepare_selected_dataset_chunks
from backend.rag_retrieval import RetrievalMatch
from backend.structured_analytics import execute_plan
from backend.upload_metadata import persist_upload_metadata


CSV = b"""Row ID,Order ID,Order Date,Ship Date,Customer ID,Customer Name,Segment,City,State,Region,Product ID,Category,Sub-Category,Product Name,Sales,Quantity,Discount,Profit
101,ORDER-1,1/1/2017,1/3/2017,C1,Ada,Consumer,Seattle,Washington,West,P1,Technology,Accessories,Wireless Keyboard,100,1,0,20
102,ORDER-2,1/2/2017,1/4/2017,C2,Lin,Consumer,Seattle,Washington,West,P2,Technology,Accessories,Wireless Mouse,250,2,0,50
103,ORDER-3,1/3/2017,1/5/2017,C3,Sam,Consumer,Boston,Massachusetts,East,P3,Furniture,Chairs,Desk Chair,90,1,0,-5
"""


def _selected_dataset(tmp_path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    stored = store_original_csv(BytesIO(CSV), "small.csv", tmp_path)
    persist_upload_metadata(stored, len(CSV), session_factory=factory)
    selected = load_selected_dataset(
        stored.upload_id,
        session_factory=factory,
        storage_directory=tmp_path,
    )
    return stored, selected, factory, engine


def test_loaded_csv_rows_are_one_based_at_the_processing_boundary(tmp_path):
    stored, selected, factory, engine = _selected_dataset(tmp_path)
    try:
        assert selected.dataframe.index.tolist() == [1, 2, 3]
        assert selected.dataframe.loc[1, "Row ID"] == 101

        chunks = prepare_selected_dataset_chunks(
            stored.upload_id,
            session_factory=factory,
            storage_directory=tmp_path,
        )
        assert [chunk.metadata["row_number"] for chunk in chunks] == [1, 2, 3]
    finally:
        engine.dispose()


def test_structured_and_mixed_results_keep_one_based_source_rows(tmp_path):
    stored, selected, _factory, engine = _selected_dataset(tmp_path)
    try:
        structured = execute_plan(
            selected.dataframe,
            plan_query("Which two Technology line items had highest sales?"),
        )
        assert [record["source_row"] for record in structured.records] == [2, 1]

        match = RetrievalMatch(
            rank=1,
            chunk_id="row-2",
            document="Wireless Mouse with sales of 250.",
            source_row=2,
            metadata={"upload_id": str(stored.upload_id), "row_number": 2},
            similarity_score=0.9,
        )
        mixed = answer_question(
            stored.upload_id,
            "Among line items containing 'wireless' in the West, which two had highest sales?",
            metadata_lookup=lambda _upload_id: {"upload_id": str(stored.upload_id)},
            dataset_loader=lambda _upload_id: selected,
            filtered_retriever=lambda *_args: [match],
        )

        assert [record["source_row"] for record in mixed.calculation["records"]] == [2, 1]
        assert [source.source_row for source in mixed.sources] == [2]
    finally:
        engine.dispose()
