import json
from pathlib import Path

import pandas as pd

import backend.chroma_metadata as metadata_module
from backend.chroma_metadata import (
    FIELD_TYPES,
    audit_and_update_metadata,
    audit_metadata_snapshot,
    expected_row_metadata,
)
from backend.rag_preparation import dataframe_to_chunks


class FakeCollection:
    def __init__(self, ids, metadatas):
        self.ids = list(ids)
        self.metadatas = [dict(item) for item in metadatas]
        self.update_calls = []

    def get(self, **_kwargs):
        return {"ids": list(self.ids), "metadatas": [dict(item) for item in self.metadatas]}

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        assert set(kwargs) == {"ids", "metadatas"}
        by_id = dict(zip(kwargs["ids"], kwargs["metadatas"]))
        self.metadatas = [by_id.get(chunk_id, old) for chunk_id, old in zip(self.ids, self.metadatas)]


class FakeStore:
    def __init__(self, collection):
        self._collection = collection


def _two_row_source(tmp_path: Path):
    source = pd.read_csv("data/sample_superstore.csv", encoding="cp1252").head(2)
    path = tmp_path / "sample_superstore.csv"
    source.to_csv(path, index=False, encoding="cp1252")
    source.index = pd.RangeIndex(1, 3, name="source_row")
    return path, source


def test_audit_identifies_missing_fields_and_preserves_valid_existing_metadata(tmp_path):
    _path, source = _two_row_source(tmp_path)
    stored = {
        "ids": ["one", "two"],
        "metadatas": [
            {"upload_id": metadata_module.FULL_DATASET_UPLOAD_ID, "row_number": 1, "region": "South"},
            {"upload_id": metadata_module.FULL_DATASET_UPLOAD_ID, "row_number": 2, "region": "South"},
        ],
    }

    audit = audit_metadata_snapshot(stored, source)

    assert audit["complete"] is False
    assert audit["missing_field_counts"]["sales"] == 2
    assert audit["missing_field_counts"]["product_name"] == 2
    assert "region" not in audit["missing_field_counts"]
    assert audit["duplicate_source_rows"] == []


def test_metadata_update_changes_only_metadata_and_validates_all_types(tmp_path, monkeypatch):
    path, source = _two_row_source(tmp_path)
    chunks = dataframe_to_chunks(
        source,
        metadata_module.FULL_DATASET_UPLOAD_ID,
        metadata_module.SOURCE_FILENAME,
    )
    original = [dict(chunk.metadata) for chunk in chunks]
    collection = FakeCollection([chunk.chunk_id for chunk in chunks], original)
    monkeypatch.setattr(metadata_module, "EXPECTED_ROWS", 2)

    report = audit_and_update_metadata(
        source_path=path,
        vector_store=FakeStore(collection),
    )

    assert report["updated_chunk_count"] == 2
    assert report["documents_supplied_to_update"] is False
    assert report["embeddings_supplied_to_update"] is False
    assert report["ids_before"] == report["ids_after"] == 2
    assert report["audit_after"]["complete"] is True
    assert collection.update_calls
    for stored in collection.metadatas:
        for field, expected_type in FIELD_TYPES.items():
            assert isinstance(stored[field], expected_type)
        assert stored["document_id"] in {item["document_id"] for item in original}


def test_expected_numeric_and_date_metadata_are_chroma_scalars(tmp_path):
    _path, source = _two_row_source(tmp_path)
    metadata = expected_row_metadata(source.loc[1])

    assert isinstance(metadata["sales"], float)
    assert isinstance(metadata["profit"], float)
    assert isinstance(metadata["quantity"], int)
    assert isinstance(metadata["order_date_epoch_day"], int)
    assert metadata["order_date"] == "2016-11-08"


def test_saved_live_audit_confirms_metadata_only_complete_update():
    report_path = Path("evaluation/reports/batch1_chroma_metadata_audit.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["operation"] == "metadata_only_update"
    assert report["updated_chunk_count"] == 9994
    assert report["ids_before"] == report["ids_after"] == 9994
    assert report["documents_supplied_to_update"] is False
    assert report["embeddings_supplied_to_update"] is False
    assert report["audit_before"]["missing_field_counts"]["sales"] == 9994
    assert report["audit_before"]["missing_field_counts"]["profit"] == 9994
    assert report["audit_after"]["complete"] is True
    assert report["audit_after"]["missing_field_counts"] == {}
    assert report["audit_after"]["incorrect_type_counts"] == {}
