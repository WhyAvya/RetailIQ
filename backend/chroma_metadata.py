"""Audit and safely enrich Superstore Chroma metadata without touching vectors."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from backend.rag_preparation import dataframe_to_chunks
from backend.vector_store import COLLECTION_NAME, ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "sample_superstore.csv"
DEFAULT_AUDIT_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "reports" / "batch1_chroma_metadata_audit.json"
)
FULL_DATASET_UPLOAD_ID = "11111111-1111-1111-1111-111111111111"
SOURCE_FILENAME = "sample_superstore.csv"
EXPECTED_ROWS = 9_994
METADATA_BATCH_SIZE = 500


class MetadataAuditError(RuntimeError):
    """Raised when metadata cannot be updated without risking row identity."""


FIELD_TYPES: dict[str, type] = {
    "row_id": int,
    "order_id": str,
    "order_date": str,
    "order_date_epoch_day": int,
    "order_year": int,
    "order_month": int,
    "ship_date": str,
    "ship_date_epoch_day": int,
    "ship_mode": str,
    "customer_id": str,
    "customer_name": str,
    "segment": str,
    "country": str,
    "city": str,
    "state": str,
    "postal_code": str,
    "region": str,
    "product_id": str,
    "category": str,
    "sub_category": str,
    "product_name": str,
    "sales": float,
    "quantity": int,
    "discount": float,
    "profit": float,
}


def load_metadata_source(source_path: Path | str = DEFAULT_SOURCE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(Path(source_path), encoding="cp1252")
    if len(frame) != EXPECTED_ROWS:
        raise MetadataAuditError(
            f"expected {EXPECTED_ROWS} source rows, found {len(frame)}"
        )
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="source_row")
    return frame


def _epoch_day(value: pd.Timestamp) -> int:
    return (value.date() - date(1970, 1, 1)).days


def expected_row_metadata(row: pd.Series) -> dict[str, str | int | float]:
    order_date = pd.to_datetime(row["Order Date"], format="mixed")
    ship_date = pd.to_datetime(row["Ship Date"], format="mixed")
    return {
        "row_id": int(row["Row ID"]),
        "order_id": str(row["Order ID"]),
        "order_date": order_date.date().isoformat(),
        "order_date_epoch_day": _epoch_day(order_date),
        "order_year": int(order_date.year),
        "order_month": int(order_date.month),
        "ship_date": ship_date.date().isoformat(),
        "ship_date_epoch_day": _epoch_day(ship_date),
        "ship_mode": str(row["Ship Mode"]),
        "customer_id": str(row["Customer ID"]),
        "customer_name": str(row["Customer Name"]),
        "segment": str(row["Segment"]),
        "country": str(row["Country"]),
        "city": str(row["City"]),
        "state": str(row["State"]),
        "postal_code": str(int(row["Postal Code"])),
        "region": str(row["Region"]),
        "product_id": str(row["Product ID"]),
        "category": str(row["Category"]),
        "sub_category": str(row["Sub-Category"]),
        "product_name": str(row["Product Name"]),
        "sales": float(row["Sales"]),
        "quantity": int(row["Quantity"]),
        "discount": float(row["Discount"]),
        "profit": float(row["Profit"]),
    }


def _valid_type(value: Any, expected: type) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, expected)


def audit_metadata_snapshot(
    stored: dict[str, Any],
    source: pd.DataFrame,
    upload_id: str = FULL_DATASET_UPLOAD_ID,
) -> dict[str, Any]:
    ids = list(stored.get("ids") or [])
    metadatas = list(stored.get("metadatas") or [])
    if len(ids) != len(metadatas):
        raise MetadataAuditError("Chroma IDs and metadata counts do not match.")
    rows_to_ids: dict[int, list[str]] = defaultdict(list)
    missing_fields: Counter[str] = Counter()
    type_mismatches: Counter[str] = Counter()
    value_mismatches: Counter[str] = Counter()
    cross_upload_rows = 0
    invalid_row_numbers = 0
    for chunk_id, metadata in zip(ids, metadatas):
        row_number = metadata.get("row_number")
        if not isinstance(row_number, int) or isinstance(row_number, bool) or row_number not in source.index:
            invalid_row_numbers += 1
            continue
        rows_to_ids[row_number].append(chunk_id)
        if metadata.get("upload_id") != upload_id:
            cross_upload_rows += 1
        expected = expected_row_metadata(source.loc[row_number])
        for field, expected_type in FIELD_TYPES.items():
            if field not in metadata:
                missing_fields[field] += 1
            elif not _valid_type(metadata[field], expected_type):
                type_mismatches[field] += 1
            elif metadata[field] != expected[field]:
                value_mismatches[field] += 1
    expected_rows = set(int(value) for value in source.index)
    stored_rows = set(rows_to_ids)
    duplicate_rows = sorted(row for row, row_ids in rows_to_ids.items() if len(row_ids) > 1)
    return {
        "expected_rows": len(expected_rows),
        "stored_chunks": len(ids),
        "unique_chunk_ids": len(set(ids)),
        "unique_source_rows": len(stored_rows),
        "missing_source_rows": sorted(expected_rows - stored_rows),
        "duplicate_source_rows": duplicate_rows,
        "invalid_row_number_count": invalid_row_numbers,
        "cross_upload_row_count": cross_upload_rows,
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "incorrect_type_counts": dict(sorted(type_mismatches.items())),
        "incorrect_value_counts": dict(sorted(value_mismatches.items())),
        "complete": (
            len(ids) == len(expected_rows)
            and len(set(ids)) == len(ids)
            and stored_rows == expected_rows
            and not duplicate_rows
            and invalid_row_numbers == 0
            and cross_upload_rows == 0
            and not missing_fields
            and not type_mismatches
            and not value_mismatches
        ),
    }


def _expected_chunk_ids(source: pd.DataFrame, upload_id: str) -> dict[int, str]:
    chunks = dataframe_to_chunks(source, upload_id, SOURCE_FILENAME)
    if len(chunks) != len(source):
        raise MetadataAuditError("Expected exactly one deterministic chunk per source row.")
    return {int(chunk.metadata["row_number"]): chunk.chunk_id for chunk in chunks}


def audit_and_update_metadata(
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    upload_id: str = FULL_DATASET_UPLOAD_ID,
    vector_store: ChromaVectorStore | None = None,
) -> dict[str, Any]:
    """Apply metadata-only updates to verified deterministic row chunks."""
    source_path = Path(source_path)
    source = load_metadata_source(source_path)
    store = vector_store or ChromaVectorStore()
    collection = store._collection
    stored = collection.get(where={"upload_id": upload_id}, include=["metadatas"])
    before = audit_metadata_snapshot(stored, source, upload_id)
    if (
        before["missing_source_rows"]
        or before["duplicate_source_rows"]
        or before["invalid_row_number_count"]
        or before["cross_upload_row_count"]
        or before["stored_chunks"] != EXPECTED_ROWS
    ):
        raise MetadataAuditError(
            "row identity or upload isolation is invalid; refusing metadata update"
        )
    expected_ids = _expected_chunk_ids(source, upload_id)
    updates: list[tuple[str, dict[str, str | int | float | bool]]] = []
    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
        row_number = int(metadata["row_number"])
        if expected_ids[row_number] != chunk_id:
            raise MetadataAuditError(
                f"source row {row_number} has a non-deterministic chunk ID"
            )
        enriched = {**metadata, **expected_row_metadata(source.loc[row_number])}
        if enriched != metadata:
            updates.append((chunk_id, enriched))
    for start in range(0, len(updates), METADATA_BATCH_SIZE):
        batch = updates[start : start + METADATA_BATCH_SIZE]
        collection.update(
            ids=[item[0] for item in batch],
            metadatas=[item[1] for item in batch],
        )
    after_stored = collection.get(where={"upload_id": upload_id}, include=["metadatas"])
    after = audit_metadata_snapshot(after_stored, source, upload_id)
    if set(after_stored["ids"]) != set(stored["ids"]):
        raise MetadataAuditError("chunk IDs changed during metadata update")
    if not after["complete"]:
        raise MetadataAuditError("metadata update completed but validation still fails")
    return {
        "operation": "metadata_only_update",
        "source": {
            "path": "data/sample_superstore.csv",
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "row_count": len(source),
        },
        "upload_id": upload_id,
        "collection": COLLECTION_NAME,
        "updated_chunk_count": len(updates),
        "documents_supplied_to_update": False,
        "embeddings_supplied_to_update": False,
        "ids_before": len(stored["ids"]),
        "ids_after": len(after_stored["ids"]),
        "audit_before": before,
        "audit_after": after,
    }


def save_audit_report(
    report: dict[str, Any], output_path: Path | str = DEFAULT_AUDIT_REPORT_PATH
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path
