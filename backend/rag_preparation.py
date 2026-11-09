import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

from backend.analytics import find_column
from backend.dataset_selection import SessionFactory, load_selected_dataset

MetadataValue = str | int | float | bool
DEFAULT_MAX_CHUNK_CHARS = 2000
MIN_MAX_CHUNK_CHARS = 256
MISSING_VALUE_TEXT = "[missing]"


@dataclass(frozen=True)
class RowDocument:
    document_id: str
    text: str
    metadata: dict[str, MetadataValue]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, MetadataValue]


def _stable_id(namespace: str, *parts: object) -> str:
    value = "\x1f".join((namespace, *(str(part) for part in parts)))
    return sha256(value.encode("utf-8")).hexdigest()


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if hasattr(missing, "__len__"):
        return False
    return bool(missing)


def _single_line(value: object) -> str:
    text = str(value).replace("\x00", "�")
    return re.sub(r"\s+", " ", text).strip()


def _display_value(value: object) -> str:
    if _is_missing(value):
        return MISSING_VALUE_TEXT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return _single_line(value.decode("utf-8", errors="replace")) or "[empty]"
    if isinstance(value, float) and not math.isfinite(value):
        return "[invalid numeric value]"
    return _single_line(value) or "[empty]"


def _safe_filename(filename: object) -> str:
    normalized = str(filename).replace("\\", "/")
    return _single_line(Path(normalized).name) or "dataset.csv"


def _column_labels(columns: pd.Index) -> list[str]:
    totals: dict[str, int] = {}
    for column in columns:
        name = _single_line(column) or "Unnamed column"
        totals[name] = totals.get(name, 0) + 1

    occurrences: dict[str, int] = {}
    labels = []
    for column in columns:
        name = _single_line(column) or "Unnamed column"
        occurrences[name] = occurrences.get(name, 0) + 1
        if totals[name] > 1:
            labels.append(f"{name} [{occurrences[name]}]")
        else:
            labels.append(name)
    return labels


def _column_position(dataframe: pd.DataFrame, field: str) -> int | None:
    column = find_column(dataframe, field)
    if column is None:
        return None
    return next(
        (position for position, name in enumerate(dataframe.columns) if name == column),
        None,
    )


def _date_metadata_key(column: object, position: int) -> str | None:
    normalized = re.sub(r"[\s-]+", "_", str(column).strip().lower())
    if not normalized.endswith(("date", "time", "datetime", "timestamp", "_at")):
        return None
    key = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")
    return key or f"date_{position + 1}"


def _date_metadata_value(value: object) -> str | None:
    if _is_missing(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", format="mixed", utc=True)
    if pd.isna(parsed):
        return None
    if parsed.hour == parsed.minute == parsed.second == parsed.microsecond == 0:
        return parsed.date().isoformat()
    return parsed.isoformat()


def _row_metadata(
    dataframe: pd.DataFrame,
    values: tuple[object, ...],
    upload_id: str,
    source_filename: str,
    row_number: int,
    document_id: str,
) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {
        "upload_id": upload_id,
        "row_number": row_number,
        "source_filename": source_filename,
        "document_id": document_id,
    }

    for field, metadata_key in (
        ("order_id", "order_id"),
        ("region", "region"),
        ("category", "category"),
        ("sub_category", "sub_category"),
    ):
        position = _column_position(dataframe, field)
        if position is None:
            continue
        value = values[position]
        if not _is_missing(value):
            display_value = _display_value(value)
            if display_value not in (MISSING_VALUE_TEXT, "[empty]"):
                metadata[metadata_key] = display_value

    used_date_keys: dict[str, int] = {}
    for position, column in enumerate(dataframe.columns):
        base_key = _date_metadata_key(column, position)
        if base_key is None:
            continue
        used_date_keys[base_key] = used_date_keys.get(base_key, 0) + 1
        metadata_key = (
            base_key
            if used_date_keys[base_key] == 1
            else f"{base_key}_{used_date_keys[base_key]}"
        )
        parsed_value = _date_metadata_value(values[position])
        if parsed_value is not None:
            metadata[metadata_key] = parsed_value

    return metadata


def dataframe_to_documents(
    dataframe: pd.DataFrame,
    upload_id: UUID | str,
    source_filename: str,
) -> list[RowDocument]:
    """Convert every DataFrame row into a context-rich text document."""
    upload_id_value = str(upload_id).strip()
    if not upload_id_value:
        raise ValueError("upload_id is required.")

    safe_filename = _safe_filename(source_filename)
    labels = _column_labels(dataframe.columns)
    documents = []
    for row_number, values in enumerate(
        dataframe.itertuples(index=False, name=None),
        start=1,
    ):
        document_id = _stable_id("retailiq-row-v1", upload_id_value, row_number)
        value_lines = [
            f"- {label}: {_display_value(value)}"
            for label, value in zip(labels, values)
        ]
        if not value_lines:
            value_lines = ["- No columns were present in this row."]
        text = "\n".join(
            (
                f"Dataset: {safe_filename}",
                f"Upload ID: {upload_id_value}",
                f"Source row: {row_number}",
                "Column values:",
                *value_lines,
            )
        )
        documents.append(
            RowDocument(
                document_id=document_id,
                text=text,
                metadata=_row_metadata(
                    dataframe,
                    values,
                    upload_id_value,
                    safe_filename,
                    row_number,
                    document_id,
                ),
            )
        )
    return documents


def _split_payload(lines: list[str], limit: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for original_line in lines:
        line = original_line
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]

        candidate = line if not current else f"{current}\n{line}"
        if current and len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate

    if current or not parts:
        parts.append(current)
    return parts


def chunk_documents(
    documents: list[RowDocument],
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[DocumentChunk]:
    """Create one chunk per row unless a row exceeds the configured character limit."""
    if max_chars < MIN_MAX_CHUNK_CHARS:
        raise ValueError(
            f"max_chars must be at least {MIN_MAX_CHUNK_CHARS} characters."
        )

    chunks = []
    for document in documents:
        if len(document.text) <= max_chars:
            payloads = [document.text]
            split_document = False
        else:
            lines = document.text.splitlines()
            context_lines = lines[:3]
            payload_lines = lines[3:]
            context = "\n".join(context_lines)
            payload_limit = max(64, max_chars - len(context) - 40)
            payloads = _split_payload(payload_lines, payload_limit)
            split_document = True

        chunk_count = len(payloads)
        for chunk_index, payload in enumerate(payloads, start=1):
            if split_document:
                context = "\n".join(document.text.splitlines()[:3])
                chunk_text = (
                    f"{context}\nChunk: {chunk_index} of {chunk_count}\n{payload}"
                )
            else:
                chunk_text = payload
            chunk_id = _stable_id(
                "retailiq-chunk-v1",
                document.document_id,
                chunk_index,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    },
                )
            )
    return chunks


def dataframe_to_chunks(
    dataframe: pd.DataFrame,
    upload_id: UUID | str,
    source_filename: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[DocumentChunk]:
    documents = dataframe_to_documents(dataframe, upload_id, source_filename)
    return chunk_documents(documents, max_chars=max_chars)


def selected_dataset_to_documents(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
) -> list[RowDocument]:
    selected = load_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
    )
    return dataframe_to_documents(
        selected.dataframe,
        selected.upload_id,
        selected.original_filename,
    )


def prepare_selected_dataset_chunks(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
    storage_directory: Path | None = None,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[DocumentChunk]:
    selected = load_selected_dataset(
        upload_id,
        session_factory=session_factory,
        storage_directory=storage_directory,
    )
    return dataframe_to_chunks(
        selected.dataframe,
        selected.upload_id,
        selected.original_filename,
        max_chars=max_chars,
    )
