from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.csv_utils import read_csv_dataframe
from backend.database.db import get_session_factory
from backend.database.models import UploadMetadata
from backend.file_storage import (
    FileStorageError,
    StoredUpload,
    remove_stored_upload,
    resolve_stored_upload_path,
)

SessionFactory = Callable[[], Session]


class DatabaseOperationError(RuntimeError):
    """Raised when upload metadata cannot be read or written."""


def _session(session_factory: SessionFactory | None = None) -> Session:
    try:
        factory = session_factory or get_session_factory()
        return factory()
    except (SQLAlchemyError, ValueError) as exc:
        raise DatabaseOperationError("The database is unavailable.") from exc


def _public_metadata(record: UploadMetadata) -> dict[str, Any]:
    return {
        "upload_id": str(record.upload_id),
        "filename": record.original_filename,
        "size": int(record.file_size),
        "uploaded_at": record.uploaded_at.isoformat(),
        "status": record.processing_status,
    }


def persist_upload_metadata(
    stored_upload: StoredUpload,
    file_size: int,
    user_email: str | None = None,
    processing_status: str = "processed",
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    session = _session(session_factory)
    record = UploadMetadata(
        upload_id=stored_upload.upload_id,
        user_email=user_email,
        original_filename=stored_upload.original_filename,
        stored_file_path=stored_upload.stored_path.name,
        file_size=file_size,
        processing_status=processing_status,
    )

    try:
        session.add(record)
        session.flush()
        session.refresh(record)
        public_metadata = _public_metadata(record)
        session.commit()
        return public_metadata
    except SQLAlchemyError as exc:
        session.rollback()
        raise DatabaseOperationError("Upload metadata could not be saved.") from exc
    finally:
        session.close()


def persist_upload_or_cleanup(
    stored_upload: StoredUpload,
    file_size: int,
    user_email: str | None = None,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    try:
        return persist_upload_metadata(
            stored_upload,
            file_size,
            user_email=user_email,
            session_factory=session_factory,
        )
    except DatabaseOperationError:
        try:
            remove_stored_upload(stored_upload)
        except FileStorageError:
            pass
        raise


def list_upload_metadata(
    session_factory: SessionFactory | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    session = _session(session_factory)
    try:
        statement = (
            select(UploadMetadata)
            .order_by(UploadMetadata.uploaded_at.desc(), UploadMetadata.upload_id.desc())
            .limit(limit)
        )
        records = session.scalars(statement).all()
        return [_public_metadata(record) for record in records]
    except SQLAlchemyError as exc:
        raise DatabaseOperationError("Upload history could not be loaded.") from exc
    finally:
        session.close()


def get_upload_metadata(
    upload_id: UUID,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any] | None:
    session = _session(session_factory)
    try:
        record = session.get(UploadMetadata, upload_id)
        if record is None:
            return None

        return {
            **_public_metadata(record),
            "stored_file_path": record.stored_file_path,
        }
    except SQLAlchemyError as exc:
        raise DatabaseOperationError("Upload metadata could not be loaded.") from exc
    finally:
        session.close()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_dataset_preview(
    metadata: dict[str, Any],
    row_limit: int = 10,
    storage_directory: Path | None = None,
) -> dict[str, Any]:
    stored_path = resolve_stored_upload_path(
        metadata["stored_file_path"],
        storage_directory=storage_directory,
    )
    with stored_path.open("rb") as stored_file:
        dataframe = read_csv_dataframe(stored_file)

    preview_frame = dataframe.head(row_limit)
    rows = [
        {str(column): _json_value(value) for column, value in row.items()}
        for row in preview_frame.to_dict(orient="records")
    ]

    return {
        "upload_id": metadata["upload_id"],
        "filename": metadata["filename"],
        "columns": [str(column) for column in dataframe.columns],
        "rows": rows,
        "preview_row_count": len(rows),
    }
