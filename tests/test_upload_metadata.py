import asyncio
from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import backend.main as main_module
from backend.database.db import Base
from backend.database.models import UploadMetadata
from backend.file_storage import store_original_csv
from backend.upload_metadata import (
    DatabaseOperationError,
    build_dataset_preview,
    get_upload_metadata,
    list_upload_metadata,
    persist_upload_metadata,
    persist_upload_or_cleanup,
    transition_upload_status,
)


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def test_persisted_metadata_uses_the_stored_upload_id(session_factory, tmp_path):
    contents = b"product,units\nTea,4\n"
    stored_upload = store_original_csv(BytesIO(contents), "sales.csv", tmp_path)

    result = persist_upload_metadata(
        stored_upload,
        len(contents),
        session_factory=session_factory,
    )

    with session_factory() as session:
        record = session.get(UploadMetadata, stored_upload.upload_id)

    assert result["upload_id"] == str(stored_upload.upload_id)
    assert record is not None
    assert record.upload_id == stored_upload.upload_id
    assert record.stored_file_path == f"{stored_upload.upload_id}.csv"
    assert "stored_file_path" not in result


def test_upload_history_is_ordered_newest_first(session_factory):
    older_id = uuid4()
    newer_id = uuid4()
    with session_factory() as session:
        session.add_all(
            [
                UploadMetadata(
                    upload_id=older_id,
                    original_filename="older.csv",
                    stored_file_path=f"{older_id}.csv",
                    file_size=10,
                    uploaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    processing_status="processed",
                ),
                UploadMetadata(
                    upload_id=newer_id,
                    original_filename="newer.csv",
                    stored_file_path=f"{newer_id}.csv",
                    file_size=20,
                    uploaded_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
                    processing_status="processed",
                ),
            ]
        )
        session.commit()

    history = list_upload_metadata(session_factory=session_factory)

    assert [item["upload_id"] for item in history] == [str(newer_id), str(older_id)]
    assert all("stored_file_path" not in item for item in history)


def test_indexing_status_transition_is_atomic_and_conditional(session_factory, tmp_path):
    contents = b"product,units\nTea,4\n"
    stored_upload = store_original_csv(BytesIO(contents), "sales.csv", tmp_path)
    persist_upload_metadata(
        stored_upload,
        len(contents),
        processing_status="pending",
        session_factory=session_factory,
    )

    assert transition_upload_status(
        stored_upload.upload_id,
        "indexing",
        ("pending",),
        session_factory=session_factory,
    )
    assert not transition_upload_status(
        stored_upload.upload_id,
        "indexing",
        ("pending",),
        session_factory=session_factory,
    )
    assert transition_upload_status(
        stored_upload.upload_id,
        "ready",
        ("indexing",),
        session_factory=session_factory,
    )
    assert get_upload_metadata(
        stored_upload.upload_id, session_factory=session_factory
    )["status"] == "ready"


def test_preview_returns_only_requested_rows(session_factory, tmp_path):
    contents = b"product,units\nTea,4\nCoffee,2\nJuice,7\n"
    stored_upload = store_original_csv(BytesIO(contents), "sales.csv", tmp_path)
    persist_upload_metadata(
        stored_upload,
        len(contents),
        session_factory=session_factory,
    )
    metadata = get_upload_metadata(stored_upload.upload_id, session_factory=session_factory)

    preview = build_dataset_preview(metadata, row_limit=2, storage_directory=tmp_path)

    assert preview["columns"] == ["product", "units"]
    assert preview["preview_row_count"] == 2
    assert preview["rows"] == [
        {"product": "Tea", "units": 4},
        {"product": "Coffee", "units": 2},
    ]
    assert "stored_file_path" not in preview


class FailingSession:
    def __init__(self):
        self.rolled_back = False
        self.closed = False

    def add(self, _record):
        return None

    def flush(self):
        return None

    def refresh(self, record):
        record.uploaded_at = datetime.now(timezone.utc)

    def commit(self):
        raise SQLAlchemyError("database unavailable")

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_failed_metadata_insert_rolls_back_and_removes_new_file(tmp_path):
    contents = b"product,units\nTea,4\n"
    stored_upload = store_original_csv(BytesIO(contents), "sales.csv", tmp_path)
    failing_session = FailingSession()

    with pytest.raises(DatabaseOperationError):
        persist_upload_or_cleanup(
            stored_upload,
            len(contents),
            session_factory=lambda: failing_session,
        )

    assert failing_session.rolled_back
    assert failing_session.closed
    assert not stored_upload.stored_path.exists()


def test_missing_upload_preview_returns_404(monkeypatch):
    monkeypatch.setattr(main_module, "get_upload_metadata", lambda _upload_id: None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.dataset_preview(uuid4()))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "The requested upload was not found."


def test_history_database_failure_returns_generic_503(monkeypatch):
    def fail_history():
        raise DatabaseOperationError("internal database detail")

    monkeypatch.setattr(main_module, "list_upload_metadata", fail_history)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main_module.upload_history())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Upload history is temporarily unavailable."
    assert "internal database detail" not in exc_info.value.detail
