from io import BytesIO
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from backend.database import db
from backend.database.db import Base, _psycopg_database_url
from backend.database.models import UploadMetadata
from backend.file_storage import store_original_csv


def test_original_uploads_are_stored_unchanged_with_unique_names(tmp_path):
    contents = b"product,units\nTea,4\n"

    first = store_original_csv(
        BytesIO(contents),
        "../../sales.csv",
        storage_directory=tmp_path,
    )
    second = store_original_csv(
        BytesIO(contents),
        "../../sales.csv",
        storage_directory=tmp_path,
    )

    assert first.original_filename == "sales.csv"
    assert second.original_filename == "sales.csv"
    assert isinstance(first.upload_id, UUID)
    assert first.upload_id != second.upload_id
    assert first.stored_path != second.stored_path
    assert first.stored_path.parent == tmp_path.resolve()
    assert second.stored_path.parent == tmp_path.resolve()
    assert first.stored_path.read_bytes() == contents
    assert second.stored_path.read_bytes() == contents


def test_upload_metadata_table_has_required_postgresql_columns():
    table = UploadMetadata.__table__

    assert table.metadata is Base.metadata
    assert table.name == "upload_metadata"
    assert set(table.columns.keys()) == {
        "upload_id",
        "user_email",
        "original_filename",
        "stored_file_path",
        "file_size",
        "uploaded_at",
        "processing_status",
    }
    assert table.c.upload_id.primary_key
    assert table.c.user_email.nullable
    assert not table.c.original_filename.nullable
    assert not table.c.stored_file_path.nullable
    assert not table.c.file_size.nullable

    statement = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    assert "CREATE TABLE upload_metadata" in statement
    assert "UUID NOT NULL" in statement
    assert "BIGINT NOT NULL" in statement
    assert "TIMESTAMP WITH TIME ZONE" in statement


def test_init_db_reports_missing_database_configuration(monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DATABASE_URL_UNPOOLED", None)

    with pytest.raises(ValueError, match="DATABASE_URL is not configured"):
        db.init_db()


def test_standard_postgresql_url_uses_installed_psycopg_driver():
    normalized = _psycopg_database_url("postgresql://user:password@example.com/retailiq")

    assert normalized == "postgresql+psycopg://user:password@example.com/retailiq"
