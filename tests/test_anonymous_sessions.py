from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.anonymous_sessions import (
    expired_session_hashes,
    new_identity,
    owns_upload,
    remove_session_records,
    session_uploads,
    touch_session,
)
from backend.database.db import Base
from backend.database.models import AnonymousSession, UploadMetadata


def _factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_tokens_are_unpredictable_and_only_hash_is_persisted():
    factory = _factory()
    first, second = new_identity(), new_identity()
    assert first.token != second.token
    assert len(first.token_hash) == 64
    touch_session(first, factory)
    with factory() as session:
        stored = session.get(AnonymousSession, first.token_hash)
        assert stored is not None
        assert first.token not in stored.session_hash


def test_upload_ownership_is_strictly_session_scoped():
    factory = _factory()
    first, second = new_identity(), new_identity()
    upload_id = uuid4()
    with factory() as session:
        session.add(UploadMetadata(upload_id=upload_id, user_email=first.token_hash, original_filename="employees.csv", stored_file_path=f"{upload_id}.csv", file_size=10, processing_status="ready"))
        session.commit()
    assert owns_upload(upload_id, first.token_hash, factory)
    assert not owns_upload(upload_id, second.token_hash, factory)
    assert [item.upload_id for item in session_uploads(first.token_hash, factory)] == [upload_id]
    assert session_uploads(second.token_hash, factory) == []


def test_expiry_and_immediate_deletion_remove_temporary_metadata():
    factory = _factory()
    identity = new_identity()
    upload_id = uuid4()
    with factory() as session:
        session.add(AnonymousSession(session_hash=identity.token_hash, csrf_hash="x" * 64, last_seen_at=datetime.now(timezone.utc), expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
        session.add(UploadMetadata(upload_id=upload_id, user_email=identity.token_hash, original_filename="inventory.csv", stored_file_path=f"{upload_id}.csv", file_size=10, processing_status="ready"))
        session.commit()
    assert expired_session_hashes(factory) == [identity.token_hash]
    assert remove_session_records(identity.token_hash, factory) == [(upload_id, f"{upload_id}.csv")]
    assert session_uploads(identity.token_hash, factory) == []
