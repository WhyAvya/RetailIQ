"""Anonymous cookie sessions and dataset ownership helpers."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from collections.abc import Callable
from sqlalchemy.exc import SQLAlchemyError

from backend.database.db import get_session_factory
from backend.database.models import AnonymousSession, UploadMetadata
from backend.runtime_config import SESSION_TTL_SECONDS


class SessionStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnonymousIdentity:
    token: str
    token_hash: str
    csrf_token: str


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_identity() -> AnonymousIdentity:
    token = secrets.token_urlsafe(32)
    return AnonymousIdentity(token, token_hash(token), secrets.token_urlsafe(24))


SessionFactory = Callable[[], Session]


def _session(session_factory: SessionFactory | None = None) -> Session:
    try:
        return (session_factory or get_session_factory())()
    except (SQLAlchemyError, ValueError) as exc:
        raise SessionStoreError("Anonymous session storage is unavailable.") from exc


def touch_session(identity: AnonymousIdentity, session_factory: SessionFactory | None = None) -> None:
    session = _session(session_factory)
    now = datetime.now(timezone.utc)
    try:
        record = session.get(AnonymousSession, identity.token_hash)
        if record is None:
            record = AnonymousSession(
                session_hash=identity.token_hash,
                csrf_hash=token_hash(identity.csrf_token),
                last_seen_at=now,
                expires_at=now + timedelta(seconds=SESSION_TTL_SECONDS),
            )
            session.add(record)
        else:
            record.last_seen_at = now
            record.expires_at = now + timedelta(seconds=SESSION_TTL_SECONDS)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise SessionStoreError("Anonymous session storage is unavailable.") from exc
    finally:
        session.close()


def owns_upload(upload_id: UUID, owner_hash: str, session_factory: SessionFactory | None = None) -> bool:
    session = _session(session_factory)
    try:
        record = session.get(UploadMetadata, upload_id)
        return record is not None and record.user_email == owner_hash
    except SQLAlchemyError as exc:
        raise SessionStoreError("Dataset ownership could not be verified.") from exc
    finally:
        session.close()


def session_uploads(owner_hash: str, session_factory: SessionFactory | None = None) -> list[UploadMetadata]:
    session = _session(session_factory)
    try:
        return list(session.scalars(select(UploadMetadata).where(UploadMetadata.user_email == owner_hash)).all())
    finally:
        session.close()


def remove_session_records(
    owner_hash: str,
    session_factory: SessionFactory | None = None,
    delete_session: bool = True,
) -> list[tuple[UUID, str]]:
    session = _session(session_factory)
    try:
        records = list(session.scalars(select(UploadMetadata).where(UploadMetadata.user_email == owner_hash)).all())
        targets = [(record.upload_id, record.stored_file_path) for record in records]
        session.execute(delete(UploadMetadata).where(UploadMetadata.user_email == owner_hash))
        if delete_session:
            session.execute(delete(AnonymousSession).where(AnonymousSession.session_hash == owner_hash))
        session.commit()
        return targets
    except SQLAlchemyError as exc:
        session.rollback()
        raise SessionStoreError("Session data could not be removed.") from exc
    finally:
        session.close()


def expired_session_hashes(session_factory: SessionFactory | None = None) -> list[str]:
    session = _session(session_factory)
    try:
        now = datetime.now(timezone.utc)
        return list(session.scalars(select(AnonymousSession.session_hash).where(AnonymousSession.expires_at <= now)).all())
    except SQLAlchemyError as exc:
        raise SessionStoreError("Expired sessions could not be listed.") from exc
    finally:
        session.close()

