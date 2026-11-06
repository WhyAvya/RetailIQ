from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

STORAGE_CHUNK_SIZE = 1024 * 1024
UPLOAD_DIRECTORY = Path(__file__).resolve().parent / "storage" / "uploads"


class FileStorageError(OSError):
    """Raised when an original upload cannot be stored safely."""


@dataclass(frozen=True)
class StoredUpload:
    upload_id: UUID
    original_filename: str
    stored_path: Path


def _safe_original_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    return Path(normalized).name or "upload.csv"


def store_original_csv(
    file_object: BinaryIO,
    original_filename: str,
    storage_directory: Path | None = None,
) -> StoredUpload:
    directory = (storage_directory or UPLOAD_DIRECTORY).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    upload_id = uuid4()
    destination = (directory / f"{upload_id}.csv").resolve()

    if destination.parent != directory:
        raise FileStorageError("The upload destination is not safe.")

    try:
        file_object.seek(0)
        with destination.open("xb") as stored_file:
            while chunk := file_object.read(STORAGE_CHUNK_SIZE):
                stored_file.write(chunk)
        file_object.seek(0)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise FileStorageError("The original CSV could not be stored.") from exc

    return StoredUpload(
        upload_id=upload_id,
        original_filename=_safe_original_filename(original_filename),
        stored_path=destination,
    )


def remove_stored_upload(stored_upload: StoredUpload) -> None:
    """Remove a newly stored upload when its metadata cannot be persisted."""
    try:
        stored_upload.stored_path.unlink(missing_ok=True)
    except OSError as exc:
        raise FileStorageError("The incomplete upload could not be removed.") from exc


def resolve_stored_upload_path(
    stored_file_path: str,
    storage_directory: Path | None = None,
) -> Path:
    """Resolve a metadata path while preventing access outside upload storage."""
    directory = (storage_directory or UPLOAD_DIRECTORY).resolve()

    if Path(stored_file_path).name != stored_file_path:
        raise FileStorageError("The stored upload path is not safe.")

    stored_path = (directory / stored_file_path).resolve()
    if stored_path.parent != directory:
        raise FileStorageError("The stored upload path is not safe.")
    if not stored_path.is_file():
        raise FileStorageError("The stored upload is unavailable.")

    return stored_path
