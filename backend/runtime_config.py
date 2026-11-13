"""Environment-backed runtime settings for the public anonymous MVP."""

from __future__ import annotations

import os
from pathlib import Path


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(value.strip().rstrip("/") for value in os.getenv(name, default).split(",") if value.strip())


SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "retailiq_session")
CSRF_COOKIE_NAME = os.getenv("CSRF_COOKIE_NAME", "retailiq_csrf")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax").lower()
ALLOWED_ORIGINS = _csv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
UPLOAD_STORAGE_PATH = Path(os.getenv("UPLOAD_STORAGE_PATH", "backend/storage/uploads"))
CHROMA_STORAGE_PATH = Path(os.getenv("CHROMA_STORAGE_PATH", "backend/storage/chroma"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "50000"))
MAX_CSV_COLUMNS = int(os.getenv("MAX_CSV_COLUMNS", "200"))
MAX_CONCURRENT_INDEX_JOBS = int(os.getenv("MAX_CONCURRENT_INDEX_JOBS", "2"))
MAX_QUESTIONS_PER_MINUTE = int(os.getenv("MAX_QUESTIONS_PER_MINUTE", "12"))

