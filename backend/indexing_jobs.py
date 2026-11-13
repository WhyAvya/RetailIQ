"""Safe in-process background indexing for uploaded datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock
from threading import BoundedSemaphore
from uuid import UUID

from backend.upload_metadata import transition_upload_status
from backend.vector_store import index_selected_dataset
from backend.runtime_config import MAX_CONCURRENT_INDEX_JOBS


logger = logging.getLogger(__name__)
Indexer = Callable[[UUID], object]
StatusTransition = Callable[[UUID, str, tuple[str, ...]], bool]
_active_lock = Lock()
_active_uploads: set[UUID] = set()
_index_slots = BoundedSemaphore(MAX_CONCURRENT_INDEX_JOBS)


def is_indexing_active(upload_id: UUID) -> bool:
    with _active_lock:
        return upload_id in _active_uploads


def run_indexing_job(
    upload_id: UUID,
    indexer: Indexer | None = None,
    transition: StatusTransition | None = None,
) -> bool:
    """Claim and index one upload; duplicate jobs safely become no-ops."""
    with _active_lock:
        if upload_id in _active_uploads:
            return False
        _active_uploads.add(upload_id)

    change_status = transition or transition_upload_status
    run_indexer = indexer or index_selected_dataset
    slot_acquired = False
    try:
        slot_acquired = _index_slots.acquire(blocking=False)
        if not slot_acquired:
            logger.warning("Indexing capacity is full; upload %s remains retryable", upload_id)
            return False
        try:
            claimed = change_status(upload_id, "indexing", ("pending",))
        except Exception:
            logger.exception("Dataset indexing could not be claimed for upload %s", upload_id)
            return False
        if not claimed:
            return False
        try:
            run_indexer(upload_id)
        except Exception:
            logger.exception("Dataset indexing failed for upload %s", upload_id)
            try:
                change_status(upload_id, "failed", ("indexing",))
            except Exception:
                logger.exception("Dataset indexing failure status could not be saved for %s", upload_id)
            return False
        try:
            return change_status(upload_id, "ready", ("indexing",))
        except Exception:
            logger.exception("Dataset indexing completion status could not be saved for %s", upload_id)
            return False
    finally:
        if slot_acquired:
            _index_slots.release()
        with _active_lock:
            _active_uploads.discard(upload_id)
