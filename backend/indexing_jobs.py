"""Safe in-process background indexing for uploaded datasets."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from threading import BoundedSemaphore
from time import monotonic
from uuid import UUID

from backend.upload_metadata import transition_upload_status
from backend.vector_store import index_selected_dataset
from backend.runtime_config import (
    GEMINI_EMBEDDING_REQUESTS_PER_MINUTE,
    MAX_CONCURRENT_INDEX_JOBS,
)


logger = logging.getLogger(__name__)
Indexer = Callable[[UUID], object]
StatusTransition = Callable[[UUID, str, tuple[str, ...]], bool]
_active_lock = Lock()
_active_uploads: set[UUID] = set()
_index_slots = BoundedSemaphore(MAX_CONCURRENT_INDEX_JOBS)


@dataclass
class _IndexingEstimate:
    total_items: int
    started_at: float | None = None


_indexing_estimates: dict[UUID, _IndexingEstimate] = {}


def prepare_indexing_estimate(upload_id: UUID, total_items: int) -> None:
    """Record enough information to estimate a queued job's remaining time."""
    if total_items < 0:
        raise ValueError("total_items cannot be negative.")
    with _active_lock:
        _indexing_estimates[upload_id] = _IndexingEstimate(total_items=total_items)


def get_indexing_progress(upload_id: UUID) -> dict[str, int] | None:
    with _active_lock:
        estimate = _indexing_estimates.get(upload_id)
        if estimate is None or GEMINI_EMBEDDING_REQUESTS_PER_MINUTE <= 0:
            return None
        total_seconds = (
            estimate.total_items * 60.0 / GEMINI_EMBEDDING_REQUESTS_PER_MINUTE
        )
        elapsed_seconds = (
            max(0.0, monotonic() - estimate.started_at)
            if estimate.started_at is not None
            else 0.0
        )
    remaining_seconds = max(0, math.ceil(total_seconds - elapsed_seconds))
    progress_percent = (
        min(99, math.floor(elapsed_seconds * 100 / total_seconds))
        if total_seconds > 0
        else 100
    )
    return {
        "estimated_seconds_remaining": remaining_seconds,
        "progress_percent": progress_percent,
    }


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
        with _active_lock:
            estimate = _indexing_estimates.get(upload_id)
            if estimate is not None:
                estimate.started_at = monotonic()
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
            _indexing_estimates.pop(upload_id, None)
