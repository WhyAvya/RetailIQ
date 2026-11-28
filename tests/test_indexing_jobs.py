from threading import Event, Thread
from uuid import uuid4

import backend.indexing_jobs as indexing_jobs
from backend.indexing_jobs import (
    get_indexing_progress,
    is_indexing_active,
    prepare_indexing_estimate,
    run_indexing_job,
)


def _state_transition(state):
    def transition(_upload_id, new_status, expected_statuses):
        if state["value"] not in expected_statuses:
            return False
        state["value"] = new_status
        return True

    return transition


def test_indexing_job_transitions_to_ready_and_retry_is_idempotent():
    upload_id = uuid4()
    state = {"value": "pending"}
    calls = []

    assert run_indexing_job(
        upload_id,
        indexer=lambda value: calls.append(value),
        transition=_state_transition(state),
    )
    assert state["value"] == "ready"
    assert calls == [upload_id]

    assert not run_indexing_job(
        upload_id,
        indexer=lambda value: calls.append(value),
        transition=_state_transition(state),
    )
    assert calls == [upload_id]


def test_indexing_failure_is_recorded_without_escaping():
    upload_id = uuid4()
    state = {"value": "pending"}

    def fail(_upload_id):
        raise RuntimeError("embedding failed")

    assert not run_indexing_job(
        upload_id,
        indexer=fail,
        transition=_state_transition(state),
    )
    assert state["value"] == "failed"
    assert not is_indexing_active(upload_id)


def test_indexing_estimate_counts_down_from_configured_rate(monkeypatch):
    upload_id = uuid4()
    clock = {"now": 100.0}
    monkeypatch.setattr(indexing_jobs, "GEMINI_EMBEDDING_REQUESTS_PER_MINUTE", 90)
    monkeypatch.setattr(indexing_jobs, "monotonic", lambda: clock["now"])
    prepare_indexing_estimate(upload_id, 180)

    assert get_indexing_progress(upload_id) == {
        "estimated_seconds_remaining": 120,
        "progress_percent": 0,
    }

    entered = Event()
    release = Event()
    state = {"value": "pending"}

    def blocking_indexer(_upload_id):
        entered.set()
        assert release.wait(timeout=2)

    worker = Thread(
        target=run_indexing_job,
        args=(upload_id,),
        kwargs={
            "indexer": blocking_indexer,
            "transition": _state_transition(state),
        },
    )
    worker.start()
    assert entered.wait(timeout=2)
    clock["now"] = 125.0
    assert get_indexing_progress(upload_id) == {
        "estimated_seconds_remaining": 95,
        "progress_percent": 20,
    }
    release.set()
    worker.join(timeout=2)

    assert get_indexing_progress(upload_id) is None


def test_status_storage_failure_does_not_escape_background_job():
    upload_id = uuid4()

    def fail_transition(*_args):
        raise RuntimeError("database unavailable")

    assert not run_indexing_job(
        upload_id,
        indexer=lambda _upload_id: None,
        transition=fail_transition,
    )
    assert not is_indexing_active(upload_id)


def test_duplicate_concurrent_job_does_not_index_twice():
    upload_id = uuid4()
    state = {"value": "pending"}
    entered = Event()
    release = Event()
    calls = []

    def blocking_indexer(value):
        calls.append(value)
        entered.set()
        assert release.wait(timeout=2)

    worker = Thread(
        target=run_indexing_job,
        args=(upload_id,),
        kwargs={
            "indexer": blocking_indexer,
            "transition": _state_transition(state),
        },
    )
    worker.start()
    assert entered.wait(timeout=2)
    assert is_indexing_active(upload_id)
    assert not run_indexing_job(
        upload_id,
        indexer=blocking_indexer,
        transition=_state_transition(state),
    )
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert calls == [upload_id]
    assert state["value"] == "ready"
