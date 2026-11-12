import asyncio
from io import BytesIO
from uuid import uuid4

import pandas as pd
from fastapi import BackgroundTasks, UploadFile

import backend.main as main_module
from backend.file_storage import StoredUpload


def test_upload_queues_indexing_with_pending_status(monkeypatch, tmp_path):
    upload_id = uuid4()
    contents = b"product,units\nTea,4\n"
    stored = StoredUpload(upload_id, "small.csv", tmp_path / f"{upload_id}.csv")

    async def validate(_file):
        return len(contents)

    monkeypatch.setattr(main_module, "validate_csv_upload", validate)
    monkeypatch.setattr(
        main_module,
        "read_csv_dataframe",
        lambda _file: pd.DataFrame({"product": ["Tea"], "units": [4]}),
    )
    monkeypatch.setattr(main_module, "profile_dataframe", lambda _frame: {"rows": 1})
    monkeypatch.setattr(main_module, "preprocess_dataframe", lambda frame: (frame, {}))
    monkeypatch.setattr(main_module, "store_original_csv", lambda *_args: stored)

    def persist(_stored, _size, processing_status):
        assert processing_status == "pending"
        return {"status": processing_status}

    monkeypatch.setattr(main_module, "persist_upload_or_cleanup", persist)
    background_tasks = BackgroundTasks()
    upload = UploadFile(filename="small.csv", file=BytesIO(contents))

    response = asyncio.run(main_module.upload_csv(background_tasks, upload))

    assert response["upload_id"] == str(upload_id)
    assert response["status"] == "pending"
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is main_module.run_indexing_job
    assert background_tasks.tasks[0].args == (upload_id,)


def test_index_status_reports_active_and_retryable(monkeypatch):
    upload_id = uuid4()
    monkeypatch.setattr(
        main_module,
        "get_upload_metadata",
        lambda _upload_id: {"upload_id": str(upload_id), "status": "failed"},
    )
    monkeypatch.setattr(main_module, "is_indexing_active", lambda _upload_id: False)

    response = asyncio.run(main_module.upload_index_status(upload_id))

    assert response == {
        "upload_id": str(upload_id),
        "state": "failed",
        "ready": False,
        "active": False,
        "retryable": True,
    }


def test_retry_endpoint_queues_one_recoverable_job(monkeypatch):
    upload_id = uuid4()
    transitions = []
    monkeypatch.setattr(
        main_module,
        "get_upload_metadata",
        lambda _upload_id: {"upload_id": str(upload_id), "status": "failed"},
    )
    monkeypatch.setattr(main_module, "is_indexing_active", lambda _upload_id: False)

    def transition(*args):
        transitions.append(args)
        return True

    monkeypatch.setattr(main_module, "transition_upload_status", transition)
    background_tasks = BackgroundTasks()

    response = asyncio.run(main_module.retry_upload_indexing(upload_id, background_tasks))

    assert response["state"] == "pending"
    assert transitions == [
        (upload_id, "pending", ("pending", "indexing", "failed", "processed"))
    ]
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].args == (upload_id,)


def test_retry_endpoint_does_not_queue_when_job_is_active(monkeypatch):
    upload_id = uuid4()
    monkeypatch.setattr(
        main_module,
        "get_upload_metadata",
        lambda _upload_id: {"upload_id": str(upload_id), "status": "indexing"},
    )
    monkeypatch.setattr(main_module, "is_indexing_active", lambda _upload_id: True)
    background_tasks = BackgroundTasks()

    response = asyncio.run(main_module.retry_upload_indexing(upload_id, background_tasks))

    assert response["active"] is True
    assert background_tasks.tasks == []
