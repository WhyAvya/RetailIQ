## Upload indexing lifecycle

New CSV uploads are saved with `pending` status and indexed by the existing MiniLM/ChromaDB pipeline in a FastAPI in-process background task. Status moves through `pending`, `indexing`, and then `ready` or `failed`. Check `GET /api/uploads/{upload_id}/index-status`; retry a failed or interrupted job with `POST /api/uploads/{upload_id}/index`. Index writes use deterministic document IDs and Chroma upserts, so retries do not create duplicate records.

These jobs are intentionally infrastructure-free and are not durable across a backend restart. A process restart can leave a database record in `pending` or `indexing`; the status endpoint reports such an inactive job as retryable, and the retry endpoint safely requeues it. For multi-process or production-scale execution, a durable external job queue would be required.
