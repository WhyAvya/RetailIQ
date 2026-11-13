from collections.abc import Callable
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import date
from time import monotonic
from collections import defaultdict, deque
from typing import Any, Literal
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from backend.csv_utils import CSVValidationError, read_csv_dataframe, validate_csv_upload
from backend.data_processing import preprocess_dataframe, profile_dataframe
from backend.analytics_filters import AnalyticsFilterError, DashboardFilters
from backend.dataset_selection import (
    DatasetInvalidError,
    DatasetNotFoundError,
    DatasetSelectionError,
    DatasetUnavailableError,
    calculate_selected_category_performance,
    calculate_selected_currency_metadata,
    calculate_selected_dashboard,
    calculate_selected_dataset_kpis,
    calculate_selected_filter_options,
    calculate_selected_monthly_revenue,
    calculate_selected_profit_analysis,
    calculate_selected_sales_by_region,
)
from backend.file_storage import FileStorageError, store_original_csv
from backend.indexing_jobs import is_indexing_active, run_indexing_job
from backend.llm_service import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.rag_preparation import MetadataValue
from backend.rag_pipeline import (
    InvalidPipelineRequestError,
    PipelineDatasetNotFoundError,
    RAGPipelineError,
    answer_question,
)
from backend.rag_retrieval import (
    MAX_TOP_K,
    DatasetNotIndexedError,
    InvalidRetrievalRequestError,
    RetrievalUnavailableError,
    retrieve_semantic_matches,
)
from backend.upload_metadata import (
    DatabaseOperationError,
    build_dataset_preview,
    get_upload_metadata,
    list_upload_metadata,
    persist_upload_or_cleanup,
    recover_interrupted_uploads,
    transition_upload_status,
)
from backend.anonymous_sessions import (
    AnonymousIdentity,
    SessionStoreError,
    new_identity,
    expired_session_hashes,
    remove_session_records,
    session_uploads,
    token_hash,
    touch_session,
)
from backend.runtime_config import (
    ALLOWED_ORIGINS,
    CSRF_COOKIE_NAME,
    MAX_CSV_COLUMNS,
    MAX_CSV_ROWS,
    MAX_QUESTIONS_PER_MINUTE,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
    SESSION_TTL_SECONDS,
)
from backend.file_storage import UPLOAD_DIRECTORY
from backend.vector_store import ChromaVectorStore
from evaluation.report_service import (
    EvaluationReportUnavailableError,
    load_evaluation_dashboard_report,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await run_in_threadpool(recover_interrupted_uploads)
        await run_in_threadpool(_cleanup_expired_sessions)
    except (SessionStoreError, DatabaseOperationError):
        pass
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="RetailIQ API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_question_windows: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def anonymous_session_middleware(request: Request, call_next):
    if request.url.path in {"/", "/health"}:
        return await call_next(request)
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in ALLOWED_ORIGINS:
        return Response("Origin is not allowed.", status_code=403)
    existing_token = request.cookies.get(SESSION_COOKIE_NAME)
    csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME)
    identity = AnonymousIdentity(
        existing_token or new_identity().token,
        token_hash(existing_token) if existing_token else "",
        csrf_cookie or new_identity().csrf_token,
    )
    if not identity.token_hash:
        identity = AnonymousIdentity(identity.token, token_hash(identity.token), identity.csrf_token)
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and existing_token and origin:
        if not csrf_cookie or request.headers.get("x-csrf-token") != csrf_cookie:
            return Response("CSRF validation failed.", status_code=403)
    request.state.identity = identity
    try:
        await run_in_threadpool(touch_session, identity)
    except SessionStoreError:
        # Dataset operations still fail closed through their normal metadata access.
        # Keeping request validation available also makes health/debug endpoints useful
        # during a transient database outage.
        pass
    response = await call_next(request)
    if getattr(request.state, "clear_session", False):
        return response
    response.set_cookie(
        SESSION_COOKIE_NAME,
        identity.token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite=SESSION_COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE_NAME,
        identity.csrf_token,
        max_age=SESSION_TTL_SECONDS,
        httponly=False,
        secure=SESSION_COOKIE_SECURE,
        samesite=SESSION_COOKIE_SAMESITE,
        path="/",
    )
    return response


async def _assert_upload_owner(request: Request | None, upload_id: UUID) -> None:
    if request is None:
        return
    try:
        metadata = await run_in_threadpool(get_upload_metadata, upload_id)
    except DatabaseOperationError:
        return
    if metadata is not None and "owner_hash" in metadata and metadata.get("owner_hash") != request.state.identity.token_hash:
        raise HTTPException(status_code=404, detail="The requested upload was not found.")


def _delete_session_data(owner_hash: str, delete_session: bool = True) -> int:
    targets = [(record.upload_id, record.stored_file_path) for record in session_uploads(owner_hash)]
    store = None
    try:
        for upload_id, _stored_name in targets:
            store = store or ChromaVectorStore()
            store.delete(upload_id)
        for _upload_id, stored_name in targets:
            path = (UPLOAD_DIRECTORY.resolve() / stored_name).resolve()
            if path.parent != UPLOAD_DIRECTORY.resolve():
                raise SessionStoreError("Stored upload path is unsafe.")
            path.unlink(missing_ok=True)
    except Exception as exc:
        if isinstance(exc, SessionStoreError):
            raise
        raise SessionStoreError("Temporary dataset files could not be removed.") from exc
    removed = remove_session_records(owner_hash, delete_session=delete_session)
    if delete_session:
        _question_windows.pop(owner_hash, None)
    return len(removed)


def _cleanup_expired_sessions() -> int:
    deleted = 0
    for owner_hash in expired_session_hashes():
        deleted += _delete_session_data(owner_hash)
    return deleted


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(60)
        try:
            await run_in_threadpool(_cleanup_expired_sessions)
        except (SessionStoreError, DatabaseOperationError):
            pass


class RetrievalRequest(BaseModel):
    upload_id: UUID
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, strict=True)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("The retrieval query cannot be empty.")
        return stripped


class RetrievalMatchResponse(BaseModel):
    rank: int
    chunk_id: str
    document: str
    source_row: int | None
    metadata: dict[str, MetadataValue]
    similarity_score: float


class RetrievalResponse(BaseModel):
    upload_id: UUID
    query: str
    matches: list[RetrievalMatchResponse]


class RAGAskRequest(BaseModel):
    upload_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, strict=True)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("The question cannot be empty.")
        return stripped


class RAGSourceResponse(BaseModel):
    chunk_id: str
    source_row: int | None
    similarity_score: float
    metadata: dict[str, MetadataValue]
    document_excerpt: str


class RAGAskResponse(BaseModel):
    upload_id: UUID
    question: str
    answer: str
    mode: Literal["analytics", "semantic_rag", "unsupported"]
    sources: list[RAGSourceResponse]
    calculation: dict[str, Any] | None = None


def _analytics_http_exception(error: DatasetSelectionError) -> HTTPException:
    if isinstance(error, DatasetNotFoundError):
        return HTTPException(status_code=404, detail="The selected dataset was not found.")
    if isinstance(error, DatasetUnavailableError):
        return HTTPException(
            status_code=404,
            detail="The selected dataset file is unavailable.",
        )
    if isinstance(error, DatasetInvalidError):
        return HTTPException(
            status_code=422,
            detail="The selected dataset is not a readable CSV.",
        )
    return HTTPException(status_code=503, detail="Analytics are temporarily unavailable.")


def _dashboard_filters(
    region: str | None,
    category: str | None,
    start_date: date | None,
    end_date: date | None,
) -> DashboardFilters:
    return DashboardFilters(
        region=region,
        category=category,
        start_date=start_date,
        end_date=end_date,
    )


async def _analytics_response(
    calculator: Callable[..., dict[str, Any]],
    upload_id: UUID,
    filters: DashboardFilters | None = None,
) -> dict[str, Any]:
    try:
        if filters is None or filters.is_empty:
            return await run_in_threadpool(calculator, upload_id)
        return await run_in_threadpool(calculator, upload_id, filters=filters)
    except AnalyticsFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatasetSelectionError as exc:
        raise _analytics_http_exception(exc) from exc


@app.get("/")
def home():
    return {"message": "RetailIQ API is running successfully!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/session")
async def session_bootstrap(request: Request) -> dict[str, Any]:
    return {"csrf_token": request.state.identity.csrf_token, "expires_in": SESSION_TTL_SECONDS}


@app.post("/api/upload")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    request: Request = None,
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No CSV file was provided.")

    try:
        file_size = await validate_csv_upload(file)
        dataframe = await run_in_threadpool(read_csv_dataframe, file.file)
        if len(dataframe) > MAX_CSV_ROWS or len(dataframe.columns) > MAX_CSV_COLUMNS:
            raise CSVValidationError(
                f"CSV exceeds the limit of {MAX_CSV_ROWS:,} rows or {MAX_CSV_COLUMNS:,} columns."
            )
        profile = await run_in_threadpool(profile_dataframe, dataframe)
        _, preprocessing = await run_in_threadpool(preprocess_dataframe, dataframe)
        if request is not None:
            await run_in_threadpool(_delete_session_data, request.state.identity.token_hash, False)
        stored_upload = await run_in_threadpool(
            store_original_csv,
            file.file,
            file.filename,
        )
        if request is not None:
            metadata = await run_in_threadpool(
                persist_upload_or_cleanup,
                stored_upload,
                file_size,
                request.state.identity.token_hash,
                "pending",
            )
        else:
            metadata = await run_in_threadpool(
                persist_upload_or_cleanup,
                stored_upload,
                file_size,
                processing_status="pending",
            )
    except CSVValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except DatabaseOperationError as exc:
        raise HTTPException(
            status_code=503,
            detail="The upload could not be completed because metadata storage is unavailable.",
        ) from exc
    except SessionStoreError as exc:
        raise HTTPException(status_code=503, detail="Existing session data could not be replaced safely.") from exc
    finally:
        await file.close()

    background_tasks.add_task(run_indexing_job, stored_upload.upload_id)
    return {
        "filename": stored_upload.original_filename,
        "upload_id": str(stored_upload.upload_id),
        "size": file_size,
        "status": metadata["status"],
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns],
        "profile": profile,
        "preprocessing": preprocessing,
    }


def _index_status_payload(metadata: dict[str, Any], active: bool) -> dict[str, Any]:
    state = str(metadata["status"])
    return {
        "upload_id": metadata["upload_id"],
        "state": state,
        "ready": state == "ready",
        "active": active,
        "retryable": (state == "failed" or state in {"pending", "indexing"}) and not active,
    }


@app.get("/api/uploads/{upload_id}/index-status")
async def upload_index_status(upload_id: UUID, request: Request = None):
    await _assert_upload_owner(request, upload_id)
    try:
        metadata = await run_in_threadpool(get_upload_metadata, upload_id)
    except DatabaseOperationError as exc:
        raise HTTPException(status_code=503, detail="Indexing status is temporarily unavailable.") from exc
    if metadata is None:
        raise HTTPException(status_code=404, detail="The requested upload was not found.")
    return _index_status_payload(metadata, is_indexing_active(upload_id))


@app.post("/api/uploads/{upload_id}/index", status_code=status.HTTP_202_ACCEPTED)
async def retry_upload_indexing(upload_id: UUID, background_tasks: BackgroundTasks, request: Request = None):
    await _assert_upload_owner(request, upload_id)
    try:
        metadata = await run_in_threadpool(get_upload_metadata, upload_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="The requested upload was not found.")
        if metadata["status"] == "ready":
            return _index_status_payload(metadata, False)
        if is_indexing_active(upload_id):
            return _index_status_payload(metadata, True)
        queued = await run_in_threadpool(
            transition_upload_status,
            upload_id,
            "pending",
            ("pending", "indexing", "failed", "processed"),
        )
    except DatabaseOperationError as exc:
        raise HTTPException(status_code=503, detail="Indexing could not be queued.") from exc
    if not queued:
        raise HTTPException(status_code=409, detail="The dataset indexing state changed; refresh and try again.")
    background_tasks.add_task(run_indexing_job, upload_id)
    return {
        "upload_id": str(upload_id),
        "state": "pending",
        "ready": False,
        "active": False,
        "retryable": False,
    }


@app.get("/api/uploads")
async def upload_history(request: Request = None):
    try:
        if request is None:
            uploads = await run_in_threadpool(list_upload_metadata)
        else:
            uploads = await run_in_threadpool(list_upload_metadata, None, 1, request.state.identity.token_hash)
    except DatabaseOperationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload history is temporarily unavailable.",
        ) from exc

    return {
        "temporary_session": True,
        "uploads": uploads,
    }


@app.get("/api/uploads/{upload_id}/preview")
async def dataset_preview(upload_id: UUID, request: Request = None):
    await _assert_upload_owner(request, upload_id)
    try:
        metadata = await run_in_threadpool(get_upload_metadata, upload_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="The requested upload was not found.")
        return await run_in_threadpool(build_dataset_preview, metadata)
    except DatabaseOperationError as exc:
        raise HTTPException(
            status_code=503,
            detail="The dataset preview is temporarily unavailable.",
        ) from exc
    except (CSVValidationError, FileStorageError) as exc:
        raise HTTPException(
            status_code=404,
            detail="The stored dataset is unavailable or cannot be read.",
        ) from exc


@app.get("/api/analytics/{upload_id}/summary")
async def analytics_summary(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    request: Request = None,
):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(
        calculate_selected_dataset_kpis,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.get("/api/analytics/{upload_id}/monthly-revenue")
async def monthly_revenue_trend(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    request: Request = None,
):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(
        calculate_selected_monthly_revenue,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.get("/api/analytics/{upload_id}/sales-by-region")
async def regional_sales(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    request: Request = None,
):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(
        calculate_selected_sales_by_region,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.get("/api/analytics/{upload_id}/category-performance")
async def category_performance(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    request: Request = None,
):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(
        calculate_selected_category_performance,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.get("/api/analytics/{upload_id}/profit-analysis")
async def profit_analysis(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    request: Request = None,
):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(
        calculate_selected_profit_analysis,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.get("/api/analytics/{upload_id}/filter-options")
async def analytics_filter_options(upload_id: UUID, request: Request = None):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(calculate_selected_filter_options, upload_id)


@app.get("/api/analytics/{upload_id}/currency")
async def analytics_currency(upload_id: UUID, request: Request = None):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(calculate_selected_currency_metadata, upload_id)


@app.get("/api/analytics/{upload_id}/dashboard")
async def analytics_dashboard(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    request: Request = None,
):
    await _assert_upload_owner(request, upload_id)
    return await _analytics_response(
        calculate_selected_dashboard,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.post("/api/rag/retrieve", response_model=RetrievalResponse)
async def rag_retrieve(request: RetrievalRequest, http_request: Request) -> RetrievalResponse:
    await _assert_upload_owner(http_request, request.upload_id)
    try:
        metadata = await run_in_threadpool(get_upload_metadata, request.upload_id)
    except DatabaseOperationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Dataset metadata is temporarily unavailable.",
        ) from exc
    if metadata is None:
        raise HTTPException(status_code=404, detail="The selected dataset was not found.")

    try:
        matches = await run_in_threadpool(
            retrieve_semantic_matches,
            request.upload_id,
            request.query,
            request.top_k,
        )
    except InvalidRetrievalRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatasetNotIndexedError as exc:
        raise HTTPException(
            status_code=404,
            detail="The selected dataset has not been indexed for retrieval.",
        ) from exc
    except RetrievalUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Semantic retrieval is temporarily unavailable.",
        ) from exc

    return RetrievalResponse(
        upload_id=request.upload_id,
        query=request.query,
        matches=[
            RetrievalMatchResponse(
                rank=match.rank,
                chunk_id=match.chunk_id,
                document=match.document,
                source_row=match.source_row,
                metadata=match.metadata,
                similarity_score=match.similarity_score,
            )
            for match in matches
        ],
    )


@app.post("/api/rag/ask", response_model=RAGAskResponse)
async def rag_ask(request: RAGAskRequest, http_request: Request) -> RAGAskResponse:
    await _assert_upload_owner(http_request, request.upload_id)
    window = _question_windows[http_request.state.identity.token_hash]
    now = monotonic()
    while window and now - window[0] >= 60:
        window.popleft()
    if len(window) >= MAX_QUESTIONS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Question limit reached. Please wait before trying again.")
    window.append(now)
    try:
        result = await run_in_threadpool(
            answer_question,
            request.upload_id,
            request.question,
            request.top_k,
        )
    except PipelineDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail="The selected dataset was not found.") from exc
    except InvalidPipelineRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatasetNotIndexedError as exc:
        raise HTTPException(
            status_code=404,
            detail="The selected dataset has not been indexed for retrieval.",
        ) from exc
    except DatabaseOperationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Dataset metadata is temporarily unavailable.",
        ) from exc
    except DatasetSelectionError as exc:
        raise _analytics_http_exception(exc) from exc
    except RetrievalUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Semantic retrieval is temporarily unavailable.",
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="The language model is not configured.",
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail="The language model request timed out.") from exc
    except LLMRateLimitError as exc:
        raise HTTPException(
            status_code=503,
            detail="The language model is temporarily rate limited.",
        ) from exc
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The language model provider rejected authentication.",
        ) from exc
    except (LLMInvalidResponseError, LLMUnavailableError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The language model service returned an unusable response.",
        ) from exc
    except RAGPipelineError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG processing is temporarily unavailable.",
        ) from exc

    return RAGAskResponse(
        upload_id=UUID(result.upload_id),
        question=result.question,
        answer=result.answer,
        mode=result.mode,
        sources=[
            RAGSourceResponse(
                chunk_id=source.chunk_id,
                source_row=source.source_row,
                similarity_score=source.similarity_score,
                metadata=source.metadata,
                document_excerpt=source.document_excerpt,
            )
            for source in result.sources
        ],
        calculation=result.calculation,
    )


@app.delete("/api/session", status_code=status.HTTP_200_OK)
async def clear_session_data(request: Request, response: Response) -> dict[str, Any]:
    try:
        deleted = await run_in_threadpool(_delete_session_data, request.state.identity.token_hash)
    except SessionStoreError as exc:
        raise HTTPException(status_code=503, detail="Session data could not be cleared.") from exc
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    request.state.clear_session = True
    return {"cleared": True, "deleted_datasets": deleted}


@app.get("/api/evaluation/report")
async def evaluation_report() -> dict[str, Any]:
    try:
        return await run_in_threadpool(load_evaluation_dashboard_report)
    except EvaluationReportUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Evaluation reports are temporarily unavailable.",
        ) from exc
