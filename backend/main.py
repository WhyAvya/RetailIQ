from collections.abc import Callable
from datetime import date
from typing import Any, Literal
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
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
    transition_upload_status,
)
from evaluation.report_service import (
    EvaluationReportUnavailableError,
    load_evaluation_dashboard_report,
)

app = FastAPI(title="RetailIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/api/upload")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No CSV file was provided.")

    try:
        file_size = await validate_csv_upload(file)
        dataframe = await run_in_threadpool(read_csv_dataframe, file.file)
        profile = await run_in_threadpool(profile_dataframe, dataframe)
        _, preprocessing = await run_in_threadpool(preprocess_dataframe, dataframe)
        stored_upload = await run_in_threadpool(
            store_original_csv,
            file.file,
            file.filename,
        )
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
async def upload_index_status(upload_id: UUID):
    try:
        metadata = await run_in_threadpool(get_upload_metadata, upload_id)
    except DatabaseOperationError as exc:
        raise HTTPException(status_code=503, detail="Indexing status is temporarily unavailable.") from exc
    if metadata is None:
        raise HTTPException(status_code=404, detail="The requested upload was not found.")
    return _index_status_payload(metadata, is_indexing_active(upload_id))


@app.post("/api/uploads/{upload_id}/index", status_code=status.HTTP_202_ACCEPTED)
async def retry_upload_indexing(upload_id: UUID, background_tasks: BackgroundTasks):
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
async def upload_history():
    try:
        uploads = await run_in_threadpool(list_upload_metadata)
    except DatabaseOperationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload history is temporarily unavailable.",
        ) from exc

    return {
        "development_only": True,
        "uploads": uploads,
    }


@app.get("/api/uploads/{upload_id}/preview")
async def dataset_preview(upload_id: UUID):
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
):
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
):
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
):
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
):
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
):
    return await _analytics_response(
        calculate_selected_profit_analysis,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.get("/api/analytics/{upload_id}/filter-options")
async def analytics_filter_options(upload_id: UUID):
    return await _analytics_response(calculate_selected_filter_options, upload_id)


@app.get("/api/analytics/{upload_id}/dashboard")
async def analytics_dashboard(
    upload_id: UUID,
    region: str | None = None,
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
):
    return await _analytics_response(
        calculate_selected_dashboard,
        upload_id,
        _dashboard_filters(region, category, start_date, end_date),
    )


@app.post("/api/rag/retrieve", response_model=RetrievalResponse)
async def rag_retrieve(request: RetrievalRequest) -> RetrievalResponse:
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
async def rag_ask(request: RAGAskRequest) -> RAGAskResponse:
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


@app.get("/api/evaluation/report")
async def evaluation_report() -> dict[str, Any]:
    try:
        return await run_in_threadpool(load_evaluation_dashboard_report)
    except EvaluationReportUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Evaluation reports are temporarily unavailable.",
        ) from exc
