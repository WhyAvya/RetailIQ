from collections.abc import Callable
from datetime import date
from typing import Any
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
from backend.upload_metadata import (
    DatabaseOperationError,
    build_dataset_preview,
    get_upload_metadata,
    list_upload_metadata,
    persist_upload_or_cleanup,
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
async def upload_csv(file: UploadFile | None = File(default=None)):
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
