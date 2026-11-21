# RetailIQ V1 deployment: Render Free + Vercel

This configuration targets one Render **Free** web service (`0.1 CPU`, `512 MB RAM`) and a separately deployed Vercel frontend. Neon PostgreSQL stores temporary session/upload metadata, while uploaded CSVs and Chroma indexes use Render's ephemeral `/tmp` filesystem.

## Render Free viability

The embedding migration removes local Sentence Transformers, PyTorch inference, and model-weight downloads. Document and query embeddings are requested from Google's `gemini-embedding-001` API at 768 dimensions, while ChromaDB remains local.

This makes the backend materially more suitable for 512 MB than the former MiniLM implementation, but it is not a measured memory guarantee. ChromaDB, Pandas dataframes, CSV parsing, and concurrent requests still consume process memory. The Free blueprint therefore limits uploads to 5 MB/5,000 rows and indexing to one job at a time. Validate peak RSS on Render with representative CSVs before treating the service as production-ready.

## Render backend settings

Create the service from the repository-root `render.yaml`:

- Service type: Web Service
- Runtime: Python 3.10.14
- Plan: Free
- Root directory: repository root
- Build: `pip install -r backend/requirements.txt`
- Start: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Instances/workers: one
- Persistent disk: none

Required secrets:

```text
DATABASE_URL=<pooled Neon PostgreSQL URL>
DATABASE_URL_UNPOOLED=<direct Neon PostgreSQL URL>
GEMINI_API_KEY=<server-side Google Gemini API key>
OPENROUTER_API_KEY=<server-side OpenRouter key>
ALLOWED_ORIGINS=https://<your-vercel-domain>
```

Embedding configuration supplied by the blueprint:

```text
EMBEDDING_BATCH_SIZE=16
GEMINI_EMBEDDING_TIMEOUT_SECONDS=30
GEMINI_EMBEDDING_RETRY_ATTEMPTS=4
GEMINI_EMBEDDING_BACKOFF_SECONDS=1
GEMINI_EMBEDDING_MAX_BACKOFF_SECONDS=8
```

The Google API key is read only by the FastAPI backend and must not be exposed through Vite variables. Transient timeouts, HTTP 429 quota responses, and server errors are retried with bounded exponential backoff. Persistent failures mark indexing as retryable or return a generic retrieval-unavailable response without exposing provider details.

No embedding model is downloaded during build or per request. The first embedding request initializes the lightweight Google SDK client and makes a network call, so its latency depends on Render wake-up time, Google API latency, and current quota. Render Free service spin-down remains the dominant cold-start risk.

## Index version and ephemeral restarts

Gemini vectors use a versioned Chroma collection, `retailiq_gemini_001_768_v1`, and an `.embedding-index-version` marker. The former 384-dimensional MiniLM collection is never opened by the new retrieval path. At startup, a missing or mismatched marker changes durable `ready` statuses to `failed`, requiring reindexing or re-upload.

Render Free loses runtime files on every restart, redeploy, and idle spin-down. RetailIQ treats CSV and Chroma data as disposable:

1. Startup reconciliation deletes Neon upload metadata and anonymous-session rows when their CSV is gone.
2. Upload-history requests repeat that check if Neon was unavailable during startup.
3. A missing or wrong-version Chroma index invalidates durable `ready` statuses.
4. Index-status polling verifies that a supposedly ready upload still has chunks.
5. Retrieval filters by upload ID and every API route enforces anonymous-session ownership.

After a restart, affected users should expect to upload their CSV again. A 30-minute browser cookie does not make ephemeral files durable.

## Vercel frontend

- Root directory: `frontend`
- Framework: Vite
- Install: `npm install`
- Build: `npm run build`
- Output: `dist`
- Environment: `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`

`frontend/vercel.json` supplies the SPA fallback. Configure `ALLOWED_ORIGINS` with the exact Vercel origin. Secure cross-site cookies require `SameSite=None`, `Secure`, credentialed CORS, and HTTPS; related custom frontend/API subdomains reduce third-party-cookie problems.

## Remaining limitations

- Render Free is non-production infrastructure and may restart or spin down the service.
- CSVs and Chroma indexes disappear after restart, so users must re-upload.
- Embedding availability and indexing throughput depend on Gemini API quota and outbound network access.
- One indexing job avoids local memory spikes but serializes simultaneous uploads.
- The 5,000-row cap bounds cost and memory; larger datasets need a different tier or storage/indexing design.
- Production readiness still requires observed Render RSS, latency, quota, and concurrent-user load tests.
