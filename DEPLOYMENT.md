# RetailIQ V1 deployment

RetailIQ V1 uses one Render web service and one persistent disk. Keep the backend at one instance: CSV files, ChromaDB, indexing locks, and cleanup coordination are local to that instance.

## Render backend

- Blueprint: `render.yaml`
- Root directory: repository root
- Build: `pip install -r backend/requirements.txt`
- Start: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Persistent disk: mount `/var/data/retailiq`, 10 GB minimum
- Instance: enough memory for Sentence Transformers/PyTorch (verify with the selected Render plan before launch)

Required secrets:

```text
DATABASE_URL=<pooled Neon PostgreSQL URL>
DATABASE_URL_UNPOOLED=<direct Neon PostgreSQL URL>
OPENROUTER_API_KEY=<server-side key>
ALLOWED_ORIGINS=https://<your-vercel-domain>
```

The blueprint supplies the model, secure cross-site cookie settings, 30-minute TTL, and persistent storage paths. Do not expose database or OpenRouter secrets to Vercel. Before public launch, delete any legacy `upload_metadata` rows whose `user_email` is null; only session-owned uploads are intended for V1.

Cross-site cookies use `SameSite=None; Secure`. Some browsers block third-party cookies; custom frontend/API subdomains under the same parent domain are preferred for reliable production sessions. Set `ALLOWED_ORIGINS` to exact origins, never `*`.

## Vercel frontend

- Root directory: `frontend`
- Framework preset: Vite
- Install: `npm install`
- Build: `npm run build`
- Output: `dist`
- Environment: `VITE_API_BASE_URL=https://<your-render-service>`

`frontend/vercel.json` provides the SPA fallback. No production URL is committed.

## Operational limits

Defaults are 10 MB, 50,000 rows, 200 columns, two concurrent indexing jobs, and 12 questions per session per minute. Override with the environment variables documented in `.env.example`. Cleanup runs once per minute inside the backend; Render should also use an external scheduled request or job if the web service may remain suspended longer than the expiry window.
