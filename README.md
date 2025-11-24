# Crypto AI Analytics Platform

A monolithic, MCP-integrated AI platform that delivers forecasting, sentiment, on-chain analytics, and explainable insights for cryptocurrencies, all orchestrated by a central LLM-powered agent.

## Production Deployment

### Backend (Cloud Run)

- Image is built from `infrastructure/Dockerfile.backend` and is compatible with Cloud Run.
- The container listens on the `PORT` environment variable (default `8000`) via `uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
- For production, build and push the image to a registry (e.g. Artifact Registry), then create a Cloud Run service using that image and configure environment variables/secrets (Postgres, Redis, Qdrant, MLflow, API keys).
- Set Cloud Run HTTP health checks to `/health` or `/healthz` on the service.

### Frontend (Firebase Hosting + CDN)

- The React/Vite frontend lives in `frontend/`.
- Production builds read `VITE_API_URL` from `frontend/.env.production`, which is set to `/api` so the browser uses same-origin calls.
- `firebase.json` configures Firebase Hosting to serve the built SPA (`frontend/dist`) and to proxy `/api/**` requests to a Cloud Run service named `crypto-analytics-backend` in region `us-central1` (adjust `serviceId` and `region` to match your environment).
- Typical deployment flow:
  - `cd frontend && npm ci && npm run build`
  - `cd .. && firebase deploy --only hosting` (with `firebase-tools` installed and `.firebaserc` pointing to your Firebase project).

### Local / Docker Compose

- `docker-compose.yml` orchestrates the backend, TimescaleDB, Redis, Qdrant, MLflow, and an Nginx-based frontend container.
- Run `docker compose up --build` from the repo root for a full local stack:
  - Backend API: `http://localhost:8000`
  - Frontend (built and served via Nginx): `http://localhost:3000`
  - MLflow UI: `http://localhost:5000`
