"""
Crypto Analytics API main application module.

FastAPI application providing unified API surface for forecasting,
sentiment analysis, on-chain analytics, and autonomous agent services.
"""

import logging
from datetime import datetime
from typing import Dict

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from core.config import settings
from core.database import get_metadata_db, get_timescale_db
from core.exceptions import APIError, CryptoAnalyticsError
from core.logging_config import setup_logging
from modules.agent.agent_client import call_mcp_tool
from services.agent import router as agent_router
from services.dashboard import router as dashboard_router
from services.onchain import router as onchain_router
from services.price import router as price_router
from services.sentiment import router as sentiment_router

setup_logging()
logger = logging.getLogger(__name__)

tags_metadata = [
    {"name": "Price", "description": "Market data forecasting and time-series endpoints."},
    {"name": "Sentiment", "description": "Sentiment, RAG, and news intelligence endpoints."},
    {"name": "Onchain", "description": "On-chain metrics, whale activity, and TA pattern endpoints."},
    {"name": "Agent", "description": "Unified AI agent for combined insights and strategies."},
    {"name": "Dashboard", "description": "Frontend-oriented aggregates for the analytics dashboard."},
]

app = FastAPI(
    title="Crypto Analytics API",
    version="2.0.0",
    description="Unified API surface for forecasting, sentiment, on-chain analytics, and the autonomous agent.",
    openapi_tags=tags_metadata,
)

# --- CORS --------------------------------------------------------------------
cors_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers -----------------------------------------------------------------
def _check_db_health() -> bool:
    """Check database connectivity for both TimescaleDB and metadata DB."""
    try:
        with get_timescale_db() as ts_session:
            ts_session.execute(text("SELECT 1"))
        with get_metadata_db() as meta_session:
            meta_session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


def _check_redis_health() -> bool:
    """Check Redis connectivity with timeout."""
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
        return True
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return False


def _base_health_payload() -> Dict[str, str]:
    """Build base health check payload with DB and Redis status."""
    return {
        "db": "ok" if _check_db_health() else "down",
        "redis": "ok" if _check_redis_health() else "down",
        "time": datetime.utcnow().isoformat(),
        "env": settings.APP_ENV,
    }


# --- Exception Handlers ------------------------------------------------------
@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(APIError)
async def handle_api_error(_: Request, exc: APIError):
    status = exc.status_code or 502
    logger.warning("APIError: %s (status=%s)", exc.message, status)
    return JSONResponse(status_code=status, content={"error": exc.message})


@app.exception_handler(CryptoAnalyticsError)
async def handle_crypto_error(_: Request, exc: CryptoAnalyticsError):
    logger.warning("CryptoAnalyticsError: %s", exc)
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again later."},
    )


# --- Routes ------------------------------------------------------------------
app.include_router(price_router, prefix="/price")
app.include_router(sentiment_router, prefix="/sentiment")
app.include_router(onchain_router, prefix="/onchain")
app.include_router(agent_router, prefix="/agent")
app.include_router(dashboard_router, prefix="/dashboard")


@app.get("/")
async def root():
    return {
        "message": "Crypto Analytics API is running",
        "version": app.version,
        "env": settings.APP_ENV,
        "time": datetime.utcnow().isoformat(),
    }


@app.get("/healthz")
async def healthcheck():
    payload = _base_health_payload()

    # Best-effort sentiment MCP/vector-store health: do not fail overall health if this errors.
    try:
        stats = await call_mcp_tool(
            "crypto-sentiment-server",
            "get_stats",
            use_cache=False,
        )
        if isinstance(stats, dict):
            payload["sentiment_mcp"] = "ok"
            payload["sentiment_stats"] = stats.get("raw") or stats.get("raw_text")
        else:
            payload["sentiment_mcp"] = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Sentiment MCP health check failed: %s", exc)
        payload["sentiment_mcp"] = "down"

    if payload["db"] == "ok" and payload["redis"] == "ok":
        return payload
    raise HTTPException(status_code=503, detail=payload)

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI on port %s in %s mode", settings.APP_PORT, settings.APP_ENV)
    uvicorn.run(app, host="0.0.0.0", port=settings.APP_PORT)
