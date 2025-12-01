"""FastAPI router for sentiment analysis and RAG query endpoints."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, constr, validator

from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.storage.models import SentimentCache as SentimentCacheModel
from modules.agent.agent_client import call_mcp_tool

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Sentiment"])

MAX_TEXT_LENGTH = 4000
MAX_BATCH = 32


def _validate_symbol(symbol: str) -> str:
    """Validate and normalize crypto symbol input."""
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _shape_sentiment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize sentiment payload into stable frontend fields."""
    if not isinstance(payload, dict):
        return {"raw": payload}
    return {
        "sentiment": (payload.get("sentiment") or payload.get("overall") or "UNKNOWN").upper(),
        "confidence": payload.get("confidence") or payload.get("top_confidence"),
        "scores": payload.get("scores") or {
            "bearish": payload.get("bearish_score"),
            "bullish": payload.get("bullish_score"),
            "neutral": payload.get("neutral_score"),
        },
        "raw_text": payload.get("raw_text"),
    }


class SentimentTextRequest(BaseModel):
    """Request model for single text sentiment analysis."""

    text: constr(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)


class SentimentBatchRequest(BaseModel):
    """Request model for batch text sentiment analysis."""
    texts: List[constr(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)]

    @validator("texts")
    def validate_batch(cls, values):
        if not values:
            raise ValueError("texts must contain at least one entry.")
        if len(values) > MAX_BATCH:
            raise ValueError(f"texts cannot exceed {MAX_BATCH} entries.")
        return values


class RagQueryRequest(BaseModel):
    """Request model for RAG query."""

    query: constr(strip_whitespace=True, min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)


@router.post("/text")
async def analyze_single_text(request: SentimentTextRequest):
    """Analyze sentiment of a single text input."""
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info("[%s] Sentiment text analysis request", request_id)

    try:
        payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_sentiment",
            {"text": request.text},
        )
    except Exception as exc:
        logger.error("[%s] Sentiment MCP failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Sentiment service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    shaped = _shape_sentiment_payload(payload or {})
    shaped.update({"request_id": request_id, "duration_ms": duration_ms})
    return shaped


@router.post("/batch")
async def analyze_text_batch(request: SentimentBatchRequest):
    """Analyze sentiment of multiple texts in batch."""
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info("[%s] Sentiment batch analysis (%s texts)", request_id, len(request.texts))

    try:
        payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_sentiment_batch",
            {"texts": request.texts},
        )
    except Exception as exc:
        logger.error("[%s] Batch sentiment MCP failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Sentiment batch service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    response = {
        "request_id": request_id,
        "duration_ms": duration_ms,
        "overview": _shape_sentiment_payload(payload or {}),
        "individual_results": payload.get("individual_results", []) if isinstance(payload, dict) else [],
    }
    return response


@router.get("/asset/{symbol}")
async def get_asset_sentiment(
    symbol: str,
    k: int = Query(5, ge=1, le=20, description="Number of RAG contexts to retrieve."),
    refresh: bool = Query(
        False,
        description="When true, bypass sentiment caches to force fresh RAG + sentiment.",
    ),
):
    sanitized_symbol = _validate_symbol(symbol)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(
        "[%s] Asset sentiment request: symbol=%s k=%s refresh=%s",
        request_id,
        sanitized_symbol,
        k,
        refresh,
    )

    try:
        sentiment_payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_with_sources",
            {
                "query": f"Market sentiment for {sanitized_symbol} from recent crypto news and on-chain headlines",
                "k": k,
                "include_sources": True,
                "refresh": refresh,
            },
            use_cache=not refresh,
        )
    except Exception as exc:
        logger.error("[%s] Asset sentiment failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Asset sentiment service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    aggregated = sentiment_payload.get("aggregated", {}) if isinstance(sentiment_payload, dict) else {}
    sources = sentiment_payload.get("sources") if isinstance(sentiment_payload, dict) else None
    response_text = sentiment_payload.get("response") if isinstance(sentiment_payload, dict) else sentiment_payload

    # Save to cache for fast dashboard loading
    try:
        from sqlalchemy.dialects.postgresql import insert

        with get_timescale_db() as session:
            cache_data = {
                "symbol": sanitized_symbol,
                "generated_at": datetime.utcnow(),
                "top_sentiment": aggregated.get("top_sentiment") or aggregated.get("sentiment"),
                "top_confidence": aggregated.get("top_confidence") or aggregated.get("confidence"),
                "bullish_score": aggregated.get("bullish_score"),
                "bearish_score": aggregated.get("bearish_score"),
                "neutral_score": aggregated.get("neutral_score"),
                "sources": sources,
                "response": response_text if isinstance(response_text, str) else None,
            }
            stmt = insert(SentimentCacheModel).values(**cache_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_=cache_data,
            )
            session.execute(stmt)
            session.commit()
            logger.info("[%s] Saved sentiment to cache for %s", request_id, sanitized_symbol)
    except Exception as cache_exc:
        logger.warning("[%s] Failed to cache sentiment: %s", request_id, cache_exc)

    response = {
        "request_id": request_id,
        "symbol": sanitized_symbol,
        "duration_ms": duration_ms,
        "aggregated": {
            "top_sentiment": aggregated.get("top_sentiment") or aggregated.get("sentiment"),
            "top_confidence": aggregated.get("top_confidence") or aggregated.get("confidence"),
            "bearish_score": aggregated.get("bearish_score"),
            "bullish_score": aggregated.get("bullish_score"),
            "neutral_score": aggregated.get("neutral_score"),
        },
        "sources": sources,
        "response": response_text,
    }
    return response


@router.get("/asset/{symbol}/cached")
async def get_cached_sentiment(
    symbol: str,
    max_age_hours: int = Query(4, ge=1, le=24, description="Max age of cached sentiment in hours."),
):
    """Get cached sentiment for fast dashboard loading. Falls back to fresh if cache is stale."""
    sanitized_symbol = _validate_symbol(symbol)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    logger.info("[%s] Cached sentiment request: symbol=%s max_age=%sh", request_id, sanitized_symbol, max_age_hours)

    try:
        from datetime import timedelta
        from sqlalchemy import select

        with get_timescale_db() as session:
            min_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            stmt = (
                select(SentimentCacheModel)
                .where(
                    SentimentCacheModel.symbol == sanitized_symbol,
                    SentimentCacheModel.generated_at >= min_time,
                )
                .order_by(SentimentCacheModel.generated_at.desc())
                .limit(1)
            )
            cached = session.execute(stmt).scalar_one_or_none()

            if cached:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                return {
                    "request_id": request_id,
                    "symbol": sanitized_symbol,
                    "duration_ms": duration_ms,
                    "aggregated": {
                        "top_sentiment": cached.top_sentiment,
                        "top_confidence": cached.top_confidence,
                        "bearish_score": cached.bearish_score,
                        "bullish_score": cached.bullish_score,
                        "neutral_score": cached.neutral_score,
                    },
                    "sources": cached.sources,
                    "response": cached.response,
                    "generated_at": cached.generated_at.isoformat() if cached.generated_at else None,
                    "from_cache": True,
                }
    except Exception as exc:
        logger.warning("[%s] Cache lookup failed: %s", request_id, exc)

    # Fallback to fresh sentiment
    logger.info("[%s] Cache miss, fetching fresh sentiment", request_id)
    return await get_asset_sentiment(sanitized_symbol, k=5, refresh=False)


@router.get("/sources/recent")
async def get_recent_sources(
    k: int = Query(5, ge=1, le=20, description="Number of recent sentiment sources to retrieve."),
    refresh: bool = Query(
        False,
        description="When true, bypass caches to force fresh RAG + sentiment for recent sources.",
    ),
):
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(
        "[%s] Recent sentiment sources request: k=%s refresh=%s",
        request_id,
        k,
        refresh,
    )

    try:
        sentiment_payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_with_sources",
            {
                "query": "Recent crypto market sentiment and news across all assets from the last few days",
                "k": k,
                "include_sources": True,
                "refresh": refresh,
            },
            use_cache=not refresh,
        )
    except Exception as exc:
        logger.error("[%s] Recent sentiment sources failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Recent sentiment sources service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    aggregated = sentiment_payload.get("aggregated", {}) if isinstance(sentiment_payload, dict) else {}
    response = {
        "request_id": request_id,
        "duration_ms": duration_ms,
        "aggregated": {
            "top_sentiment": aggregated.get("top_sentiment") or aggregated.get("sentiment"),
            "top_confidence": aggregated.get("top_confidence") or aggregated.get("confidence"),
            "bearish_score": aggregated.get("bearish_score"),
            "bullish_score": aggregated.get("bullish_score"),
            "neutral_score": aggregated.get("neutral_score"),
        }
        if aggregated
        else None,
        "sources": sentiment_payload.get("sources") if isinstance(sentiment_payload, dict) else None,
        "response": sentiment_payload.get("response") if isinstance(sentiment_payload, dict) else sentiment_payload,
    }
    return response


@router.post("/rag-query")
async def query_sentiment_rag(request: RagQueryRequest):
    """Query RAG system for crypto sentiment insights."""
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info("[%s] RAG query: %s", request_id, request.query[:120])

    try:
        rag_payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "query_rag",
            {"query": request.query, "k": request.k},
        )
    except Exception as exc:
        logger.error("[%s] RAG query failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="RAG service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    response = {
        "request_id": request_id,
        "duration_ms": duration_ms,
        "query": request.query,
        "answer": rag_payload.get("response") if isinstance(rag_payload, dict) else rag_payload,
        "contexts": rag_payload.get("contexts") if isinstance(rag_payload, dict) else [],
    }
    return response


def _fetch_fng_from_db() -> Dict[str, Any]:
    """Fetch Fear & Greed Index directly from database (bypasses MCP for speed)."""
    from data.storage.models import IngestionJob as IngestionJobModel

    with get_timescale_db() as db:
        latest = db.query(IngestionJobModel).filter(
            IngestionJobModel.pipeline == 'fear_greed'
        ).order_by(IngestionJobModel.last_success.desc()).first()

        if not latest or not (details := latest.details) or not isinstance(details, dict):
            return None

        value = float(details.get("value", 50))

        # Classify sentiment based on FNG value
        if value >= 75:
            sentiment = "EXTREME GREED"
            market_bias = "BEARISH"  # Contrarian indicator
        elif value >= 55:
            sentiment = "GREED"
            market_bias = "CAUTION"
        elif value >= 45:
            sentiment = "NEUTRAL"
            market_bias = "NEUTRAL"
        elif value >= 25:
            sentiment = "FEAR"
            market_bias = "OPPORTUNITY"
        else:
            sentiment = "EXTREME FEAR"
            market_bias = "BULLISH"  # Contrarian indicator

        return {
            "value": details.get("value"),
            "current_value": details.get("value"),
            "sentiment": sentiment,
            "classification": details.get("value_classification"),
            "market_bias": market_bias,
            "timestamp": latest.last_success.isoformat() if latest.last_success else None,
            "last_updated": latest.last_run.isoformat() if latest.last_run else None,
        }


@router.get("/fng/current")
async def get_fng_current():
    """Get current Fear & Greed Index value with market bias."""
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info("[%s] FNG current request", request_id)

    try:
        # Direct DB query - bypasses MCP for faster response
        data = _fetch_fng_from_db()
        if not data:
            data = {"error": "No FNG data available", "message": "Fear & Greed data unavailable"}
    except Exception as exc:
        logger.error("[%s] FNG DB fetch failed: %s", request_id, exc, exc_info=True)
        data = {"error": f"DB error: {exc}", "message": "Fear & Greed data unavailable"}

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    response = {
        "request_id": request_id,
        "duration_ms": duration_ms,
        "fng": data,
        "value": data.get("current_value") if isinstance(data, dict) else None,
        "sentiment": data.get("sentiment") or data.get("classification") if isinstance(data, dict) else None,
        "market_bias": data.get("market_bias"),
        "last_updated": data.get("last_updated") or data.get("timestamp"),
    }
    return response
