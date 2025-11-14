import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, constr, validator

from core.logging_config import setup_logging
from modules.agent.agent_client import call_mcp_tool

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Sentiment"])

MAX_TEXT_LENGTH = 4000
MAX_BATCH = 32


def _validate_symbol(symbol: str) -> str:
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _shape_sentiment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    text: constr(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)


class SentimentBatchRequest(BaseModel):
    texts: List[constr(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)]

    @validator("texts")
    def validate_batch(cls, values):
        if not values:
            raise ValueError("texts must contain at least one entry.")
        if len(values) > MAX_BATCH:
            raise ValueError(f"texts cannot exceed {MAX_BATCH} entries.")
        return values


class RagQueryRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)


@router.post("/text")
async def analyze_single_text(request: SentimentTextRequest):
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
    refresh: bool = Query(False, description="Trigger ingestion before fetching sentiment."),
    days_back: int = Query(7, ge=1, le=90, description="Days of history to ingest when refresh=true."),
):
    sanitized_symbol = _validate_symbol(symbol)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(
        "[%s] Asset sentiment request: symbol=%s k=%s refresh=%s days_back=%s",
        request_id,
        sanitized_symbol,
        k,
        refresh,
        days_back,
    )

    try:
        if refresh:
            await call_mcp_tool(
                "crypto-sentiment-server",
                "ingest_documents",
                {"days_back": days_back},
                use_cache=False,
            )
        sentiment_payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_with_sources",
            {
                "query": f"Market sentiment for {sanitized_symbol} from recent crypto news and on-chain headlines",
                "k": k,
                "include_sources": True,
            },
        )
    except Exception as exc:
        logger.error("[%s] Asset sentiment failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Asset sentiment service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    aggregated = sentiment_payload.get("aggregated", {}) if isinstance(sentiment_payload, dict) else {}
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
        "sources": sentiment_payload.get("sources") if isinstance(sentiment_payload, dict) else None,
        "response": sentiment_payload.get("response") if isinstance(sentiment_payload, dict) else sentiment_payload,
    }
    return response


@router.post("/rag-query")
async def query_sentiment_rag(request: RagQueryRequest):
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
