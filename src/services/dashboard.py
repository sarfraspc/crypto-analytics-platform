"""FastAPI router for dashboard overview, backtest, and whale activity endpoints."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, select

from core.database import get_timescale_db
from data.storage.models import OnchainMetric as OnchainMetricModel
from core.logging_config import setup_logging
from data.storage.models import WhaleAlert as WhaleAlertModel
from modules.agent.agent_client import call_mcp_tool
from modules.agent.backtester import PortfolioBacktester
from .onchain import _shape_metrics as _shape_raw_onchain_metrics

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Dashboard"])

ALLOWED_WINDOWS = {"1h", "24h", "7d"}
BACKTESTER = PortfolioBacktester()


def _validate_symbol(symbol: str) -> str:
    """Validate and normalize crypto symbol input."""
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _shape_forecast(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize forecast payload into stable frontend fields."""
    if not isinstance(payload, dict):
        return {"raw_text": payload}
    return {
        "model_used": payload.get("model_used", "sarimax_v3"),
        "predicted_close": payload.get("predicted_close"),
        "raw_text": payload.get("raw_text"),
    }


def _shape_sentiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize sentiment payload into stable frontend fields."""
    if not isinstance(payload, dict):
        return {"raw": payload}
    aggregated = payload.get("aggregated", {})
    return {
        "top_sentiment": aggregated.get("top_sentiment") or aggregated.get("sentiment"),
        "top_confidence": aggregated.get("top_confidence") or aggregated.get("confidence"),
        "scores": aggregated.get("scores") or {
            "bearish": aggregated.get("bearish_score"),
            "bullish": aggregated.get("bullish_score"),
            "neutral": aggregated.get("neutral_score"),
        },
        "sources": payload.get("sources"),
    }


def _shape_onchain_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize on-chain metrics payload into stable frontend fields."""
    if not isinstance(payload, dict):
        return {"raw": payload}
    flattened = _shape_raw_onchain_metrics(payload)
    flow_trend = flattened.get("flow_trend_24h")
    # Map numeric flow trend into a human-readable dominant flow label.
    dominant_flow_label: str | None = None
    try:
        if flow_trend is not None:
            val = float(flow_trend)
            if val > 5:
                dominant_flow_label = "inflow-dominated"
            elif val < -5:
                dominant_flow_label = "outflow-dominated"
            else:
                dominant_flow_label = "balanced"
    except Exception:
        dominant_flow_label = None

    return {
        "whale_transactions": flattened.get("whale_transactions"),
        "inflow_usd": flattened.get("exchange_inflow_usd"),
        "outflow_usd": flattened.get("exchange_outflow_usd"),
        "market_pressure_index": flattened.get("market_pressure_index"),
        "market_bias": flattened.get("market_bias"),
        # Use short-term flow trend as a proxy for dominant flow direction.
        "dominant_flow": dominant_flow_label,
        "flow_trend_24h": flow_trend,
    }


def _fetch_onchain_metrics_from_db(chain: str, window: str) -> Dict[str, Any]:
    """Fetch on-chain metrics directly from database (bypasses MCP for speed)."""
    with get_timescale_db() as ts_db:
        def latest(metric_name: str):
            stmt = (
                select(OnchainMetricModel.value)
                .where(
                    OnchainMetricModel.chain == chain,
                    OnchainMetricModel.metric == metric_name,
                    OnchainMetricModel.raw.op("->>")("window") == window,
                )
                .order_by(OnchainMetricModel.time.desc())
                .limit(1)
            )
            value = ts_db.execute(stmt).scalar_one_or_none()
            return float(value) if value is not None else None

        flows = {
            "exchange_inflow_usd": latest("exchange_inflow_usd"),
            "exchange_outflow_usd": latest("exchange_outflow_usd"),
            "net_flow_usd": latest("net_flow_usd"),
            "exchange_flow_ratio": latest("exchange_flow_ratio"),
            "flow_trend_24h": latest("flow_trend_24h"),
        }

        whales = {
            "whale_count": latest("whale_count"),
            "total_whale_volume_usd": latest("total_whale_volume_usd"),
            "avg_whale_tx_size_usd": latest("avg_whale_tx_size_usd"),
            "whale_exchange_inflow": latest("whale_exchange_inflow"),
            "whale_exchange_outflow": latest("whale_exchange_outflow"),
            "whale_exchange_ratio": latest("whale_exchange_ratio"),
            "unique_whale_addresses": latest("unique_whale_addresses"),
        }

        aggregated_metrics = {
            "market_pressure_index": latest("market_pressure_index"),
            "whale_to_exchange_ratio": latest("whale_to_exchange_ratio"),
            "price_whale_corr_7d": latest("price_whale_corr_7d"),
            "flow_trend_7d": latest("flow_trend_7d"),
            "price_change_pct": latest("price_change_pct"),
        }

    # Derive market_bias
    net_flow = flows.get("net_flow_usd") or 0.0
    price_change = aggregated_metrics.get("price_change_pct") or 0.0
    whale_ratio = aggregated_metrics.get("whale_to_exchange_ratio") or 0.0

    flow_bias = 1 if net_flow > 0 else -1 if net_flow < 0 else 0
    price_bias = 1 if price_change > 0 else -1 if price_change < 0 else 0
    ratio_bias = 1 - whale_ratio

    bias_score = (flow_bias * 0.4) + (price_bias * 0.3) + (ratio_bias * 0.3)
    if bias_score > 0.3:
        market_bias = "bullish"
    elif bias_score < -0.3:
        market_bias = "bearish"
    else:
        market_bias = "neutral"

    return {
        "flows": flows,
        "whales": whales,
        "aggregated": {**aggregated_metrics, "market_bias": market_bias},
        "errors": [],
    }


def _fetch_forecast_from_cache(symbol: str) -> Dict[str, Any]:
    """Fetch forecast from cache table (fast, no MCP overhead)."""
    from data.storage.models import ForecastCache as ForecastCacheModel

    with get_timescale_db() as ts_db:
        stmt = (
            select(ForecastCacheModel)
            .where(ForecastCacheModel.symbol == symbol)
            .limit(1)
        )
        cached = ts_db.execute(stmt).scalar_one_or_none()
        if cached:
            return {
                "model_used": cached.model_used,
                "forecast_points": cached.forecast_points,
                "last_point": cached.last_point,
                "raw_text": cached.raw_text,
                "generated_at": cached.generated_at.isoformat() if cached.generated_at else None,
            }
    return {}


def _fetch_sentiment_from_cache(symbol: str) -> Dict[str, Any]:
    """Fetch sentiment from cache table (fast, no MCP overhead)."""
    from data.storage.models import SentimentCache as SentimentCacheModel

    with get_timescale_db() as ts_db:
        stmt = (
            select(SentimentCacheModel)
            .where(SentimentCacheModel.symbol == symbol)
            .limit(1)
        )
        cached = ts_db.execute(stmt).scalar_one_or_none()
        if cached:
            return {
                "aggregated": {
                    "top_sentiment": cached.top_sentiment,
                    "top_confidence": cached.top_confidence,
                    "bullish_score": cached.bullish_score,
                    "bearish_score": cached.bearish_score,
                    "neutral_score": cached.neutral_score,
                },
                "sources": cached.sources,
                "response": cached.response,
                "generated_at": cached.generated_at.isoformat() if cached.generated_at else None,
            }
    return {}


async def _gather_overview(symbol: str, horizon_hours: int, window: str, k_docs: int):
    """Gather forecast, sentiment, and on-chain data from cache (fast dashboard loading)."""
    import asyncio

    def fetch_all_from_db():
        try:
            forecast = _fetch_forecast_from_cache(symbol)
            sentiment = _fetch_sentiment_from_cache(symbol)
            onchain = _fetch_onchain_metrics_from_db("ethereum", window)
            return {"forecast": forecast, "sentiment": sentiment, "onchain": onchain}
        except Exception as e:
            logger.error("Dashboard DB fetch failed: %s", e)
            return {"error": str(e)}

    result = await asyncio.to_thread(fetch_all_from_db)
    return result


@router.get("/overview/{symbol}")
async def get_dashboard_overview(
    symbol: str,
    horizon_days: int = Query(3, ge=1, le=30),
    window: str = Query("24h", description="Lookback window for on-chain metrics."),
    k_docs: int = Query(5, ge=1, le=20),
):
    sanitized_symbol = _validate_symbol(symbol)
    if window not in ALLOWED_WINDOWS:
        raise HTTPException(status_code=400, detail=f"window must be one of {sorted(ALLOWED_WINDOWS)}")

    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    horizon_hours = horizon_days * 24
    logger.info(
        "[%s] Dashboard overview: symbol=%s horizon_days=%s window=%s",
        request_id,
        sanitized_symbol,
        horizon_days,
        window,
    )

    shaped = await _gather_overview(sanitized_symbol, horizon_hours, window, k_docs)
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    response = {
        "request_id": request_id,
        "symbol": sanitized_symbol,
        "duration_ms": duration_ms,
        "forecast": _shape_forecast(shaped.get("forecast", {})),
        "sentiment": _shape_sentiment(shaped.get("sentiment", {})),
        "onchain": _shape_onchain_metrics(shaped.get("onchain", {})),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Optional plain-text summary for frontend fallbacks (InsightSummary).
    try:
        sent = response["sentiment"]
        onchain = response["onchain"]
        sentiment_label = (sent.get("top_sentiment") or "mixed").lower()
        whale_tx = onchain.get("whale_transactions")
        flow = onchain.get("dominant_flow")
        pressure = onchain.get("market_pressure_index")

        parts = [f"Market mood for {sanitized_symbol} is {sentiment_label}."]
        if whale_tx:
            parts.append(f"{whale_tx} whale transactions observed in the recent window.")
        if flow:
            parts.append(f"Dominant flow is {flow}.")
        if pressure is not None:
            parts.append(f"Market pressure index is {pressure}.")

        response["response"] = " ".join(parts)
    except Exception as e:
        logger.warning("Failed to build overview summary: %s", e)

    return response


@router.get("/backtest/{symbol}")
async def get_backtest_summary(
    symbol: str,
    days: int = Query(30, ge=7, le=180, description="Backtest window in days."),
):
    sanitized_symbol = _validate_symbol(symbol)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info("[%s] Dashboard backtest: symbol=%s days=%s", request_id, sanitized_symbol, days)

    try:
        backtest_result = await BACKTESTER.run_hybrid_backtest(
            sanitized_symbol,
            days,
            ["combined"],
        )
    except Exception as exc:
        logger.error("[%s] Backtest failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Backtest service unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    response = {
        "request_id": request_id,
        "symbol": sanitized_symbol,
        "duration_ms": duration_ms,
        "metrics": backtest_result.get("metrics") if isinstance(backtest_result, dict) else backtest_result,
        "equity_curve": backtest_result.get("equity_curve") if isinstance(backtest_result, dict) else None,
        "trades": backtest_result.get("trades") if isinstance(backtest_result, dict) else None,
    }
    return response


@router.get("/whales/{symbol}")
async def get_recent_whale_activity(
    symbol: str,
    limit: int = Query(20, ge=5, le=50, description="Number of recent whale alerts to return."),
):
    sanitized_symbol = _validate_symbol(symbol)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info("[%s] Whale feed: symbol=%s limit=%s", request_id, sanitized_symbol, limit)

    try:
        with get_timescale_db() as session:
            stmt = (
                select(WhaleAlertModel)
                .where(WhaleAlertModel.asset == sanitized_symbol)
                .order_by(desc(WhaleAlertModel.time))
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()
    except Exception as exc:
        logger.error("[%s] Whale fetch failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Failed to load whale alerts.") from exc

    alerts: List[Dict[str, Any]] = []
    for row in results:
        alerts.append(
            {
                "time": row.time.isoformat() if row.time else None,
                "tx_hash": row.tx_hash,
                "chain": row.chain,
                "from_address": row.from_address,
                "to_address": row.to_address,
                "asset": row.asset,
                "amount": float(row.amount) if row.amount is not None else None,
                "usd_value": float(row.usd_value) if row.usd_value is not None else None,
            }
        )

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    return {
        "request_id": request_id,
        "symbol": sanitized_symbol,
        "duration_ms": duration_ms,
        "alerts": alerts,
    }
