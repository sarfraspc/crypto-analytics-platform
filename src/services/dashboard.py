import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, select

from core.database import get_timescale_db
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
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _shape_forecast(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"raw_text": payload}
    return {
        "model_used": payload.get("model_used", "sarimax_v3"),
        "predicted_close": payload.get("predicted_close"),
        "raw_text": payload.get("raw_text"),
    }


def _shape_sentiment(payload: Dict[str, Any]) -> Dict[str, Any]:
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


async def _gather_overview(symbol: str, horizon_hours: int, window: str, k_docs: int):
    tasks = {
        "forecast": call_mcp_tool(
            "crypto-prophet-server",
            "forecast_prophet",
            {"symbol": symbol, "horizon": horizon_hours},
        ),
        "sentiment": call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_with_sources",
            {
                "query": f"Market sentiment for {symbol} from news, Reddit, and on-chain context",
                "k": k_docs,
                "include_sources": True,
            },
        ),
        "onchain": call_mcp_tool(
            "crypto-onchain-server",
            "run_metrics_only",
            {"window": window},
        ),
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    shaped = {}
    for key, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.error("Dashboard %s task failed: %s", key, result)
            shaped[key] = {"error": str(result)}
        else:
            shaped[key] = result
    return shaped


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
