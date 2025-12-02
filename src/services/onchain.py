"""FastAPI router for on-chain metrics and TA pattern endpoints."""

import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from core.config import settings
from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.storage.models import OHLCV as OHLCVModel
from data.storage.models import TASignal as TASignalModel
from modules.agent.agent_client import call_mcp_tool

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Onchain"])

ALLOWED_WINDOWS = {"1h", "24h", "7d"}


def _validate_symbol(symbol: str) -> str:
    """Validate and normalize crypto symbol input."""
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _validate_window(window: str) -> str:
    """Validate time window parameter."""
    if window not in ALLOWED_WINDOWS:
        raise HTTPException(status_code=400, detail=f"window must be one of {sorted(ALLOWED_WINDOWS)}")
    return window


def _sanitize_numeric(value: Any) -> Any:
    """Ensure numbers are JSON-safe (no NaN/inf)."""
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    return value


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Apply numeric sanitization across the metrics payload."""
    return {key: _sanitize_numeric(value) for key, value in metrics.items()}


def _shape_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MCP metrics payload into stable fields for the frontend."""
    if not isinstance(payload, dict):
        return {"raw": payload}

    flows = payload.get("flows") or {}
    whales = payload.get("whales") or {}
    aggregated = payload.get("aggregated") or payload.get("aggregated_metrics") or {}

    def _primary_or_fallback(container: Dict[str, Any], primary: str, fallback: str) -> Any:
        value = container.get(primary)
        if value is not None:
            return value
        return container.get(fallback)

    return {
        "whale_transactions": whales.get("whale_count"),
        "total_whale_volume_usd": whales.get("total_whale_volume_usd"),
        "avg_whale_tx_size_usd": whales.get("avg_whale_tx_size_usd"),
        "whale_exchange_inflow": whales.get("whale_exchange_inflow"),
        "whale_exchange_outflow": whales.get("whale_exchange_outflow"),
        "whale_exchange_ratio": whales.get("whale_exchange_ratio"),
        "unique_whale_addresses": whales.get("unique_whale_addresses"),
        "exchange_inflow_usd": _primary_or_fallback(flows, "exchange_inflow_usd", "inflow_usd"),
        "exchange_outflow_usd": _primary_or_fallback(flows, "exchange_outflow_usd", "outflow_usd"),
        "net_flow_usd": _primary_or_fallback(flows, "net_flow_usd", "netflow"),
        "exchange_flow_ratio": flows.get("exchange_flow_ratio"),
        "flow_trend_24h": flows.get("flow_trend_24h"),
        "market_pressure_index": aggregated.get("market_pressure_index"),
        "market_bias": aggregated.get("market_bias"),
        "price_change_pct": aggregated.get("price_change_pct"),
        "flow_trend_7d": aggregated.get("flow_trend_7d"),
        "price_whale_corr_7d": aggregated.get("price_whale_corr_7d"),
        "errors": payload.get("errors"),
    }


def _compute_price_change_pct_for_symbol(symbol: str, window: str) -> Optional[float]:
    """
    Compute simple % price change over the given window for a specific symbol
    using recent OHLCV data.
    """
    symbol = _validate_symbol(symbol)
    if window not in ALLOWED_WINDOWS:
        window = "24h"

    now = datetime.now(timezone.utc)
    if window == "1h":
        delta = timedelta(hours=1)
    elif window == "7d":
        delta = timedelta(days=7)
    else:
        delta = timedelta(days=1)

    start_time = now - delta

    with get_timescale_db() as session:
        stmt = (
            select(OHLCVModel.close)
            .where(
                OHLCVModel.symbol == symbol,
                OHLCVModel.exchange == settings.MARKET_EXCHANGE_ID,
                OHLCVModel.interval == "1h",
                OHLCVModel.time >= start_time,
                OHLCVModel.time <= now,
            )
            .order_by(OHLCVModel.time.asc())
        )
        closes = session.execute(stmt).scalars().all()

    if len(closes) < 2:
        return None

    first = closes[0] or 0
    last = closes[-1] or 0
    if first <= 0:
        return None

    try:
        return (float(last) - float(first)) / float(first) * 100.0
    except Exception:
        return None


def _fetch_metrics_from_db(chain: str, window: str) -> Dict[str, Any]:
    """Fetch on-chain metrics directly from database (bypasses MCP for speed)."""
    with get_timescale_db() as ts_db:
        from data.storage.models import OnchainMetric as OnchainMetricModel

        def latest(metric_name: str) -> Optional[float]:
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

    # Derive market_bias on the fly
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

    aggregated = {
        **aggregated_metrics,
        "market_bias": market_bias,
    }

    return {
        "flows": flows,
        "whales": whales,
        "aggregated": aggregated,
        "errors": [],
    }


@router.get("/metrics")
async def get_onchain_metrics(
    window: str = Query("24h", description="Lookback window (1h, 24h, 7d)."),
    chain: str = Query("ethereum", description="Blockchain (e.g., ethereum)."),
    symbol: Optional[str] = Query(
        None,
        description="Optional asset symbol to contextualize price_change_pct (e.g., BTC, ETH).",
    ),
):
    window = _validate_window(window)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(
        "[%s] Metrics request: chain=%s window=%s symbol=%s",
        request_id,
        chain,
        window,
        symbol,
    )

    try:
        # Direct DB query - bypasses MCP for faster response
        payload = _fetch_metrics_from_db(chain, window)
    except Exception as exc:
        logger.error("[%s] Metrics DB query failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="On-chain metrics unavailable.") from exc

    metrics = _shape_metrics(payload)

    # Optionally override price_change_pct with symbol-specific change
    if symbol:
        try:
            symbol_change = _compute_price_change_pct_for_symbol(symbol, window)
        except Exception as exc:
            logger.warning(
                "[%s] Symbol-specific price change failed for %s: %s",
                request_id,
                symbol,
                exc,
            )
            symbol_change = None

        if symbol_change is not None:
            metrics["price_change_pct"] = symbol_change

    metrics = _sanitize_metrics(metrics)

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    response = {
        "request_id": request_id,
        "chain": chain,
        "window": window,
        "symbol": _validate_symbol(symbol) if symbol else None,
        "duration_ms": duration_ms,
        "metrics": metrics,
    }
    return response


@router.get("/patterns")
async def get_ta_patterns(
    exchange: str = Query(settings.MARKET_EXCHANGE_ID, description="Exchange for TA data."),
    interval: str = Query("1d", description="Candlestick interval."),
    limit: int = Query(20, ge=5, le=100, description="Maximum symbols to scan."),
):
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(
        "[%s] TA patterns: exchange=%s interval=%s limit=%s",
        request_id,
        exchange,
        interval,
        limit,
    )

    try:
        with get_timescale_db() as session:
            stmt = (
                select(TASignalModel)
                .where(
                    TASignalModel.exchange == exchange,
                    TASignalModel.interval == interval,
                )
                .order_by(TASignalModel.symbol)
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
            formatted: List[Dict[str, Any]] = [
                {
                    "symbol": row.symbol,
                    "exchange": row.exchange,
                    "interval": row.interval,
                    "time": row.time.isoformat() if row.time else None,
                    "signal": row.signal,
                    "rsi": _sanitize_numeric(float(row.rsi)) if row.rsi is not None else None,
                    "macd_hist": _sanitize_numeric(float(row.macd_hist)) if row.macd_hist is not None else None,
                    "pattern": row.pattern,
                }
                for row in rows
            ]
    except Exception as exc:
        logger.error("[%s] Patterns query failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="On-chain patterns unavailable.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    return {
        "request_id": request_id,
        "duration_ms": duration_ms,
        "exchange": exchange,
        "interval": interval,
        "patterns": formatted,
        "raw_text": None,
    }


@router.get("/pattern-symbols")
async def list_pattern_symbols(
    exchange: Optional[str] = Query(None, description="Filter by exchange (e.g., kraken)."),
    interval: Optional[str] = Query(None, description="Filter by interval (e.g., 1d)."),
    limit: int = Query(200, ge=1, le=1000, description="Maximum number of symbols to return."),
):
    request_id = str(uuid.uuid4())
    logger.info(
        "[%s] Pattern symbols: exchange=%s interval=%s limit=%s",
        request_id,
        exchange,
        interval,
        limit,
    )

    try:
        with get_timescale_db() as session:
            stmt = select(TASignalModel.symbol).distinct()
            if exchange:
                stmt = stmt.where(TASignalModel.exchange == exchange)
            if interval:
                stmt = stmt.where(TASignalModel.interval == interval)
            stmt = stmt.order_by(TASignalModel.symbol).limit(limit)
            rows = session.execute(stmt).scalars().all()
    except Exception as exc:
        logger.error("[%s] Pattern symbol query failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Unable to load pattern symbols.") from exc

    symbols = [symbol for symbol in rows if isinstance(symbol, str) and symbol.strip()]
    return {
        "request_id": request_id,
        "exchange": exchange,
        "interval": interval,
        "count": len(symbols),
        "symbols": symbols,
    }
