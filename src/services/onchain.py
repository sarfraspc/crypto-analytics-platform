import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import math
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.storage.models import TASignal as TASignalModel
from modules.agent.agent_client import call_mcp_tool

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Onchain"])

ALLOWED_WINDOWS = {"1h", "24h", "7d"}


def _validate_symbol(symbol: str) -> str:
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _validate_window(window: str) -> str:
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

    return {
        "whale_transactions": whales.get("whale_count"),
        "total_whale_volume_usd": whales.get("total_whale_volume_usd"),
        "avg_whale_tx_size_usd": whales.get("avg_whale_tx_size_usd"),
        "whale_exchange_inflow": whales.get("whale_exchange_inflow"),
        "whale_exchange_outflow": whales.get("whale_exchange_outflow"),
        "whale_exchange_ratio": whales.get("whale_exchange_ratio"),
        "unique_whale_addresses": whales.get("unique_whale_addresses"),
        "exchange_inflow_usd": flows.get("exchange_inflow_usd") or flows.get("inflow_usd"),
        "exchange_outflow_usd": flows.get("exchange_outflow_usd") or flows.get("outflow_usd"),
        "net_flow_usd": flows.get("net_flow_usd") or flows.get("netflow"),
        "exchange_flow_ratio": flows.get("exchange_flow_ratio"),
        "flow_trend_24h": flows.get("flow_trend_24h"),
        "market_pressure_index": aggregated.get("market_pressure_index"),
        "market_bias": aggregated.get("market_bias"),
        "price_change_pct": aggregated.get("price_change_pct"),
        "flow_trend_7d": aggregated.get("flow_trend_7d"),
        "price_whale_corr_7d": aggregated.get("price_whale_corr_7d"),
        "errors": payload.get("errors"),
    }


@router.get("/metrics")
async def get_onchain_metrics(
    window: str = Query("24h", description="Lookback window (1h, 24h, 7d)."),
    chain: str = Query("ethereum", description="Blockchain (e.g., ethereum)."),
):
    window = _validate_window(window)
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(
        "[%s] Metrics request: chain=%s window=%s",
        request_id,
        chain,
        window,
    )

    async def _fetch_metrics():
        return await call_mcp_tool(
            "crypto-onchain-server",
            "run_metrics_only",
            {"chain": chain, "window": window},
        )

    try:
        payload = await _fetch_metrics()
    except Exception as exc:
        logger.error("[%s] Metrics tool failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="On-chain metrics unavailable.") from exc

    metrics = _shape_metrics(payload)
    metrics = _sanitize_metrics(metrics)

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    response = {
        "request_id": request_id,
        "chain": chain,
        "window": window,
        "duration_ms": duration_ms,
        "metrics": metrics,
    }
    return response


@router.get("/patterns")
async def get_ta_patterns(
    exchange: str = Query("binance", description="Exchange for TA data."),
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
    exchange: Optional[str] = Query(None, description="Filter by exchange (e.g., binance)."),
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
