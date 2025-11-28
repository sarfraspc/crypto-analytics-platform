"""FastAPI router for price forecast and symbol listing endpoints."""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import Integer, cast, func, select

from core.database import get_metadata_db, get_timescale_db
from core.logging_config import setup_logging
from data.storage.models import OHLCV as OHLCVModel
from data.storage.models import Token as TokenModel
from modules.agent.agent_client import call_mcp_tool
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_utils import _scaler_path_for, load_scaler_with_meta

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Price"])

MAX_HORIZON_DAYS = 30
_PREPROCESSOR: Optional[CoinPreprocessor] = None
PRIORITY_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "AVAX",
    "DOGE",
    "MATIC",
    "LTC",
    "DOT",
]


def _validate_symbol(symbol: str) -> str:
    """Validate and normalize crypto symbol input."""
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _normalize_forecast_points(payload: Dict[str, Any], horizon_hours: int) -> List[Dict[str, Any]]:
    """Extract and normalize forecast points from MCP payload."""
    predicted = payload.get("predicted_close")
    timestamps = payload.get("timestamps") or payload.get("forecast_timestamps")
    points: List[Dict[str, Any]] = []

    if isinstance(predicted, list):
        for idx, value in enumerate(predicted):
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            timestamp = None
            if isinstance(timestamps, list) and idx < len(timestamps):
                timestamp = timestamps[idx]
            points.append({"timestamp": timestamp, "predicted_close": price})
        if points:
            return points[:horizon_hours]

    raw_text = payload.get("raw_text") or ""
    if raw_text:
        row_regex = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+(-?\d+\.?\d*)$')
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or "timestamp predicted_close" in line:
                continue
            match = row_regex.match(line)
            if match:
                ts, price_str = match.groups()
                try:
                    price = float(price_str)
                except ValueError:
                    continue
                points.append({"timestamp": ts, "predicted_close": price})
                if len(points) >= horizon_hours:
                    break
    return points


def _get_preprocessor() -> CoinPreprocessor:
    """Get or create singleton CoinPreprocessor instance."""
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        _PREPROCESSOR = CoinPreprocessor()
    return _PREPROCESSOR


def _denormalize_forecast_points(symbol: str, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Denormalize scaled forecast values to real USD prices."""
    if not points:
        return points

    # NEW SAFETY CHECK
    # If the first predicted price is greater than 1.0, assume it is
    # ALREADY a real price (not a 0-1 scaled value).
    # This allows the API to handle both old models (0-1) and new models (Real Prices)
    first_val = points[0].get("predicted_close", 0)
    try:
        if float(first_val) > 1.0:
            logger.info(f"[{symbol}] Forecast appears to be real prices ({first_val}), skipping denormalization.")
            return points
    except (ValueError, TypeError):
        pass
    # ------------------------

    preprocessor = _get_preprocessor()
    scaler_path = _scaler_path_for(preprocessor.scaler_dir, symbol.upper(), None)
    scaler, cols = load_scaler_with_meta(scaler_path)

    close_idx = None
    if scaler and cols and "close" in cols:
        close_idx = cols.index("close")

    valid_points: List[Dict[str, Any]] = []
    values: List[float] = []
    for point in points:
        value = point.get("predicted_close")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        valid_points.append(point)
        values.append(numeric)

    if not values:
        return points

    # Preferred path: use stored scaler metadata when available.
    if scaler is not None and close_idx is not None and cols:
        template = np.zeros((len(values), len(cols)))
        template[:, close_idx] = np.array(values)
        try:
            denorm = scaler.inverse_transform(template)[:, close_idx]
        except Exception as exc:
            logger.warning("Denormalizing forecast failed for %s: %s", symbol, exc)
            return points

        for point, denorm_value in zip(valid_points, denorm):
            point["predicted_close"] = float(denorm_value)
        return points

    # Fallback path: if no scaler metadata is available but the values look
    # like 0–1 scaled outputs for a large‑priced asset, approximate an
    # inverse MinMax scaling using recent raw OHLCV history.
    try:
        max_pred = max(values)
        if max_pred < 1.0:
            df_raw = preprocessor.load_data(symbol.upper(), interval="1h", lookback_days=365)
            if not df_raw.empty and "close" in df_raw.columns:
                min_close = float(df_raw["close"].min())
                max_close = float(df_raw["close"].max())
                if max_close > min_close and max_close > 10.0:
                    scale = max_close - min_close
                    for point, v in zip(valid_points, values):
                        real_val = float(v) * scale + min_close
                        point["predicted_close"] = real_val
                    logger.info(
                        "[%s] Applied heuristic denormalization fallback "
                        "(no scaler meta, inferred MinMax range %.2f–%.2f)",
                        symbol,
                        min_close,
                        max_close,
                    )
                    return points
    except Exception as exc:
        logger.warning("Heuristic denormalization fallback failed for %s: %s", symbol, exc)

    return points


@router.get("/symbols")
async def list_available_symbols(
    exchange: Optional[str] = Query(None, description="Optional exchange filter (e.g., binance)."),
    interval: Optional[str] = Query(None, description="Optional timeframe filter (e.g., 1h)."),
    limit: int = Query(200, ge=1, le=1000, description="Maximum number of symbols to return."),
):
    request_id = str(uuid.uuid4())
    logger.info("[%s] Listing symbols exchange=%s interval=%s limit=%s", request_id, exchange, interval, limit)

    try:
        with get_timescale_db() as session:
            stmt = select(OHLCVModel.symbol).distinct()
            if exchange:
                stmt = stmt.where(OHLCVModel.exchange == exchange)
            if interval:
                stmt = stmt.where(OHLCVModel.interval == interval)
            stmt = stmt.order_by(OHLCVModel.symbol).limit(limit)
            rows = session.execute(stmt).scalars().all()
    except Exception as exc:
        logger.error("[%s] Failed to list symbols: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Unable to load available symbols.") from exc

    priority_map = {sym.upper(): idx for idx, sym in enumerate(PRIORITY_SYMBOLS)}
    ordered_symbols = rows
    rank_map: Dict[str, int] = {}
    if rows:
        upper_rows = [sym.upper() for sym in rows if isinstance(sym, str)]
        try:
            with get_metadata_db() as meta_session:
                if upper_rows:
                    rank_stmt = (
                        select(
                            TokenModel.symbol,
                            cast(
                                func.nullif(
                                    func.jsonb_extract_path_text(TokenModel.token_metadata, "market_cap_rank"),
                                    "",
                                ),
                                Integer,
                            ).label("rank"),
                        ).where(func.upper(TokenModel.symbol).in_(upper_rows))
                    )
                    rank_rows = meta_session.execute(rank_stmt).all()
                    rank_map = {
                        (symbol or "").upper(): rank for symbol, rank in rank_rows if symbol and rank is not None
                    }
        except Exception as exc:
            logger.warning("[%s] Unable to apply market-cap ordering: %s", request_id, exc)

        def sort_key(sym: str):
            sym_upper = sym.upper()
            if sym_upper in priority_map:
                return (0, priority_map[sym_upper])
            if sym_upper in rank_map:
                return (1, rank_map[sym_upper])
            return (2, sym)

        ordered_symbols = sorted(rows, key=sort_key)

    return {
        "request_id": request_id,
        "exchange": exchange,
        "interval": interval,
        "count": len(ordered_symbols),
        "symbols": ordered_symbols,
    }


@router.get("/forecast/{symbol}")
async def get_price_forecast(
    symbol: str,
    horizon_days: int = Query(3, ge=1, le=MAX_HORIZON_DAYS, description="Forecast horizon in days."),
    start_date: Optional[str] = Query(
        None,
        description="Optional ISO date indicating when the forecast window should start.",
    ),
):
    sanitized_symbol = _validate_symbol(symbol)
    horizon_hours = horizon_days * 24
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    logger.info(
        "[%s] Price forecast request: symbol=%s horizon_days=%s start_date=%s",
        request_id,
        sanitized_symbol,
        horizon_days,
        start_date,
    )

    try:
        forecast_payload = await call_mcp_tool(
            "crypto-prophet-server", 
            "forecast_prophet",      
            {"symbol": sanitized_symbol, "horizon": horizon_hours, "start_date": start_date} if start_date else {"symbol": sanitized_symbol, "horizon": horizon_hours},
        )
        
        # CRITICAL RESTORATION: JSON PARSING 
        # The MCP server sends a String. We must convert it to a Dict.
        if isinstance(forecast_payload, str):
            try:
                forecast_payload = json.loads(forecast_payload)
            except json.JSONDecodeError:
                logger.error("[%s] Failed to parse MCP response as JSON: %s", request_id, forecast_payload[:100])
        # ---------------------------------------------

    except Exception as exc:
        logger.error("[%s] Forecast MCP call failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Forecast service unavailable.") from exc

    points = _normalize_forecast_points(forecast_payload or {}, horizon_hours)
    
    # The safety check inside _denormalize_forecast_points handles the real prices from Prophet
    points = _denormalize_forecast_points(sanitized_symbol, points)
    
    raw_text = forecast_payload.get("raw_text") if isinstance(forecast_payload, dict) else None
    
    # Updated default model name to match reality
    model_used = forecast_payload.get("model_used", "prophet_v1") if isinstance(forecast_payload, dict) else "prophet_v1"

    if not points and not raw_text:
        logger.warning("[%s] Forecast payload missing usable data", request_id)
        raise HTTPException(status_code=502, detail="Forecast data unavailable.")

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    response = {
        "request_id": request_id,
        "symbol": sanitized_symbol,
        "horizon_hours": horizon_hours,
        "model_used": model_used,
        "start_date": start_date,
        "forecast_points": points,
        "last_point": points[-1] if points else None,
        "raw_text": raw_text,
        "generated_at": datetime.utcnow().isoformat(),
        "duration_ms": duration_ms,
    }
    logger.info("[%s] Forecast returned %s points in %sms", request_id, len(points), duration_ms)
    return response
