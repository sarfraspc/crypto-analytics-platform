import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from core.logging_config import setup_logging
from modules.agent.agent_client import call_mcp_tool

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Price"])

MAX_HORIZON_DAYS = 30


def _validate_symbol(symbol: str) -> str:
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


def _normalize_forecast_points(payload: Dict[str, Any], horizon_hours: int) -> List[Dict[str, Any]]:
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
            return points

    raw_text = payload.get("raw_text") or ""
    if raw_text:
        for line in raw_text.splitlines():
            numbers = re.findall(r"-?\d+\.\d+|-?\d+", line)
            if numbers:
                try:
                    price = float(numbers[-1])
                except ValueError:
                    continue
                points.append({"timestamp": None, "predicted_close": price})
            if len(points) >= horizon_hours:
                break
    return points


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
            "crypto-sarimax-server",
            "forecast_sarimax",
            {"symbol": sanitized_symbol, "horizon": horizon_hours, "start_date": start_date} if start_date else {"symbol": sanitized_symbol, "horizon": horizon_hours},
        )
    except Exception as exc:
        logger.error("[%s] Forecast MCP call failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Forecast service unavailable.") from exc

    points = _normalize_forecast_points(forecast_payload or {}, horizon_hours)
    raw_text = forecast_payload.get("raw_text") if isinstance(forecast_payload, dict) else None
    model_used = forecast_payload.get("model_used", "sarimax_v3") if isinstance(forecast_payload, dict) else "sarimax_v3"

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
