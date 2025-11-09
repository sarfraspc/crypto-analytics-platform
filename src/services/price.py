from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
from datetime import datetime
import re

from modules.agent.agent_client import call_mcp_tool
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/price", tags=["Price"])

@router.get("/forecast/{symbol}")
async def get_price_forecast(
    symbol: str,
    horizon: Optional[int] = Query(7, description="Forecast horizon in days")  # Adjusted to days for consistency
):
    start_time = datetime.now()
    request_id = f"price_forecast_{hash(str(start_time)) % 1000000}"
    logger.info(f"[{request_id}] Request: symbol={symbol}, horizon={horizon}")

    try:
        # Call MCP tool (matches sarimax_mcp tool name)
        raw_result = await call_mcp_tool("crypto-sarimax-server", "forecast_sarimax", {"symbol": symbol, "horizon": horizon})

        # Improved parsing: extract table lines more robustly
        result_text = raw_result.get("raw_text", raw_result) if isinstance(raw_result, dict) and "raw_text" in raw_result else str(raw_result)
        if "SARIMAX Forecast" in result_text:
            # Find the forecast table section
            table_match = re.search(r'Next \d+ (hours|days):\s*(.+?)(?=\n\n|$)', result_text, re.DOTALL)
            if table_match:
                lines = table_match.group(2).strip().split("\n")
                predicted_closes = []
                for line in lines:
                    if line.strip():
                        parts = re.findall(r'\d+\.?\d*', line)  # Extract floats
                        if parts:
                            predicted_closes.append(float(parts[-1]))  # Last number is predicted_close

                # Placeholder confidence; in future, parse from model or compute
                confidence = 0.87  # TODO: Derive from forecast variance if available in response

                # Build standardized output
                result = {
                    "symbol": symbol.upper(),
                    "model_used": "sarimax_v3",
                    "predicted_close": predicted_closes,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                result = {"error": "No forecast table found in response"}
        else:
            result = {"error": "No forecast data in response"}

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error calling Price MCP: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upstream Price server error; please retry.")