from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
from datetime import datetime
import json
import re

from modules.agent.agent_client import call_mcp_tool
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/onchain", tags=["Onchain"])

# Thresholds from config or hardcoded for now
MARKET_PRESSURE_BUY = 0.6
MARKET_PRESSURE_SELL = 0.4

@router.get("/metrics/{symbol}")
async def get_onchain_metrics(
    symbol: str,
    window: Optional[str] = Query("24h", description="Lookback window")
):
    start_time = datetime.now()
    request_id = f"onchain_metrics_{hash(str(start_time)) % 1000000}"
    logger.info(f"[{request_id}] Request: symbol={symbol}, window={window}")

    try:
        raw_result = await call_mcp_tool("crypto-onchain-server", "run_metrics_only", {"symbol": symbol, "window": window})

        # Parse the result
        result_text = raw_result.get("raw_text", raw_result) if isinstance(raw_result, dict) and "raw_text" in raw_result else str(raw_result)
        if "Metrics Result:" in result_text:
            # Extract JSON block
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                # Fallback split
                json_str = result_text.split("\n\n", 1)[1] if "\n\n" in result_text else result_text
                parsed = json.loads(json_str)

            # Standardize output with config thresholds
            pressure_index = parsed.get("market_pressure_index", 0.5)
            market_pressure = "buy" if pressure_index > MARKET_PRESSURE_BUY else "sell" if pressure_index < MARKET_PRESSURE_SELL else "neutral"
            dominant_flow = parsed.get("dominant_flow", "none")

            result = {
                "symbol": symbol.upper(),
                "total_whale_txs": parsed.get("whale_transactions", 0),
                "exchange_inflow": parsed.get("inflow_usd", 0),
                "exchange_outflow": parsed.get("outflow_usd", 0),
                "market_pressure": market_pressure,
                "dominant_activity": dominant_flow,
                "timestamp": datetime.now().isoformat()
            }
        else:
            result = {"error": "No metrics data in response"}

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms")

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"[{request_id}] Invalid JSON from Onchain MCP: {e}")
        raise HTTPException(status_code=500, detail="Invalid response format from upstream service")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error calling Onchain MCP: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upstream Onchain server error; please retry.")