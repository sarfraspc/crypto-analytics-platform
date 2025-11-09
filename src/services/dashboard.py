from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime
import json

from modules.agent.agent_client import call_mcp_tool
from core.logging_config import setup_logging
import asyncio

from modules.dashboard.serializers import format_overview, format_portfolio

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/overview")
async def get_dashboard_overview():
    start_time = datetime.now()
    request_id = f"dashboard_overview_{hash(str(start_time)) % 1000000}"
    logger.info(f"[{request_id}] Request: overview")

    try:
        # Aggregate from multiple MCPs in parallel
        tasks = {
            "agent": call_mcp_tool("crypto-agent-server", "get_agent_insight", {"symbol": "BTC", "question": "Quick overview"}),
            "market_sentiment": call_mcp_tool("crypto-sentiment-server", "analyze_sentiment_batch", {"texts": ["global crypto market sentiment", "bitcoin ethereum trends"]}),
            "whale_metrics": call_mcp_tool("crypto-onchain-server", "run_metrics_only", {"symbol": "BTC", "window": "24h"}),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Handle partial failures
        agent_result = results[0] if not isinstance(results[0], Exception) else {}
        sent_result = results[1] if not isinstance(results[1], Exception) else {}
        whale_result = results[2] if not isinstance(results[2], Exception) else {}

        result = format_overview(agent_result, sent_result, whale_result)

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error aggregating dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Dashboard aggregation error; please retry.")

@router.get("/portfolio")
async def get_portfolio():
    start_time = datetime.now()
    request_id = f"dashboard_portfolio_{hash(str(start_time)) % 1000000}"
    logger.info(f"[{request_id}] Request: portfolio")

    semaphore = asyncio.Semaphore(3)  # Concurrency limit

    async def limited_call(sym):
        async with semaphore:
            return await call_mcp_tool("crypto-agent-server", "get_agent_insight", {
                "symbol": sym,
                "question": "30-day report with backtest",
                "options": json.dumps({"horizon": 30})
            })

    try:
        symbols = ["BTC", "ETH"]
        backtests = await asyncio.gather(*[limited_call(sym) for sym in symbols], return_exceptions=True)

        result = format_portfolio(backtests)

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error aggregating portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Portfolio aggregation error; please retry.")