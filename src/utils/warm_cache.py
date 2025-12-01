#!/usr/bin/env python3
"""
Cache warming script for dashboard data.

Run after ingestion to keep forecast and sentiment caches fresh
for fast dashboard loading.

Usage:
    python -m utils.warm_cache
"""

import asyncio
import logging
from datetime import datetime

from core.database import get_timescale_db
from core.logging_config import setup_logging
from modules.agent.agent_client import call_mcp_tool

setup_logging()
logger = logging.getLogger(__name__)

# Priority symbols to cache
PRIORITY_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"]


async def warm_forecast_cache(symbol: str) -> bool:
    """Warm forecast cache for a symbol."""
    try:
        from sqlalchemy.dialects.postgresql import insert
        from data.storage.models import ForecastCache as ForecastCacheModel

        logger.info(f"Warming forecast cache for {symbol}...")

        payload = await call_mcp_tool(
            "crypto-prophet-server",
            "forecast_prophet",
            {"symbol": symbol, "horizon": 240},
        )

        if not payload or (isinstance(payload, dict) and "error" in payload):
            logger.warning(f"Forecast failed for {symbol}: {payload}")
            return False

        # Build forecast points
        points = []
        if isinstance(payload.get("predicted_close"), list):
            timestamps = payload.get("timestamps", [])
            for i, price in enumerate(payload["predicted_close"]):
                ts = timestamps[i] if i < len(timestamps) else None
                points.append({"timestamp": ts, "predicted_close": price})

        with get_timescale_db() as session:
            cache_data = {
                "symbol": symbol,
                "model_used": payload.get("model_used", "prophet_v1"),
                "generated_at": datetime.utcnow(),
                "horizon_hours": 240,
                "forecast_points": points if points else None,
                "last_point": points[-1] if points else None,
                "raw_text": payload.get("raw_text"),
            }

            stmt = insert(ForecastCacheModel).values(**cache_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_=cache_data,
            )
            session.execute(stmt)
            session.commit()

        logger.info(f"Forecast cache warmed for {symbol}")
        return True

    except Exception as e:
        logger.error(f"Failed to warm forecast cache for {symbol}: {e}")
        return False


async def warm_sentiment_cache(symbol: str) -> bool:
    """Warm sentiment cache for a symbol."""
    try:
        from sqlalchemy.dialects.postgresql import insert
        from data.storage.models import SentimentCache as SentimentCacheModel

        logger.info(f"Warming sentiment cache for {symbol}...")

        payload = await call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_with_sources",
            {
                "query": f"Market sentiment for {symbol} from recent crypto news",
                "k": 5,
                "include_sources": True,
            },
        )

        if not payload or (isinstance(payload, dict) and "error" in payload):
            logger.warning(f"Sentiment failed for {symbol}: {payload}")
            return False

        aggregated = payload.get("aggregated", {}) if isinstance(payload, dict) else {}

        with get_timescale_db() as session:
            cache_data = {
                "symbol": symbol,
                "generated_at": datetime.utcnow(),
                "top_sentiment": aggregated.get("top_sentiment"),
                "top_confidence": aggregated.get("top_confidence"),
                "bullish_score": aggregated.get("bullish_score"),
                "bearish_score": aggregated.get("bearish_score"),
                "neutral_score": aggregated.get("neutral_score"),
                "sources": payload.get("sources") if isinstance(payload, dict) else None,
                "response": payload.get("response") if isinstance(payload, dict) else None,
            }
            stmt = insert(SentimentCacheModel).values(**cache_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_=cache_data,
            )
            session.execute(stmt)
            session.commit()

        logger.info(f"Sentiment cache warmed for {symbol}")
        return True

    except Exception as e:
        logger.error(f"Failed to warm sentiment cache for {symbol}: {e}")
        return False


async def main():
    """Warm caches for all priority symbols."""
    logger.info("=" * 60)
    logger.info("Starting cache warming...")
    logger.info(f"Symbols: {PRIORITY_SYMBOLS}")
    logger.info("=" * 60)

    results = {"forecast": {}, "sentiment": {}}

    for symbol in PRIORITY_SYMBOLS:
        results["forecast"][symbol] = await warm_forecast_cache(symbol)
        results["sentiment"][symbol] = await warm_sentiment_cache(symbol)
        await asyncio.sleep(1)

    logger.info("=" * 60)
    forecast_ok = sum(1 for v in results["forecast"].values() if v)
    sentiment_ok = sum(1 for v in results["sentiment"].values() if v)
    logger.info(f"Forecast: {forecast_ok}/{len(PRIORITY_SYMBOLS)} cached")
    logger.info(f"Sentiment: {sentiment_ok}/{len(PRIORITY_SYMBOLS)} cached")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
