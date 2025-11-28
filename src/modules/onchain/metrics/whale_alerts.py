"""Whale alert summarization and metrics computation."""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Set

from sqlalchemy import select

from core.config import settings
from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.storage.crud import upsert_onchain_metrics
from data.storage.models import WhaleAlert as WhaleAlertModel
from data.validation import OnchainMetric
from utils.cache import RedisCache

setup_logging()
logger = logging.getLogger(__name__)

redis_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    expire_seconds=3600
)

EXCHANGE_ADDRS = {addr.lower(): exchange for exchange, addrs in settings.EXCHANGE_ADDRESSES.items() for addr in addrs}


def summarize_whale_alerts(
    chain: str = 'ethereum',
    time_window: str = '24h'
):
    """Summarize whale transfer activity with exchange flow breakdown."""
    cache_key = f"onchain:whale_alerts:{chain}:{time_window}"
    cached = redis_cache.get_json(cache_key)
    if cached:
        logger.info(f"Returning cached whale summary for {cache_key}")
        return cached

    window_delta = timedelta(hours=1) if time_window == '1h' else timedelta(days=1)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - window_delta - timedelta(minutes=10)
        
    with get_timescale_db() as db:
        try:
            query = select(WhaleAlertModel).where(
                WhaleAlertModel.chain == chain,
                WhaleAlertModel.time >= start_time.replace(tzinfo=timezone.utc),
                WhaleAlertModel.time <= end_time.replace(tzinfo=timezone.utc)
            )
            alerts = db.execute(query).scalars().all()
            logger.info(f"UTC query: {start_time.isoformat()} to {end_time.isoformat()}, found {len(alerts)}")

            if not alerts:
                logger.info(f"No whale alerts in {time_window} window; using defaults")
                whale_count = 0
                total_volume = Decimal(0)
                avg_size = Decimal(0)
                unique_whale_addresses = 0
                inflows = 0
                outflows = 0
                unique_addresses: Set[str] = set() 
            else:
                whale_count = len(alerts)
                total_volume = sum((alert.usd_value or Decimal(0)) for alert in alerts)
                avg_size = total_volume / whale_count if whale_count > 0 else Decimal(0)
                unique_addresses: Set[str] = set()
                inflows = 0
                outflows = 0

                for alert in alerts:
                    unique_addresses.add(alert.from_address or '')
                    unique_addresses.add(alert.to_address or '')
                    from_lower = alert.from_address.lower() if alert.from_address else None
                    to_lower = alert.to_address.lower() if alert.to_address else None

                    if to_lower in EXCHANGE_ADDRS:
                        inflows += 1
                    if from_lower in EXCHANGE_ADDRS:
                        outflows += 1

                unique_whale_addresses = len(unique_addresses)

            total_exchange = inflows + outflows
            whale_exchange_ratio = Decimal(str((inflows / total_exchange * 100) if total_exchange > 0 else 0))

            result = {
                "time": end_time,
                "chain": chain,
                "window": time_window,
                "whale_count": whale_count,
                "total_whale_volume_usd": float(total_volume),
                "avg_whale_tx_size_usd": float(avg_size),
                "whale_exchange_inflow": inflows,
                "whale_exchange_outflow": outflows,
                "whale_exchange_ratio": float(whale_exchange_ratio),
                "unique_whale_addresses": unique_whale_addresses
            }

            raw_base = {"window": time_window}
            metrics = [
                OnchainMetric(time=end_time, chain=chain, metric="whale_count", value=Decimal(whale_count), raw={**raw_base, "description": "number of whale transactions"}),
                OnchainMetric(time=end_time, chain=chain, metric="total_whale_volume_usd", value=total_volume, raw={**raw_base, "description": "sum of all whale transfer volumes"}),
                OnchainMetric(time=end_time, chain=chain, metric="avg_whale_tx_size_usd", value=avg_size, raw={**raw_base, "description": "average size per whale tx"}),
                OnchainMetric(time=end_time, chain=chain, metric="whale_exchange_inflow", value=Decimal(inflows), raw={**raw_base, "description": "# of whale transfers → exchange addresses"}),
                OnchainMetric(time=end_time, chain=chain, metric="whale_exchange_outflow", value=Decimal(outflows), raw={**raw_base, "description": "# of whale transfers ← exchange addresses"}),
                OnchainMetric(time=end_time, chain=chain, metric="whale_exchange_ratio", value=whale_exchange_ratio, raw={**raw_base, "description": "inflow / (outflow + inflow)"}),
                OnchainMetric(time=end_time, chain=chain, metric="unique_whale_addresses", value=Decimal(unique_whale_addresses), raw={**raw_base, "description": "distinct whales in period"})
            ]
            upsert_onchain_metrics(db, metrics)

            result['time'] = result['time'].isoformat()
            redis_cache.set_json(cache_key, result)  
            logger.info(f"Summarized whales: count={whale_count}, volume={total_volume}")
            return result

        except Exception as e:
            logger.error(f"Error summarizing whale alerts: {e}")
            return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize whale alerts")
    parser.add_argument("--window", default="24h", choices=["1h", "24h"])
    args = parser.parse_args()

    summary = summarize_whale_alerts(time_window=args.window)
    if summary:
        print(summary)
    else:
        print("No summary computed")