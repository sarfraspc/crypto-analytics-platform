import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from sqlalchemy import select

from core.config import settings
from core.database import get_timescale_db
from data.storage.models import WhaleAlert as WhaleAlertModel
from data.validation import OnchainMetric
from data.storage.crud import upsert_onchain_metrics
from utils.cache import RedisCache
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

redis_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    expire_seconds=3600  
)

EXCHANGE_ADDRS = {addr.lower(): exchange for exchange, addrs in settings.EXCHANGE_ADDRESSES.items() for addr in addrs}


def compute_exchange_flows(
    chain: str = 'ethereum',
    time_window: str = '24h' 
):
    cache_key = f"onchain:exchange_flows:{chain}:{time_window}"
    cached = redis_cache.get_json(cache_key)
    if cached:
        logger.info(f"Returning cached exchange flows for {cache_key}")
        return cached

    window_delta = timedelta(hours=1) if time_window == '1h' else timedelta(days=1)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - window_delta
    
    prev_start_time = start_time - window_delta  

    with get_timescale_db() as db:
        try:
            query = select(WhaleAlertModel).where(
                WhaleAlertModel.chain == chain,
                WhaleAlertModel.time >= start_time,
                WhaleAlertModel.time < end_time
            )
            alerts = db.execute(query).scalars().all()

            if not alerts:
                logger.warning(f"No alerts for flows in {time_window} window")
                return None

            inflows = Decimal(0)
            outflows = Decimal(0)
            for alert in alerts:
                amount = alert.amount or Decimal(0)
                from_lower = alert.from_address.lower() if alert.from_address else None
                to_lower = alert.to_address.lower() if alert.to_address else None

                if from_lower in EXCHANGE_ADDRS and to_lower not in EXCHANGE_ADDRS:
                    outflows += amount
                elif to_lower in EXCHANGE_ADDRS and from_lower not in EXCHANGE_ADDRS:
                    inflows += amount

            net_flow = outflows - inflows
            total_flow = inflows + outflows
            flow_ratio = (net_flow / total_flow * 100) if total_flow > 0 else Decimal(0)

            prev_query = select(WhaleAlertModel).where(
                WhaleAlertModel.chain == chain,
                WhaleAlertModel.time >= prev_start_time,
                WhaleAlertModel.time < start_time
            )
            prev_alerts = db.execute(prev_query).scalars().all()
            prev_inflows = Decimal(0)
            prev_outflows = Decimal(0)
            for alert in prev_alerts:
                amount = alert.amount or Decimal(0)
                from_lower = alert.from_address.lower() if alert.from_address else None
                to_lower = alert.to_address.lower() if alert.to_address else None

                if from_lower in EXCHANGE_ADDRS and to_lower not in EXCHANGE_ADDRS:
                    prev_outflows += amount
                elif to_lower in EXCHANGE_ADDRS and from_lower not in EXCHANGE_ADDRS:
                    prev_inflows += amount

            prev_net = prev_outflows - prev_inflows
            prev_total = prev_inflows + prev_outflows
            prev_ratio = (prev_net / prev_total * 100) if prev_total > 0 else Decimal(0)
            flow_trend_24h = Decimal(str(((flow_ratio - prev_ratio) / abs(prev_ratio) * 100) if prev_ratio != 0 else 0))

            result = {
                "time": end_time,
                "chain": chain,
                "window": time_window,
                "exchange_inflow_eth": float(inflows),
                "exchange_outflow_eth": float(outflows),
                "net_flow_eth": float(net_flow),
                "exchange_flow_ratio": float(flow_ratio),
                "flow_trend_24h": float(flow_trend_24h)
            }

            raw_base = {"window": time_window}
            metrics = [
                OnchainMetric(time=end_time, chain=chain, metric="exchange_inflow_eth", value=inflows, raw={**raw_base, "description": "Total ETH sent to exchange wallets"}),
                OnchainMetric(time=end_time, chain=chain, metric="exchange_outflow_eth", value=outflows, raw={**raw_base, "description": "Total ETH sent from exchange wallets"}),
                OnchainMetric(time=end_time, chain=chain, metric="net_flow_eth", value=net_flow, raw={**raw_base, "description": "outflow − inflow"}),
                OnchainMetric(time=end_time, chain=chain, metric="exchange_flow_ratio", value=flow_ratio, raw={**raw_base, "description": "(outflow − inflow) / (inflow + outflow)"}),
                OnchainMetric(time=end_time, chain=chain, metric="flow_trend_24h", value=flow_trend_24h, raw={**raw_base, "description": "percent change vs previous day"})
            ]
            upsert_onchain_metrics(db, metrics)

            redis_cache.set_json(cache_key, result)
            logger.info(f"Computed flows: inflow={inflows}, outflow={outflows}, net={net_flow}")
            return result

        except Exception as e:
            logger.error(f"Error computing exchange flows: {e}")
            return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute exchange flows")
    parser.add_argument("--window", default="24h", choices=["1h", "24h"])
    args = parser.parse_args()

    flows = compute_exchange_flows(time_window=args.window)
    if flows:
        print(flows)
    else:
        print("No flows computed")