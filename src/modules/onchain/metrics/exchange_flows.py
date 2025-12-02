"""Exchange flow metrics computation from whale transfer data."""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select

from core.config import settings
from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.storage.crud import upsert_onchain_metrics
from data.storage.models import WhaleAlert as WhaleAlertModel
from data.validation import OnchainMetric

setup_logging()
logger = logging.getLogger(__name__)

EXCHANGE_ADDRS = {addr.lower(): exchange for exchange, addrs in settings.EXCHANGE_ADDRESSES.items() for addr in addrs}


def compute_exchange_flows(
    chain: str = 'ethereum',
    time_window: str = '24h'
):
    """Compute exchange inflow/outflow metrics from whale alerts."""
    window_delta = timedelta(hours=1) if time_window == '1h' else timedelta(days=1)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - window_delta - timedelta(minutes=10)
    
    prev_start_time = start_time - window_delta  

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
                logger.info(f"No alerts for flows in {time_window} window; using defaults")
                inflows = Decimal(0)
                outflows = Decimal(0)
                prev_inflows = Decimal(0)
                prev_outflows = Decimal(0)
            else:
                inflows = Decimal(0)
                outflows = Decimal(0)
                for alert in alerts:
                    usd_value = alert.usd_value or Decimal(0)
                    from_lower = alert.from_address.lower() if alert.from_address else None
                    to_lower = alert.to_address.lower() if alert.to_address else None

                    if from_lower in EXCHANGE_ADDRS and to_lower not in EXCHANGE_ADDRS:
                        outflows += usd_value
                    elif to_lower in EXCHANGE_ADDRS and from_lower not in EXCHANGE_ADDRS:
                        inflows += usd_value

                prev_query = select(WhaleAlertModel).where(
                    WhaleAlertModel.chain == chain,
                    WhaleAlertModel.time >= prev_start_time,
                    WhaleAlertModel.time <= start_time.replace(tzinfo=timezone.utc)
                )
                prev_alerts = db.execute(prev_query).scalars().all()
                prev_inflows = Decimal(0)
                prev_outflows = Decimal(0)
                for alert in prev_alerts:
                    usd_value = alert.usd_value or Decimal(0)
                    from_lower = alert.from_address.lower() if alert.from_address else None
                    to_lower = alert.to_address.lower() if alert.to_address else None

                    if from_lower in EXCHANGE_ADDRS and to_lower not in EXCHANGE_ADDRS:
                        prev_outflows += usd_value
                    elif to_lower in EXCHANGE_ADDRS and from_lower not in EXCHANGE_ADDRS:
                        prev_inflows += usd_value

            net_flow = outflows - inflows
            total_flow = inflows + outflows
            flow_ratio = (net_flow / total_flow * 100) if total_flow > 0 else Decimal(0)

            prev_net = prev_outflows - prev_inflows
            prev_total = prev_inflows + prev_outflows
            prev_ratio = (prev_net / prev_total * 100) if prev_total > 0 else Decimal(0)
            flow_trend_24h = Decimal(str(((flow_ratio - prev_ratio) / abs(prev_ratio) * 100) if prev_ratio != 0 else 0))

            result = {
                "time": end_time,
                "chain": chain,
                "window": time_window,
                "exchange_inflow_usd": float(inflows),
                "exchange_outflow_usd": float(outflows),
                "net_flow_usd": float(net_flow),
                "exchange_flow_ratio": float(flow_ratio),
                "flow_trend_24h": float(flow_trend_24h)
            }

            raw_base = {"window": time_window}
            metrics = [
                OnchainMetric(time=end_time, chain=chain, metric="exchange_inflow_usd", value=inflows, raw={**raw_base, "description": "Total USD sent to exchange wallets"}),
                OnchainMetric(time=end_time, chain=chain, metric="exchange_outflow_usd", value=outflows, raw={**raw_base, "description": "Total USD sent from exchange wallets"}),
                OnchainMetric(time=end_time, chain=chain, metric="net_flow_usd", value=net_flow, raw={**raw_base, "description": "outflow − inflow"}),
                OnchainMetric(time=end_time, chain=chain, metric="exchange_flow_ratio", value=flow_ratio, raw={**raw_base, "description": "(outflow − inflow) / (inflow + outflow)"}),
                OnchainMetric(time=end_time, chain=chain, metric="flow_trend_24h", value=flow_trend_24h, raw={**raw_base, "description": "percent change vs previous day"})
            ]
            upsert_onchain_metrics(db, metrics)

            result['time'] = result['time'].isoformat()
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