import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from decimal import Decimal
import numpy as np
from sqlalchemy import select

from core.config import settings
from core.database import get_timescale_db
from data.storage.models import OnchainMetric as OnchainMetricModel, OHLCV as OHLCVModel
from data.validation import OnchainMetric
from data.storage.crud import upsert_onchain_metrics
from utils.cache import RedisCache
from core.logging_config import setup_logging
from modules.onchain.patterns.ta_patterns import generate_ta_signal

setup_logging()
logger = logging.getLogger(__name__)

redis_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    expire_seconds=3600  
)

def combine_metrics(
    chain: str = 'ethereum',
    time_window: str = '24h',
    symbol: str = 'BTC'
):
    cache_key = f"onchain:aggregated_metrics:{chain}:{time_window}:{symbol}"
    cached = redis_cache.get_json(cache_key)
    if cached:
        logger.info(f"Returning cached aggregated metrics for {cache_key}")
        return cached

    with get_timescale_db() as db:
        try:
            end_time = datetime.now(timezone.utc)
            window_delta = timedelta(hours=1) if time_window == '1h' else timedelta(days=1)
            start_time = end_time - window_delta

            flow_query = select(OnchainMetricModel.value).where(
                OnchainMetricModel.chain == chain,
                OnchainMetricModel.metric == 'net_flow_eth',
                OnchainMetricModel.time >= start_time
            ).order_by(OnchainMetricModel.time.desc()).limit(1)
            net_flow_obj = db.execute(flow_query).scalar_one_or_none()
            net_flow = float(net_flow_obj) if net_flow_obj else 0.0

            whale_inflow_query = select(OnchainMetricModel.value).where(
                OnchainMetricModel.chain == chain,
                OnchainMetricModel.metric == 'whale_exchange_inflow',
                OnchainMetricModel.time >= start_time
            ).order_by(OnchainMetricModel.time.desc()).limit(1)
            whale_inflow_obj = db.execute(whale_inflow_query).scalar_one_or_none()
            whale_inflows = float(whale_inflow_obj) if whale_inflow_obj else 0.0

            if net_flow == 0 and whale_inflows == 0:
                logger.warning("No recent flows/whales; using defaults for aggregation")

            ohlcv_query = select(OHLCVModel).where(
                OHLCVModel.symbol == symbol,
                OHLCVModel.exchange == 'binance',
                OHLCVModel.interval == '1d',
                OHLCVModel.time >= start_time
            ).order_by(OHLCVModel.time.desc()).limit(2)
            ohlcvs = db.execute(ohlcv_query).scalars().all()
            price_change = 0.0
            if len(ohlcvs) >= 2:
                prev_close = ohlcvs[1].close or 0
                curr_close = ohlcvs[0].close or 0
                price_change = ((curr_close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            ta_signal = generate_ta_signal(symbol, 'binance', '1d')
            ta_bias = 1 if ta_signal and ta_signal.get('signal') == 'bullish' else -1 if ta_signal and ta_signal.get('signal') == 'bearish' else 0

            whale_to_exchange_ratio = whale_inflows / 10.0  
            market_pressure_index = max(0, min(1, (net_flow * -1 + whale_inflows + (100 - price_change)) / 300.0))

            corr_start = end_time - timedelta(days=7)
            corr_flow_query = select(OnchainMetricModel.value).where(
                OnchainMetricModel.chain == chain,
                OnchainMetricModel.metric == 'net_flow_eth',
                OnchainMetricModel.time >= corr_start
            ).order_by(OnchainMetricModel.time).limit(7)
            flows_7d = [float(row[0]) for row in db.execute(corr_flow_query).all()]

            corr_ohlcv_query = select(OHLCVModel.close).where(
                OHLCVModel.symbol == symbol,
                OHLCVModel.time >= corr_start
            ).order_by(OHLCVModel.time).limit(7)
            prices_7d = [row[0] for row in db.execute(corr_ohlcv_query).all()]
            price_changes_7d = np.diff(prices_7d) / np.array(prices_7d[:-1]) * 100 if len(prices_7d) > 1 else [0]
            price_whale_corr_7d = float(np.corrcoef(flows_7d[:-1], price_changes_7d)[0, 1]) if len(flows_7d) > 1 and len(price_changes_7d) > 0 else 0.0

            flow_trend_7d = np.mean(np.diff(flows_7d) / np.array(flows_7d[:-1]) * 100) if len(flows_7d) > 1 else 0.0

            bias_score = (ta_bias * 0.4) + (1 if net_flow > 0 else -1 if net_flow < 0 else 0) * 0.3 + (1 - whale_to_exchange_ratio) * 0.3
            market_bias = "bullish" if bias_score > 0.3 else "bearish" if bias_score < -0.3 else "neutral"

            result = {
                "time": end_time,
                "chain": chain,
                "window": time_window,
                "symbol": symbol,
                "market_pressure_index": market_pressure_index,
                "whale_to_exchange_ratio": whale_to_exchange_ratio,
                "price_whale_corr_7d": price_whale_corr_7d,
                "flow_trend_7d": flow_trend_7d,
                "market_bias": market_bias,
                "price_change_pct": price_change,
                "ta_bias": ta_bias
            }

            raw_base = {"window": time_window}
            metrics = [
                OnchainMetric(time=end_time, chain=chain, metric="market_pressure_index", value=Decimal(str(market_pressure_index)), raw={**raw_base, "description": "weighted sum of whale inflow + exchange inflow – price change"}),
                OnchainMetric(time=end_time, chain=chain, metric="whale_to_exchange_ratio", value=Decimal(str(whale_to_exchange_ratio)), raw={**raw_base, "description": "whales → exchanges / total whales"}),
                OnchainMetric(time=end_time, chain=chain, metric="price_whale_corr_7d", value=Decimal(str(price_whale_corr_7d)), raw={**raw_base, "description": "correlation between whale volume and price"}),
                OnchainMetric(time=end_time, chain=chain, metric="flow_trend_7d", value=Decimal(str(flow_trend_7d)), raw={**raw_base, "description": "mean % change in flows over 7d"})
            ]
            upsert_onchain_metrics(db, metrics)

            redis_cache.set_json(cache_key, result)
            logger.info(f"Aggregated: pressure={market_pressure_index}, bias={market_bias}")
            return result

        except Exception as e:
            logger.error(f"Error in aggregator: {e}")
            return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate onchain metrics")
    parser.add_argument("--window", default="24h", choices=["1h", "24h"])
    parser.add_argument("--symbol", default="BTC")
    args = parser.parse_args()

    agg = combine_metrics(time_window=args.window, symbol=args.symbol)
    if agg:
        print(agg)
    else:
        print("No aggregation computed")