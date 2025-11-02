import logging
from datetime import datetime, timezone

from core.logging_config import setup_logging
from modules.onchain.metrics.exchange_flows import compute_exchange_flows
from modules.onchain.metrics.whale_alerts import summarize_whale_alerts
from modules.onchain.metrics.aggregator import combine_metrics

setup_logging()
logger = logging.getLogger(__name__)


def run_onchain_metrics(
    chain: str = 'ethereum',
    time_window: str = '24h',
    symbol: str = 'BTC'
):
    logger.info(f"Starting onchain metrics pipeline for {chain}, {time_window}")
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain": chain,
        "window": time_window,
        "flows": None,
        "whales": None,
        "aggregated": None,
        "errors": []
    }

    try:
        flows = compute_exchange_flows(chain, time_window)
        if flows:
            status["flows"] = flows
        else:
            status["errors"].append("Exchange flows failed")

        whales = summarize_whale_alerts(chain, time_window)
        if whales:
            status["whales"] = whales
        else:
            status["errors"].append("Whale summary failed")

        agg = combine_metrics(chain, time_window, symbol)
        if agg:
            status["aggregated"] = agg
        else:
            status["errors"].append("Aggregation failed")

        logger.info(f"Pipeline complete: {len(status['errors'])} errors")
        return status

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        status["errors"].append(str(e))
        return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run onchain metrics pipeline")
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--window", default="24h", choices=["1h", "24h"])
    parser.add_argument("--symbol", default="BTC")
    args = parser.parse_args()

    status = run_onchain_metrics(args.chain, args.window, args.symbol)
    print(status)