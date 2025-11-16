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
    errors = []
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain": chain,
        "window": time_window,
        "flows": None,
        "whales": None,
        "aggregated": None,
    }

    try:
        flows = compute_exchange_flows(chain, time_window)
        status["flows"] = flows
        if not flows:
            errors.append("Exchange flows failed")

        whales = summarize_whale_alerts(chain, time_window)
        status["whales"] = whales
        if not whales:
            errors.append("Whale summary failed")

        agg = combine_metrics(chain, time_window, symbol)
        status["aggregated"] = agg
        if not agg:
            errors.append("Aggregation failed")

        if errors:
            logger.warning(f"Metrics had partial failures: {errors}")
        else:
            logger.info("All metrics computed")
        return {
            "flows": flows or {},
            "whales": whales or {},
            "aggregated": agg or {},
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        errors.append(str(e))
        return {"flows": None, "whales": None, "aggregated": None, "errors": errors}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run onchain metrics pipeline")
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--window", default="24h", choices=["1h", "24h"])
    parser.add_argument("--symbol", default="BTC")
    args = parser.parse_args()

    status = run_onchain_metrics(args.chain, args.window, args.symbol)
    print(status)
