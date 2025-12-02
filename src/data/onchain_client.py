"""On-chain analytics client for whale detection and metrics aggregation."""

import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List

import mlflow

from core.config import settings
from core.logging_config import setup_logging
from data.ingestion import chain_client
from modules.onchain.metrics.pipeline import run_onchain_metrics

setup_logging()
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Configure MLflow tracking URI from settings."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info("MLflow configured with: %s", settings.MLFLOW_TRACKING_URI)


def log_pipeline_to_mlflow(
    experiment_name: str,
    pipeline: str,
    duration: float,
    steps: List[str],
    status: Dict,
):
    """Log on-chain pipeline run metrics and status to MLflow."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=f"{pipeline} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ):
        mlflow.log_param("pipeline", pipeline)
        mlflow.log_param("duration", duration)
        mlflow.log_param("steps", ",".join(steps))
        mlflow.log_param("chain", status.get("chain", "ethereum"))
        mlflow.log_param("window", status.get("window", "24h"))
        mlflow.log_param("errors_count", len(status.get("errors", [])))
        mlflow.log_param("errors", "; ".join(status.get("errors", [])))

        ingestion = status.get("ingestion", {})
        if ingestion:
            mlflow.log_metric(
                "ingestion_whale_alerts",
                ingestion.get("ingestion", {}).get("whale_alerts", 0),
            )
            mlflow.log_param("ingestion_status", ingestion.get("status", "unknown"))

        metrics = status.get("metrics", {})
        if metrics:
            agg = metrics.get("metrics", {})
            mlflow.log_metric(
                "metrics_market_pressure", agg.get("market_pressure_index", 0)
            )
            mlflow.log_metric(
                "metrics_whale_count",
                agg.get("whales", {}).get("whale_count", 0),
            )
            mlflow.log_param("metrics_status", metrics.get("status", "unknown"))

        patterns = status.get("patterns", {})
        if patterns:
            mlflow.log_metric(
                "patterns_signal_count", len(patterns.get("patterns", {}))
            )
            mlflow.log_param("patterns_status", patterns.get("status", "unknown"))

        logger.info(
            "Logged pipeline run for '%s' to MLflow experiment '%s'.",
            pipeline,
            experiment_name,
        )


def run_whale_ingestion(
    chain: str = "ethereum",
    threshold_usd: float = 500000.0,
    time_window: str = "24h",
):
    """Ingest whale alerts for the given chain/window."""
    try:
        logger.info(
            "Starting whale ingestion for %s (window=%s, threshold_usd=%s)",
            chain,
            time_window,
            threshold_usd,
        )
        result = chain_client.scan_eth_transfers(threshold_usd=threshold_usd)
        if result and result.get("whale_alerts", 0) > 0:
            logger.info("Ingested %s whale alerts", result["whale_alerts"])
            return {"ingestion": result, "status": "success"}

        logger.warning("No new whale alerts ingested")
        return {"ingestion": result or {}, "status": "no_data"}
    except Exception as e:
        logger.error("Whale ingestion failed: %s", e)
        return {"ingestion": {}, "status": "error", "error": str(e)}


def run_metrics_update(
    chain: str = "ethereum",
    time_window: str = "24h",
):
    """Refresh onchain metrics (exchange flows + whale summaries + aggregator)."""
    try:
        logger.info("Starting metrics update for %s, %s", chain, time_window)
        status = run_onchain_metrics(chain, time_window)
        if not status.get("errors"):
            logger.info("Metrics updated successfully")
            return {"metrics": status, "status": "success"}

        logger.warning("Metrics update had errors: %s", status["errors"])
        return {"metrics": status, "status": "partial"}
    except Exception as e:
        logger.error("Metrics update failed: %s", e)
        return {"metrics": {}, "status": "error", "error": str(e)}


def run_onchain_pipeline(
    chain: str = "ethereum",
    threshold_usd: float = 500000.0,
    time_window: str = "24h",
    run_steps: List[str] = None,
):
    """
    High-level onchain pipeline: whale ingestion + onchain metrics.
    """
    if run_steps is None:
        run_steps = ["ingestion", "metrics"]

    pipeline_start = datetime.now()
    logger.info("Starting on-chain pipeline for %s; steps: %s", chain, run_steps)
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain": chain,
        "window": time_window,
        "steps": run_steps,
        "ingestion": None,
        "metrics": None,
        "errors": [],
    }

    if "ingestion" in run_steps:
        ingestion = run_whale_ingestion(chain, threshold_usd, time_window)
        status["ingestion"] = ingestion
        if ingestion and ingestion.get("status") == "error":
            status["errors"].append("Ingestion failed")

    if "metrics" in run_steps:
        metrics = run_metrics_update(chain, time_window)
        status["metrics"] = metrics
        if metrics and metrics.get("status") == "error":
            status["errors"].append("Metrics failed")

    duration = (datetime.now() - pipeline_start).total_seconds()
    log_pipeline_to_mlflow("onchain_pipeline", "full_pipeline", duration, run_steps, status)

    logger.info("Pipeline complete: %s errors", len(status["errors"]))
    return status


def main():
    """CLI entry point for on-chain analytics pipeline."""
    setup_mlflow()
    parser = argparse.ArgumentParser(description="Run on-chain analytics pipeline")
    parser.add_argument("--chain", default="ethereum", help="Blockchain (default: ethereum)")
    parser.add_argument(
        "--threshold-usd",
        type=float,
        default=500000.0,
        help="Whale USD threshold",
    )
    parser.add_argument(
        "--window",
        default="24h",
        choices=["1h", "24h"],
        help="Time window",
    )
    parser.add_argument(
        "--run",
        default="all",
        help="Steps to run: 'all' or comma-separated e.g., 'ingestion,metrics'",
    )
    args = parser.parse_args()

    steps = args.run.split(",") if args.run != "all" else None

    result = run_onchain_pipeline(
        chain=args.chain,
        threshold_usd=args.threshold_usd,
        time_window=args.window,
        run_steps=steps,
    )
    print(result)


if __name__ == "__main__":
    main()
