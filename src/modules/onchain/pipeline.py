import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List
from sqlalchemy.orm import Session
from data.storage.models import Token as TokenModel
from sqlalchemy import func, select, literal_column
import mlflow

from core.database import get_metadata_db, get_timescale_db
from core.logging_config import setup_logging
from data.ingestion import chain_client  
from modules.onchain.metrics.pipeline import run_onchain_metrics
from modules.onchain.patterns.ta_patterns import generate_ta_signal
from utils.cache import RedisCache
from core.config import settings

setup_logging()
logger = logging.getLogger(__name__)

redis_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    expire_seconds=3600
)

def setup_mlflow():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow configured with: {settings.MLFLOW_TRACKING_URI}")

def log_pipeline_to_mlflow(
    experiment_name: str,
    pipeline: str,
    duration: float,
    steps: List[str],
    status: Dict
):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{pipeline} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        mlflow.log_param("pipeline", pipeline)
        mlflow.log_param("duration", duration)
        mlflow.log_param("steps", ",".join(steps))
        mlflow.log_param("chain", status.get("chain", "ethereum"))
        mlflow.log_param("window", status.get("window", "24h"))
        mlflow.log_param("errors_count", len(status.get("errors", [])))
        mlflow.log_param("errors", "; ".join(status.get("errors", [])))

        ingestion = status.get("ingestion", {})
        if ingestion:
            mlflow.log_metric("ingestion_whale_alerts", ingestion.get("ingestion", {}).get("whale_alerts", 0))
            mlflow.log_param("ingestion_status", ingestion.get("status", "unknown"))

        metrics = status.get("metrics", {})
        if metrics:
            agg = metrics.get("metrics", {})
            mlflow.log_metric("metrics_market_pressure", agg.get("market_pressure_index", 0))
            # FIXED: Correct nesting for whales
            mlflow.log_metric("metrics_whale_count", agg.get("whales", {}).get("whale_count", 0))
            mlflow.log_param("metrics_status", metrics.get("status", "unknown"))

        patterns = status.get("patterns", {})
        if patterns:
            mlflow.log_metric("patterns_signal_count", len(patterns.get("patterns", {})))
            mlflow.log_param("patterns_status", patterns.get("status", "unknown"))

        logger.info(f"Logged pipeline run for '{pipeline}' to MLflow experiment '{experiment_name}'.")

def get_top_symbols(db: Session, limit: int = 50):
    try:
        # FIXED: Raw column name (no "TokenModel.")
        rank_expr = literal_column(
            "CAST(jsonb_extract_path_text(metadata, 'market_cap_rank') AS INTEGER)"
        )
        
        result = db.execute(
            select(TokenModel.symbol)
            .where(
                func.jsonb_extract_path_text(TokenModel.token_metadata, 'market_cap_rank').isnot(None),
                func.jsonb_extract_path_text(TokenModel.token_metadata, 'market_cap_rank') != ''
            )
            .order_by(rank_expr.asc())  # ASC for low ranks first (BTC=1)
            .limit(limit)
        ).scalars().all()
        
        symbols = [row for row in result if row]  # Filter Nones
        logger.info(f"Loaded {len(symbols)} top symbols for patterns")
        return symbols[:limit]  # Enforce limit
    except Exception as e:
        logger.error(f"Error fetching top symbols: {e}")
        # Robust fallback: Top 50 by market cap rank (Nov 2025 snapshot)
        fallback = [
            'BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'XRP', 'USDC', 'DOGE', 'TON', 'ADA',
            'TRX', 'AVAX', 'SHIB', 'WBTC', 'LINK', 'BCH', 'DOT', 'NEAR', 'LTC', 'MATIC',
            'UNI', 'ICP', 'PEPE', 'KAS', 'ETC', 'APT', 'XMR', 'STX', 'HBAR', 'IMX',
            'VET', 'FIL', 'CRO', 'ATOM', 'MKR', 'FDUSD', 'ARB', 'OP', 'INJ', 'GRT',
            'LDO', 'TIA', 'RUNE', 'THETA', 'FTM', 'SEI', 'JUP', 'ALGO', 'CORE', 'BGB'
        ]
        logger.warning(f"Using fallback {len(fallback)} symbols")
        return fallback

def run_whale_ingestion(
    db: Session,
    chain: str = 'ethereum',
    batch_size: int = 100,
    threshold_usd: float = 500000.0,
    time_window: str = '24h',  # NEW: For dynamic cache keys
    symbol: str = 'BTC'        # NEW: For dynamic cache keys
):
    try:
        logger.info(f"Starting whale ingestion for {chain} (batch_size={batch_size})")
        result = chain_client.scan_eth_transfers(db, batch_size=batch_size, threshold_usd=threshold_usd)
        if result and result.get('whale_alerts', 0) > 0:
            logger.info(f"Ingested {result['whale_alerts']} whale alerts")
            
            # FIXED: Dynamic keys + safe multi-delete (chain calls; add multi to RedisCache if needed)
            try:
                keys = [
                    f"onchain:whale_alerts:{chain}:{time_window}",
                    f"onchain:exchange_flows:{chain}:{time_window}",
                    f"onchain:aggregated_metrics:{chain}:{time_window}:{symbol}"
                ]
                deleted = sum(redis_cache.delete(key) for key in keys)  # Chain single deletes
                logger.info(f"Invalidated {deleted} caches post-ingestion")
            except Exception as cache_err:
                logger.warning(f"Cache invalidation failed: {cache_err} (metrics may stale)")
            
            return {"ingestion": result, "status": "success"}
        else:
            logger.warning("No new whale alerts ingested")
            return {"ingestion": result or {}, "status": "no_data"}
    except Exception as e:
        logger.error(f"Whale ingestion failed: {e}")
        return {"ingestion": {}, "status": "error", "error": str(e)}

def run_metrics_update(
    chain: str = 'ethereum',
    time_window: str = '24h',
    symbol: str = 'BTC'
):
    try:
        logger.info(f"Starting metrics update for {chain}, {time_window}")
        # NEW: Backup clear before compute (ensures fresh even if no ingest)
        try:
            keys = [
                f"onchain:exchange_flows:{chain}:{time_window}",
                f"onchain:whale_alerts:{chain}:{time_window}",
                f"onchain:aggregated_metrics:{chain}:{time_window}:{symbol}"
            ]
            deleted = sum(redis_cache.delete(key) for key in keys)
            logger.info(f"Pre-metrics: Invalidated {deleted} caches")
        except Exception as clear_err:
            logger.warning(f"Pre-clear failed: {clear_err}")
        
        status = run_onchain_metrics(chain, time_window, symbol)
        if not status.get('errors'):
            logger.info("Metrics updated successfully")
            return {"metrics": status, "status": "success"}
        else:
            logger.warning(f"Metrics update had errors: {status['errors']}")
            return {"metrics": status, "status": "partial"}
    except Exception as e:
        logger.error(f"Metrics update failed: {e}")
        return {"metrics": {}, "status": "error", "error": str(e)}

def run_ta_patterns(
    symbols: List[str],
    exchange: str = 'binance',
    interval: str = '1d'
):
    try:
        logger.info(f"Starting TA patterns for {len(symbols)} symbols, {interval}")
        signals = {}
        for symbol in symbols:
            signal = generate_ta_signal(symbol, exchange, interval)
            if signal:
                signals[symbol] = signal
        if signals:
            logger.info(f"Generated TA signals for {len(signals)} symbols")
            cache_key = f"ta_patterns:{exchange}:{interval}"
            redis_cache.set_json(cache_key, signals)
            return {"patterns": signals, "status": "success"}
        else:
            logger.warning("No TA signals generated")
            return {"patterns": {}, "status": "no_data"}
    except Exception as e:
        logger.error(f"TA patterns failed: {e}")
        return {"patterns": {}, "status": "error", "error": str(e)}

def run_onchain_pipeline(
    db: Session,
    chain: str = 'ethereum',
    batch_size: int = 100,
    threshold_usd: float = 500000.0,
    time_window: str = '24h',
    symbol: str = 'BTC',
    run_steps: List[str] = None 
):
    if run_steps is None:
        run_steps = ['ingestion', 'metrics', 'patterns']

    pipeline_start = datetime.now()
    logger.info(f"Starting on-chain pipeline for {chain}; steps: {run_steps}")
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain": chain,
        "window": time_window,
        "steps": run_steps,
        "ingestion": None,
        "metrics": None,
        "patterns": None,
        "errors": []
    }

    if 'ingestion' in run_steps:
        # FIXED: Pass time_window/symbol for dynamic deletes
        ingestion = run_whale_ingestion(db, chain, batch_size, threshold_usd, time_window, symbol)
        status["ingestion"] = ingestion
        if ingestion and ingestion.get("status") == "error":
            status["errors"].append("Ingestion failed")

    if 'metrics' in run_steps:
        metrics = run_metrics_update(chain, time_window, symbol)
        status["metrics"] = metrics
        if metrics and metrics.get("status") == "error":
            status["errors"].append("Metrics failed")

    if 'patterns' in run_steps:
        with get_metadata_db() as meta_db:
            symbols = get_top_symbols(meta_db, limit=50)
        patterns = run_ta_patterns(symbols, interval='1d')
        status["patterns"] = patterns
        if patterns and patterns.get("status") == "error":
            status["errors"].append("Patterns failed")

    duration = (datetime.now() - pipeline_start).total_seconds()
    log_pipeline_to_mlflow("onchain_pipeline", "full_pipeline", duration, run_steps, status)

    logger.info(f"Pipeline complete: {len(status['errors'])} errors")
    return status

if __name__ == "__main__":
    setup_mlflow()
    parser = argparse.ArgumentParser(description="Run on-chain analytics pipeline")
    parser.add_argument("--chain", default="ethereum", help="Blockchain (default: ethereum)")
    parser.add_argument("--batch-size", type=int, default=100, help="Ingestion batch size")
    parser.add_argument("--threshold-usd", type=float, default=500000.0, help="Whale USD threshold")
    parser.add_argument("--window", default="24h", choices=["1h", "24h"], help="Time window")
    parser.add_argument("--symbol", default="BTC", help="Symbol for metrics")
    parser.add_argument("--run", default="all", help="Steps to run: 'all' or comma-separated e.g., 'ingestion,metrics'")
    # NEW: For manual flush
    parser.add_argument("--clear-cache", action="store_true", help="Flush all onchain Redis caches")
    args = parser.parse_args()

    steps = args.run.split(',') if args.run != "all" else None

    # NEW: Manual clear if flagged
    if args.clear_cache:
        try:
            deleted = redis_cache.delete_by_pattern("onchain:*")
            logger.info(f"Manual cache flush: Deleted {deleted} keys")
        except Exception as e:
            logger.error(f"Cache flush failed: {e}")

    with get_timescale_db() as db:
        result = run_onchain_pipeline(
            db=db,
            chain=args.chain,
            batch_size=args.batch_size,
            threshold_usd=args.threshold_usd,
            time_window=args.window,
            symbol=args.symbol,
            run_steps=steps
        )
        print(result)