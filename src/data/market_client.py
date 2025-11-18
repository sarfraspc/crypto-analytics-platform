import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, select, Integer

import mlflow

from core.config import settings
from core.database import get_timescale_db, get_metadata_db
from data.storage.crud import update_ingestion_job, get_last_success
from data.validation import IngestionJob
from data.storage.models import Token, OHLCV as OHLCVModel
from data.ingestion.market_client import backfill_ohlcv_ccxt, poll_trades_ccxt
from modules.onchain.patterns.ta_patterns import generate_ta_signal

from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def setup_mlflow():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow configured with: {settings.MLFLOW_TRACKING_URI}")

def get_symbols_from_tokens(db: Session, limit: int = 50):
    try:
        result = db.execute(
            select(Token).where(func.jsonb_extract_path_text(Token.token_metadata, 'market_cap_rank').isnot(None))  
            .order_by(func.cast(func.jsonb_extract_path_text(Token.token_metadata, 'market_cap_rank'), Integer)) 
            .limit(limit)
        ).scalars().all()
        symbols = [
            {
                'label': row.symbol,
                'use_ccxt_symbol': f"{row.symbol}/USDT",
                'exchange': 'binance'
            }
            for row in result
        ]
        logger.info("Loaded %d top-ranked symbols from tokens", len(symbols))
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return []

def get_top_symbols(db: Session, limit: int = 50):
    """
    Select TA symbols directly from the OHLCV table so TA runs only
    on assets that actually have market data, with no token-metadata
    lookup or hardcoded fallback list.
    """
    try:
        result = (
            db.execute(
                select(OHLCVModel.symbol)
                .where(
                    OHLCVModel.exchange == "binance",
                    OHLCVModel.interval == "1h",
                )
                .group_by(OHLCVModel.symbol)
                .order_by(func.max(OHLCVModel.time).desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        symbols = [s for s in result if s]
        logger.info("Loaded %d OHLCV-backed symbols for TA patterns", len(symbols))
        return symbols
    except Exception as e:
        logger.error(f"Error fetching OHLCV symbols for TA: {e}")
        return []

def run_ta_patterns(
    symbols: List[str],
    exchange: str = "binance",
    interval: str = "1d",
):
    """
    Generate TA signals for a list of symbols.
    Used by ingestion and the MCP onchain TA tool.
    """
    try:
        logger.info(
            "Starting TA patterns for %d symbols on %s (%s)",
            len(symbols),
            exchange,
            interval,
        )
        signals = {}
        for symbol in symbols:
            signal = generate_ta_signal(symbol, exchange, interval)
            if signal:
                signals[symbol] = signal

        if signals:
            logger.info("Generated TA signals for %d symbols", len(signals))
            return {"patterns": signals, "status": "success"}

        logger.warning("No TA signals generated")
        return {"patterns": {}, "status": "no_data"}
    except Exception as e:
        logger.error(f"TA patterns failed: {e}")
        return {"patterns": {}, "status": "error", "error": str(e)}

def backfill_and_ta(db_timescale, db_metadata, exchange, symbol, interval, since_ms):
    """
    Backfill OHLCV for a symbol.
    TA generation is run in a separate phase after OHLCV
    has been updated for all symbols in the cycle.
    """
    return backfill_ohlcv_ccxt(db_timescale, db_metadata, exchange, symbol, interval, since_ms)


async def run_backfill(db_metadata: Session, db_timescale: Session, symbols: List[Dict] = None):
    start_time = datetime.now()
    symbols = symbols or get_symbols_from_tokens(db_metadata, limit=50)
    logger.info("Starting backfill for %d symbols", len(symbols))

    old_since_ms = 0
    
    loop = asyncio.get_running_loop()
    tasks = []
    for i, s in enumerate(symbols):
        logger.info("Backfilling %s/%s (1h only): %s", i + 1, len(symbols), s['label'])
        tasks.append(
            loop.run_in_executor(
                None,
                backfill_ohlcv_ccxt,
                db_timescale,
                db_metadata,
                s['exchange'],
                s['use_ccxt_symbol'],
                '1h',
                old_since_ms,
            ),
        )

    try:
        market_start = datetime.now()
        market_results = await asyncio.gather(*tasks, return_exceptions=True)
        market_duration = (datetime.now() - market_start).total_seconds()

        total_ohlcv_inserted = 0
        for res in market_results:
            if isinstance(res, Exception):
                logger.error(f"Market task failed: {res}")
            else:
                total_ohlcv_inserted += res

        market_details = {
            'ohlcv_inserted': total_ohlcv_inserted,
            'total_inserted': total_ohlcv_inserted,
            'total_skipped': 0
        }
        log_ingestion_to_mlflow("market_client", "backfill", market_duration, symbols, market_details)
        logger.info(f"Market Backfill summary: OHLCV Inserted: {total_ohlcv_inserted}")

    except Exception as e:
        logger.error(f"Backfill failed: {e}")

    total_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total backfill duration: {total_duration}s")
    logger.info("Backfill complete")

async def run_polling(db_metadata: Session, db_timescale: Session, symbols: List[Dict] = None):
    start_time = datetime.now()
    symbols = symbols or get_symbols_from_tokens(db_metadata, limit=10)
    logger.info("Starting polling for %d symbols", len(symbols))
    
    loop = asyncio.get_running_loop()
    tasks = []
    for s in symbols:
        tasks.append(
            loop.run_in_executor(
                None,
                poll_trades_ccxt,
                db_timescale,
                s['exchange'],
                s['use_ccxt_symbol'],
                1000
            )
        )
    try:
        poll_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_polled = 0
        for res in poll_results:
            if isinstance(res, Exception):
                logger.error(f"Polling task failed: {res}")
            else:
                total_polled += res if isinstance(res, (int, float)) else 0

        duration = (datetime.now() - start_time).total_seconds()
        details = {
            'trades_polled': total_polled,
            'total_inserted': total_polled,
            'total_skipped': 0
        }
        log_ingestion_to_mlflow("market_client", "poll", duration, symbols, details)
        logger.info(f"Polling summary: Trades Polled: {total_polled}")
    except Exception as e:
        logger.error(f"Polling failed: {e}")

    logger.info("Polling complete")

def log_ingestion_to_mlflow(experiment_name: str, pipeline: str, duration: float, symbols: List[Dict], details: Dict):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{pipeline} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        mlflow.log_param("pipeline", pipeline)
        mlflow.log_param("duration", duration)
        mlflow.log_param("symbols_count", len(symbols))
        mlflow.log_param("symbols", [s['label'] for s in symbols])
        
        for key, value in details.items():
            if value is not None:
                mlflow.log_metric(key, value)
        
        logger.info(f"Logged ingestion run for pipeline '{pipeline}' to MLflow experiment '{experiment_name}'.")

async def run_ingestion_cycle(db_metadata: Session, db_timescale: Session, pipeline: str = 'full_cycle', symbols: List[Dict] = None, delta_only: bool = True):
    start_time = datetime.now()
    symbols = symbols or get_symbols_from_tokens(db_metadata, limit=50)
    logger.info(f"Starting {pipeline} with %d symbols", len(symbols))

    if delta_only:
        last_success = get_last_success(db_timescale, pipeline)
        since_ts = int(last_success.timestamp() * 1000)
        logger.info(f"Using last_success from ingestion_jobs: {last_success} (since_ts={since_ts} ms)")  
    else:
        since_ts = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 1000)
    logger.info(f"Using since_ts: {datetime.fromtimestamp(since_ts / 1000)} (delta_only={delta_only})")

    loop = asyncio.get_running_loop()
    
    try:
        semaphore = asyncio.Semaphore(5) 
        async def limited_backfill(s):
            async with semaphore:
                return await loop.run_in_executor(
                    None,
                    backfill_and_ta,
                    db_timescale,
                    db_metadata,
                    s['exchange'],
                    s['use_ccxt_symbol'],
                    '1h',
                    since_ts
                )
        
        batch_size = 5
        market_results = []
        market_start = datetime.now()
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_tasks = [limited_backfill(s) for s in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            market_results.extend(batch_results)
            if i + batch_size < len(symbols):  
                await asyncio.sleep(5)  

        market_duration = (datetime.now() - market_start).total_seconds()  

        total_ohlcv_inserted = 0
        for res in market_results:
            if isinstance(res, Exception):
                logger.error(f"Market task failed: {res}")
            else:
                total_ohlcv_inserted += res

        # Phase 2: TA generation (after all OHLCV backfill completes)
        # Use only symbols that actually have OHLCV data in Timescale,
        # so TA does not waste work (or log \"no data\") for invalid pairs.
        ta_candidates = get_top_symbols(db_timescale, limit=len(symbols))
        requested_labels = {s["label"] for s in symbols}
        ta_symbols = [sym for sym in ta_candidates if sym in requested_labels]
        logger.info(
            "TA phase: %d symbols with OHLCV out of %d requested",
            len(ta_symbols),
            len(symbols),
        )
        if not ta_symbols:
            logger.warning("TA phase skipped: no symbols with OHLCV data found")
            ta_success = 0
            ta_errors = 0
            ta_no_data = 0
            ta_duration = 0.0
        else:
            logger.info("Starting TA generation for symbols after OHLCV backfill")
        ta_start = datetime.now()
        ta_tasks = [
            loop.run_in_executor(
                None,
                generate_ta_signal,
                symbol,
                "binance",
                "1h",
                False,
            )
            for symbol in ta_symbols
        ]
        ta_results = await asyncio.gather(*ta_tasks, return_exceptions=True)
        ta_success = 0
        ta_errors = 0
        for res in ta_results:
            if isinstance(res, Exception):
                ta_errors += 1
                logger.error(f"TA task failed: {res}")
            elif res:
                ta_success += 1
        ta_no_data = max(0, len(ta_symbols) - ta_success - ta_errors)
        ta_duration = (datetime.now() - ta_start).total_seconds()
        logger.info(
            f"TA generation complete for {len(ta_symbols)} symbols "
            f"(success={ta_success}, no_data={ta_no_data}, errors={ta_errors}) "
            f"in {ta_duration:.2f}s"
        )

        # Log combined OHLCV + TA metrics to the market_client experiment
        market_details = {
            'ohlcv_inserted': total_ohlcv_inserted,
            'total_inserted': total_ohlcv_inserted,
            'total_skipped': 0,
            'ta_success': ta_success,
            'ta_no_data': ta_no_data,
            'ta_errors': ta_errors,
            'ta_duration_sec': ta_duration,
        }
        log_ingestion_to_mlflow("market_client", pipeline, market_duration + ta_duration, symbols, market_details)
        logger.info(f"Market Cycle summary: OHLCV Inserted: {total_ohlcv_inserted}, TA success={ta_success}, no_data={ta_no_data}")

        # Persist market/TA ingestion job summary only (news is handled in separate module)
        update_ingestion_job(db_timescale, IngestionJob(
            pipeline=pipeline,
            last_run=start_time,
            last_success=datetime.now(),
            details={
                'symbols': [s['label'] for s in symbols],
                'summary': {
                    'market': market_details,
                    'overall': {
                        'total_inserted': total_ohlcv_inserted,
                        'total_skipped': 0
                    }
                }
            }
        ))
        logger.info(f"{pipeline} complete")
    except Exception as e:
        logger.error(f"Cycle fetch failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Data Ingestion Pipeline")
    parser.add_argument('--backfill', action='store_true', help="Run full historical backfill")
    parser.add_argument('--poll', action='store_true', help="Run continuous trades polling")
    parser.add_argument('--limit', type=int, default=50, help="Number of symbols to process (default: 50)")
    parser.add_argument('--pipeline', type=str, default='full_cycle', help="Pipeline mode: full_cycle or custom (default: full_cycle)")
    args = parser.parse_args()

    setup_mlflow()
    with get_metadata_db() as db_metadata, get_timescale_db() as db_timescale:
        if args.backfill:
            asyncio.run(run_backfill(db_metadata, db_timescale))
        elif args.poll:
            asyncio.run(run_polling(db_metadata, db_timescale))
        else:
            symbols = get_symbols_from_tokens(db_metadata, limit=args.limit)
            asyncio.run(run_ingestion_cycle(db_metadata, db_timescale, pipeline=args.pipeline, symbols=symbols))
