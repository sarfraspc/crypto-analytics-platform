import asyncio
import logging
from datetime import datetime
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, select, Integer

import mlflow

from core.config import settings
from core.database import get_timescale_db, get_metadata_db
from data.storage.crud import update_ingestion_job, get_last_success
from data.validation import IngestionJob
from data.storage.models import Token
from data.ingestion.market_client import backfill_ohlcv_ccxt, poll_trades_ccxt
from data.ingestion.news_client import ingest_cryptopanic, ingest_reddit_praw, ingest_fng

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

def backfill_and_ta(db_timescale, db_metadata, exchange, symbol, interval, since_ms):
    inserted_rows = backfill_ohlcv_ccxt(db_timescale, db_metadata, exchange, symbol, interval, since_ms)
    return inserted_rows


async def run_backfill(db_metadata: Session, db_timescale: Session, symbols: List[Dict] = None):
    start_time = datetime.now()
    symbols = symbols or get_symbols_from_tokens(db_metadata, limit=50)
    logger.info("Starting backfill for %d symbols", len(symbols))

    old_since_ms = 0
    
    loop = asyncio.get_running_loop()
    tasks = []
    for i, s in enumerate(symbols):
        logger.info("Backfilling %s/%s: %s", i + 1, len(symbols), s['label'])
        tasks.append(loop.run_in_executor(None, backfill_ohlcv_ccxt, db_timescale, db_metadata, s['exchange'], s['use_ccxt_symbol'], '1d', old_since_ms))
        tasks.append(loop.run_in_executor(None, backfill_ohlcv_ccxt, db_timescale, db_metadata, s['exchange'], s['use_ccxt_symbol'], '1h', old_since_ms))

    market_tasks = tasks[:len(symbols)*2]
    alt_tasks = tasks[len(symbols)*2:]

    try:
        market_start = datetime.now()
        market_results = await asyncio.gather(*market_tasks, return_exceptions=True)
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

        alt_start = datetime.now()
        alt_data_results = await asyncio.gather(*alt_tasks, return_exceptions=True)
        alt_duration = (datetime.now() - alt_start).total_seconds()

        cp_count, cp_skipped = (0, 0)
        reddit_count, reddit_skipped = (0, 0)
        fng_count, fng_skipped = (0, 0)

        if len(alt_data_results) >= 1:
            res0 = alt_data_results[0]
            if not isinstance(res0, Exception):
                if isinstance(res0, tuple) and len(res0) >= 2:
                    cp_count, cp_skipped = res0
                else:
                    cp_count = res0
        if len(alt_data_results) >= 2:
            res1 = alt_data_results[1]
            if not isinstance(res1, Exception):
                if isinstance(res1, tuple) and len(res1) >= 2:
                    reddit_count, reddit_skipped = res1
                else:
                    reddit_count = res1
        if len(alt_data_results) >= 3:
            res2 = alt_data_results[2]
            if not isinstance(res2, Exception):
                if isinstance(res2, tuple) and len(res2) >= 2:
                    fng_count, fng_skipped = res2
                else:
                    fng_count = res2

        news_details = {
            'cryptopanic_inserted': cp_count,
            'cryptopanic_skipped': cp_skipped,
            'reddit_inserted': reddit_count,
            'reddit_skipped': reddit_skipped,
            'fng_inserted': fng_count,
            'fng_skipped': fng_skipped,
            'total_inserted': cp_count + reddit_count + fng_count,
            'total_skipped': cp_skipped + reddit_skipped + fng_skipped
        }
        log_ingestion_to_mlflow("news_client", "backfill", alt_duration, [], news_details)
        logger.info(f"News Backfill summary: CryptoPanic Inserted: {cp_count}, CryptoPanic Skipped: {cp_skipped}, Reddit Inserted: {reddit_count}, Reddit Skipped: {reddit_skipped}, FNG Inserted: {fng_count}, FNG Skipped: {fng_skipped}. Total Inserted: {news_details['total_inserted']}, Total Skipped: {news_details['total_skipped']}")

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

async def run_ingestion_cycle(db_metadata: Session, db_timescale: Session, pipeline: str = 'full_cycle', symbols: List[Dict] = None):
    start_time = datetime.now()
    symbols = symbols or get_symbols_from_tokens(db_metadata, limit=50)
    logger.info(f"Starting {pipeline} with %d symbols", len(symbols))

    since_ts = int(get_last_success(db_timescale, pipeline).timestamp() * 1000)

    loop = asyncio.get_running_loop()
    
    market_tasks = []
    for s in symbols:
        task = loop.run_in_executor(
            None,
            backfill_and_ta,
            db_timescale,
            db_metadata,
            s['exchange'],
            s['use_ccxt_symbol'],
            '1h',
            since_ts
        )
        market_tasks.append(task)

    logger.info("Starting alternative data ingestion: CryptoPanic, Reddit, FNG, Whale Alerts")
    alt_data_tasks = [
        loop.run_in_executor(None, ingest_cryptopanic, db_timescale),
        loop.run_in_executor(None, ingest_reddit_praw, db_timescale, "cryptocurrency", 50),
        loop.run_in_executor(None, ingest_fng, db_timescale)
    ]

    try:
        market_start = datetime.now()
        market_results = await asyncio.gather(*market_tasks, return_exceptions=True)
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
        log_ingestion_to_mlflow("market_client", pipeline, market_duration, symbols, market_details)
        logger.info(f"Market Cycle summary: OHLCV Inserted: {total_ohlcv_inserted}")

        alt_start = datetime.now()
        alt_data_results = await asyncio.gather(*alt_data_tasks, return_exceptions=True)
        alt_duration = (datetime.now() - alt_start).total_seconds()

        cp_count, cp_skipped = (0, 0)
        reddit_count, reddit_skipped = (0, 0)
        fng_count, fng_skipped = (0, 0)

        if len(alt_data_results) >= 1:
            res0 = alt_data_results[0]
            if not isinstance(res0, Exception):
                if isinstance(res0, tuple) and len(res0) >= 2:
                    cp_count, cp_skipped = res0
                else:
                    cp_count = res0
        if len(alt_data_results) >= 2:
            res1 = alt_data_results[1]
            if not isinstance(res1, Exception):
                if isinstance(res1, tuple) and len(res1) >= 2:
                    reddit_count, reddit_skipped = res1
                else:
                    reddit_count = res1
        if len(alt_data_results) >= 3:
            res2 = alt_data_results[2]
            if not isinstance(res2, Exception):
                if isinstance(res2, tuple) and len(res2) >= 2:
                    fng_count, fng_skipped = res2
                else:
                    fng_count = res2

        news_details = {
            'cryptopanic_inserted': cp_count,
            'cryptopanic_skipped': cp_skipped,
            'reddit_inserted': reddit_count,
            'reddit_skipped': reddit_skipped,
            'fng_inserted': fng_count,
            'fng_skipped': fng_skipped,
            'total_inserted': cp_count + reddit_count + fng_count,
            'total_skipped': cp_skipped + reddit_skipped + fng_skipped
        }
        log_ingestion_to_mlflow("news_client", pipeline, alt_duration, [], news_details)
        logger.info(f"News Cycle summary: CryptoPanic Inserted: {cp_count}, CryptoPanic Skipped: {cp_skipped}, Reddit Inserted: {reddit_count}, Reddit Skipped: {reddit_skipped}, FNG Inserted: {fng_count}, FNG Skipped: {fng_skipped}. Total Inserted: {news_details['total_inserted']}, Total Skipped: {news_details['total_skipped']}")

        combined_details = {
            'market': market_details,
            'news': news_details
        }
        total_inserted = total_ohlcv_inserted + cp_count + reddit_count + fng_count
        total_skipped = cp_skipped + reddit_skipped + fng_skipped
        combined_details['overall'] = {
            'total_inserted': total_inserted,
            'total_skipped': total_skipped
        }
        update_ingestion_job(db_timescale, IngestionJob(
            pipeline=pipeline,
            last_run=start_time,
            last_success=datetime.now(),
            details={
                'symbols': [s['label'] for s in symbols],
                'summary': combined_details
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