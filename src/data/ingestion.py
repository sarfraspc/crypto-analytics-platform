import asyncio
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from sqlalchemy import text

import mlflow  

from core.database import get_timescale_engine, get_metadata_engine
from core.config import settings
from data.storage.crud import update_ingestion_job
from data.validation import IngestionJob
from data.ingestion.market_client import backfill_ohlcv_coingecko, backfill_ohlcv_ccxt, poll_trades_ccxt
from data.ingestion.chain_client import scan_eth_transfers
from data.ingestion.news_client import ingest_cryptopanic, ingest_reddit_praw, ingest_fng

logger = logging.getLogger(__name__)

TS_ENG = get_timescale_engine()
META_ENG = get_metadata_engine()

def get_symbols_from_tokens(limit: int = 50) -> List[Dict]:
    """Load symbols from tokens table (all ~190, capped)."""
    with META_ENG.connect() as conn:
        result = conn.execute(text("SELECT symbol, coingecko_id FROM tokens ORDER BY symbol LIMIT :limit"), {'limit': limit}).fetchall()
        symbols = [
            {
                'coingecko_id': row.coingecko_id,
                'label': row.symbol,
                'use_ccxt_symbol': f"{row.symbol}/USDT",  
                'exchange': 'binance'
            }
            for row in result if row.coingecko_id
        ]
    logger.info("Loaded %d symbols from tokens", len(symbols))
    return symbols

def get_last_success(pipeline: str) -> datetime:
    """Get last successful timestamp for deltas."""
    with TS_ENG.connect() as conn:
        last = conn.execute(
            text("SELECT last_success FROM ingestion_jobs WHERE pipeline = :pipeline ORDER BY last_success DESC LIMIT 1"),
            {'pipeline': pipeline}
        ).scalar()
    return last or (datetime.now(timezone.utc) - timedelta(hours=1))

async def run_backfill(symbols: List[Dict] = None):
    """Full historical backfill (run once)."""
    symbols = symbols or get_symbols_from_tokens(limit=50)
    logger.info("Starting backfill for %d symbols", len(symbols))
    
    for i, s in enumerate(symbols):
        logger.info("Backfilling %s/%s: %s", i+1, len(symbols), s['label'])
        try:
            backfill_ohlcv_coingecko(s['coingecko_id'], days='365', symbol_label=s.get('label'))
            backfill_ohlcv_ccxt(s['exchange'], s['use_ccxt_symbol'], timeframe='1h')
        except Exception as e:
            logger.error(f"Backfill failed for {s['label']}: {e}")
            continue  # Skip to next symbol
        time.sleep(1)
    
    try:
        scan_eth_transfers(batch_blocks=500)
        ingest_cryptopanic()
        ingest_reddit_praw(limit=100)
        ingest_fng()
    except Exception as e:
        logger.error(f"Backfill of news/onchain failed: {e}")
        
    logger.info("Backfill complete")

async def run_ingestion_cycle(pipeline: str = 'full_cycle', symbols: List[Dict] = None):
    """Delta cycle for real-time (run every 5min)."""
    start_time = datetime.now()
    symbols = symbols or get_symbols_from_tokens(limit=50)
    logger.info(f"Starting {pipeline} with %d symbols", len(symbols))
    
    since_ts = int(get_last_success(pipeline).timestamp() * 1000)
    
    loop = asyncio.get_running_loop()
    tasks = []
    for s in symbols:
        task = loop.run_in_executor(
            None,  # Use the default executor
            backfill_ohlcv_ccxt,
            s['exchange'],
            s['use_ccxt_symbol'],
            '1h',
            since_ts
        )
        tasks.append(task)
    
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Cycle fetch failed: {e}")

    try:
        scan_eth_transfers(batch_blocks=500)
        ingest_cryptopanic()
        ingest_reddit_praw(limit=50)
        ingest_fng()
        
        # MLflow trigger for sentiment/forecast
        mlflow.set_experiment("crypto_ingestion")
        with mlflow.start_run():
            mlflow.log_param("symbols_count", len(symbols))
        
        update_ingestion_job(IngestionJob(
            pipeline=pipeline,
            last_run=start_time,
            last_success=datetime.now(),
            details={'symbols': [s['label'] for s in symbols]}
        ))
        logger.info(f"{pipeline} complete")
    except Exception as e:
        logger.error(f"{pipeline} failed: {e}")

async def run_polling(symbols: List[str] = None):
    """Real-time trades polling."""
    symbols = symbols or [s['use_ccxt_symbol'] for s in get_symbols_from_tokens(limit=10)]
    logger.info("Polling %d symbols", len(symbols))
    tasks = [asyncio.create_task(poll_trades_ccxt('binance', symbol, poll_interval=5.0)) for symbol in symbols]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == '--backfill':
        asyncio.run(run_backfill())
    elif arg == '--poll':
        asyncio.run(run_polling())
    else:
        asyncio.run(run_ingestion_cycle())