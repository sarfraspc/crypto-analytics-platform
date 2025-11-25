import os  #
os.environ['PYTHONIOENCODING'] = 'utf-8'  

import time
import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
import signal
from collections import deque

import ccxt

from data.validation import OHLCV, Trade
from data.storage.crud import upsert_ohlcv, upsert_trades, get_token
from core.logging_config import setup_logging
from core.config import settings

setup_logging()
logger = logging.getLogger(__name__)


def get_valid_ccxt_pairs(exchange_id: str | None = None):
    exchange_id = exchange_id or settings.MARKET_EXCHANGE_ID
    ExchangeClass = getattr(ccxt, exchange_id)
    exchange = ExchangeClass({'enableRateLimit': True})
    markets = exchange.load_markets()
    quote = settings.MARKET_QUOTE_SYMBOL
    usdt_pairs = [s for s in markets if s.endswith(f'/{quote}') and markets[s]['active']]
    return usdt_pairs

def backfill_ohlcv_ccxt(db_timescale: Session, db_metadata: Session, exchange_id: str, symbol: str, timeframe: str = '1m', since_ts_ms: Optional[int] = None, limit: int = 1000):
    since_str = datetime.fromtimestamp(since_ts_ms / 1000) if since_ts_ms else 'None'  
    logger.info("Starting backfill_ohlcv_ccxt for %s %s (since=%s)", exchange_id, symbol, since_str)
    valid_pairs = get_valid_ccxt_pairs(exchange_id)
    if symbol not in valid_pairs:
        logger.warning(f"{exchange_id} does not have market {symbol}; skipping")
        return 0

    ExchangeClass = getattr(ccxt, exchange_id)
    exchange = ExchangeClass({
        'enableRateLimit': True,
        'timeout': 10000, 
        'options': {
            'defaultType': 'spot',
            'recvWindow': 10000,
        },
        'encoding': 'utf-8',
    })
    exchange.rateLimit = 200  
    exchange.headers = {'Accept-Charset': 'utf-8'}
    if hasattr(exchange, 'httpResponse'):  
        pass
    exchange.verbose = False 

    all_bars = []
    since = since_ts_ms
    retry_count = 0
    max_retries = 1 
    while True:
        try:
            def patched_fetch(*args, **kwargs):
                nonlocal retry_count
                try:
                    return exchange.fetch_ohlcv(*args, **kwargs)
                except UnicodeEncodeError as ue:
                    if retry_count < max_retries:
                        retry_count += 1
                        clean_msg = "Encoding error (likely rate limit table in response); skipping symbol after retry." 
                        logger.warning(f"{clean_msg} for {symbol} (retry {retry_count})")
                        exchange.timeout = 15000
                        time.sleep(5)  
                        return exchange.fetch_ohlcv(*args, **kwargs)
                    raise Exception(clean_msg) 
                except Exception as e:
                    if 'codec' in str(e).lower() and retry_count < max_retries: 
                        pass
                    raise
            bars = patched_fetch(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e: 
            clean_error = str(e).encode('utf-8', 'ignore').decode('utf-8')  
            logger.warning(f"CCXT fetch skipped for {symbol}: {clean_error}") 
            break  
        if not bars:
            logger.info(f"No more bars for {symbol} since {since_str}") 
            break
        retry_count = 0  
        base_symbol = symbol.split('/')[0]
        if not get_token(db_metadata, base_symbol):
            logger.warning(f"Unknown symbol: {base_symbol}")
            break
        for bar in bars:
            ts = datetime.fromtimestamp(bar[0] / 1000.0, tz=timezone.utc)
            all_bars.append(OHLCV(
                time=ts, symbol=base_symbol, interval=timeframe, exchange=exchange_id,
                open=bar[1], high=bar[2], low=bar[3], close=bar[4], volume=bar[5],
                raw={'ccxt': bar}
            ))
        since = bars[-1][0] + 1 
        if len(bars) < limit:
            break
        time.sleep(1.0)  
    if all_bars:
        upsert_ohlcv(db_timescale, all_bars)
        logger.info(f"Upserted {len(all_bars)} bars for {symbol}")  
    logger.info("CCXT backfill done: %s %s bars=%d", exchange_id, symbol, len(all_bars))
    return len(all_bars)

def poll_trades_ccxt(db: Session, exchange_id: str, symbol: str, poll_interval: float = 2.0):
    logger.info("Starting poll_trades_ccxt for %s %s", exchange_id, symbol)
    shutdown = False
    def shutdown_handler(signum, frame):
        nonlocal shutdown
        logger.info("Shutdown signal received. Exiting poll_trades_ccxt...")
        shutdown = True
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    ExchangeClass = getattr(ccxt, exchange_id)
    exchange = ExchangeClass({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'recvWindow': 10000,
        },
        'encoding': 'utf-8',
    })
    last_seen = deque(maxlen=3000)
    while not shutdown:
        try:
            trades = exchange.fetch_trades(symbol, limit=1000)
            rows = []
            base_symbol = symbol.split('/')[0]
            for t in trades:
                trade_id = str(t.get('id') or f"{t.get('timestamp')}-{t.get('price')}-{t.get('amount')}")
                if trade_id in last_seen:
                    continue
                ts = datetime.fromtimestamp(t['timestamp'] / 1000.0, tz=timezone.utc)
                rows.append(Trade(
                    time=ts, exchange=exchange_id, symbol=base_symbol,
                    trade_id=trade_id, price=float(t['price']), amount=float(t['amount']),
                    side=t.get('side'), raw=t
                ))
                last_seen.append(trade_id)
            if rows:
                upsert_trades(db, rows)
            time.sleep(poll_interval) 
        except ccxt.RateLimitExceeded:
            logger.warning("CCXT poll rate limit; sleeping 60s")
            time.sleep(60)
        except Exception as e:
            logger.exception("poll_trades_ccxt error: %s", e)
            time.sleep(5) 
    return 0  
