import time
import logging
from datetime import datetime, timezone
from typing import Optional, List

import ccxt

from core.config import settings
from data.validation import OHLCV, Trade
from data.storage.crud import upsert_ohlcv, upsert_trades, get_token 

logger = logging.getLogger(__name__)


def get_valid_ccxt_pairs(exchange_id: str = 'binance') -> List[str]:
    exchange = ccxt.binance({'enableRateLimit': True})
    markets = exchange.load_markets()
    usdt_pairs = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
    return usdt_pairs

def backfill_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str = '1m', since_ts_ms: Optional[int] = None, limit: int = 1000):
    valid_pairs = get_valid_ccxt_pairs(exchange_id)
    if symbol not in valid_pairs:
        logger.warning(f"{exchange_id} does not have market {symbol}; skipping")
        return 0

    ExchangeClass = getattr(ccxt, exchange_id)
    exchange = ExchangeClass({'enableRateLimit': True})
    if exchange_id.lower() == 'binance' and settings.BINANCE_API_KEY:
        exchange.apiKey = settings.BINANCE_API_KEY
        exchange.secret = settings.BINANCE_API_SECRET

    all_bars = []
    since = since_ts_ms
    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.RateLimitExceeded:
            logger.warning(f"CCXT rate limit for {symbol}; sleeping 60s")
            time.sleep(60)
            continue
        except Exception as e:
            logger.exception("CCXT fetch_ohlcv error: %s", e)
            break
        if not bars:
            break
        base_symbol = symbol.split('/')[0]
        if not get_token(base_symbol):
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
        upsert_ohlcv(all_bars)
    logger.info("CCXT backfill done: %s %s bars=%d", exchange_id, symbol, len(all_bars))
    return len(all_bars)

def poll_trades_ccxt(exchange_id: str, symbol: str, poll_interval: float = 2.0):
    ExchangeClass = getattr(ccxt, exchange_id)
    exchange = ExchangeClass({'enableRateLimit': True})
    last_seen = set()
    while True:
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
                last_seen.add(trade_id)
            if rows:
                upsert_trades(rows)
            if len(last_seen) > 5000:
                last_seen = set(list(last_seen)[-3000:])
            time.sleep(poll_interval)
        except ccxt.RateLimitExceeded:
            logger.warning("CCXT poll rate limit; sleeping 60s")
            time.sleep(60)
        except Exception as e:
            logger.exception("poll_trades_ccxt error: %s", e)
            time.sleep(5)