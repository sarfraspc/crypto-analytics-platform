import time
import logging
from datetime import datetime, timezone, timedelta
import json
from typing import Optional, List, Dict

import ccxt
from pycoingecko import CoinGeckoAPI
import requests
from web3 import Web3
import praw

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from core.database import get_timescale_engine
from core.config import settings
from data.validation import OHLCV, Trade, NewsArticle, RedditPost, WhaleAlert
from data.crud import (
    upsert_ohlcv,
    upsert_trades,
    upsert_news,
    upsert_reddit,
    upsert_whale_alerts,
)

logger = logging.getLogger(__name__)

TS_ENG = get_timescale_engine()


# Helpers
def _to_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _now():
    return datetime.now(timezone.utc)

# CoinGecko: historical OHLCV
CG = CoinGeckoAPI()

def backfill_ohlcv_coingecko(symbol_coingecko_id: str, vs_currency: str = "usd", days: str = "max", interval: str = "daily", symbol_label: str = None):
    """
    Backfill OHLCV using CoinGecko's /coins/{id}/ohlc endpoint (historic market data).
    days: number or 'max'
    """
    logger.info("Starting CoinGecko backfill: %s", symbol_coingecko_id)
    # CoinGecko: /coins/{id}/ohlc?vs_currency=usd&days=max
    try:
        raw = CG.get_coin_ohlc_by_id(id=symbol_coingecko_id, vs_currency=vs_currency, days=days)
    except Exception as e:
        logger.exception("CoinGecko request failed: %s", e)
        return

    rows = []
    for point in raw:  # each point: [timestamp_ms, open, high, low, close]
        ts = datetime.fromtimestamp(point[0] / 1000.0, tz=timezone.utc)
        o, h, l, c = point[1], point[2], point[3], point[4]
        rows.append(OHLCV(
            time=ts,
            symbol=(symbol_label or f"{symbol_coingecko_id}".upper()),
            interval=interval,
            exchange="coingecko",
            open=o, high=h, low=l, close=c, volume=0, # Volume is not provided by this endpoint
            raw={'source':'coingecko'}
        ))
    upsert_ohlcv(rows)
    logger.info("CoinGecko backfill inserted %d rows for %s", len(rows), symbol_coingecko_id)

# CCXT: realtime polling + historical via exchange
def backfill_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str = '1m', since_ts_ms: Optional[int] = None, limit: int = 1000):
    Exchange = getattr(ccxt, exchange_id)
    exchange = Exchange({'enableRateLimit': True})
    if settings.BINANCE_API_KEY and exchange_id.lower() == 'binance':
        exchange.apiKey = settings.BINANCE_API_KEY
        exchange.secret = settings.BINANCE_API_SECRET

    all_bars = []
    since = since_ts_ms
    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            logger.exception("CCXT fetch_ohlcv error: %s", e)
            break
        if not bars:
            break
        for bar in bars:
            ts = datetime.fromtimestamp(bar[0] / 1000.0, tz=timezone.utc)
            all_bars.append(OHLCV(time=ts, symbol=symbol, interval=timeframe, exchange=exchange_id, open=bar[1], high=bar[2], low=bar[3], close=bar[4], volume=bar[5], raw={'ccxt': bar}))
        since = bars[-1][0] + 1
        if len(bars) < limit:
            break
        time.sleep(0.2)
    if all_bars:
        upsert_ohlcv(all_bars)
    logger.info("CCXT backfill done: %s %s bars=%d", exchange_id, symbol, len(all_bars))
    return len(all_bars)

def poll_trades_ccxt(exchange_id: str, symbol: str, poll_interval: float = 2.0):
    Exchange = getattr(ccxt, exchange_id)
    exchange = Exchange({'enableRateLimit': True})
    last_seen = set()
    while True:
        try:
            trades = exchange.fetch_trades(symbol, limit=1000)
            rows = []
            for t in trades:
                ts = datetime.fromtimestamp(t['timestamp'] / 1000.0, tz=timezone.utc)
                trade_id = str(t.get('id') or f"{t.get('timestamp')}-{t.get('price')}-{t.get('amount')}")
                if trade_id in last_seen:
                    continue
                rows.append(Trade(time=ts, exchange=exchange_id, symbol=symbol, trade_id=trade_id, price=float(t['price']), amount=float(t['amount']), side=t.get('side'), raw=t))
                last_seen.add(trade_id)
            if rows:
                upsert_trades(rows)
            # keep last_seen small
            if len(last_seen) > 5000:
                last_seen = set(list(last_seen)[-3000:])
            time.sleep(poll_interval)
        except Exception as e:
            logger.exception("poll_trades_ccxt error: %s", e)
            time.sleep(5)

# CryptoPanic news ingestion
def ingest_cryptopanic(api_key: str = None, limit: int = 50, max_retries: int = 3):
    key = api_key or settings.CRYPTOPANIC_API_KEY
    if not key:
        logger.warning("CryptoPanic API key not configured")
        return
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={key}&kind=news&public=true"
    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            j = resp.json()
            posts = []
            for p in j.get('results', [])[:limit]:
                aid = p.get('id') or (p.get('published_at') or "") + (p.get('title') or '')
                article = NewsArticle(
                    id=str(aid),
                    title=p.get('title'),
                    source=p.get('source', {}).get('title'),
                    url=p.get('url'),
                    published=(datetime.fromisoformat(p.get('published_at').replace('Z','+00:00')) if p.get('published_at') else None),
                    text=p.get('body') or p.get('title'),
                    raw=p
                )
                posts.append(article)
            if posts:
                upsert_news(posts)
                logger.info("Inserted %d CryptoPanic articles", len(posts))
            return
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "") or ""
            logger.warning("CryptoPanic HTTPError %s - body: %s", getattr(e.response, "status_code", "N/A"), body[:1000])
            # If provider explicitly says monthly quota exceeded — bail out (no point retrying)
            if getattr(e.response, "status_code", None) == 429 and ("quota" in body.lower() or "monthly quota" in body.lower()):
                logger.error("CryptoPanic monthly quota exceeded (API response). Skipping CryptoPanic ingestion until plan upgraded.")
                return
            if getattr(e.response, "status_code", None) == 429:
                logger.warning("CryptoPanic rate limited (429), backing off %s sec (attempt %d)", backoff, attempt+1)
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                logger.exception("CryptoPanic ingestion failed with HTTPError: %s", e)
                break
        except Exception as e:
            logger.exception("CryptoPanic error: %s", e)
            time.sleep(backoff)
            backoff *= 2
    logger.info("CryptoPanic ingestion finished (attempts exhausted or failed).")

# Fear & Greed Index
def ingest_fng():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        j = resp.json()
        data = j.get('data', [])
        if not data:
            return None
        item = data[0]
        # store into ingestion_jobs as quick cache (or add a dedicated table)
        with TS_ENG.begin() as conn:
            conn.execute(text("""
                INSERT INTO ingestion_jobs (pipeline, last_run, last_success, details)
                VALUES ('fear_greed', now(), now(), :details)
                ON CONFLICT (pipeline) DO UPDATE SET last_run = now(), last_success = now(), details = EXCLUDED.details
            """), {'details': json.dumps(item)})
        logger.info("Ingested Fear & Greed: %s %s", item.get('value'), item.get('value_classification'))
        return item
    except Exception as e:
        logger.exception("FNG ingestion failed: %s", e)
        return None

def ingest_reddit_praw(subreddit: str = "cryptocurrency", limit: int = 100):
    cid = getattr(settings, 'REDDIT_CLIENT_ID', None)
    secret = getattr(settings, 'REDDIT_CLIENT_SECRET', None)
    ua = getattr(settings, 'REDDIT_USER_AGENT', None)
    if not (cid and secret and ua):
        logger.warning("PRAW credentials not configured; skipping Reddit ingestion.")
        return

    try:
        reddit = praw.Reddit(client_id=cid, client_secret=secret, user_agent=ua, request_timeout=20)
        posts = []
        for submission in reddit.subreddit(subreddit).new(limit=limit):
            created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            p = RedditPost(
                id=str(submission.id),
                subreddit=subreddit,
                author=str(submission.author) if submission.author else None,
                title=submission.title,
                body=getattr(submission, 'selftext', None),
                score=getattr(submission, 'score', None),
                created=created,
                raw={
                    'url': submission.url,
                    'num_comments': submission.num_comments,
                    'permalink': submission.permalink
                }
            )
            posts.append(p)
        if posts:
            upsert_reddit(posts)
            logger.info("Inserted %d reddit posts (praw) for %s", len(posts), subreddit)
    except Exception as e:
        logger.exception("ingest_reddit_praw failed: %s", e)

# Reddit ingestion (Pushshift fallback)
def ingest_reddit_pushshift(subreddit: str = "cryptocurrency", limit: int = 100, max_retries: int = 3):
    """
    Uses Pushshift.io (no auth) to fetch recent posts.
    """
    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={limit}&sort=desc&sort_type=created_utc"
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            j = r.json()
            posts = []
            for item in j.get('data', []):
                created = datetime.fromtimestamp(item.get('created_utc', 0), tz=timezone.utc)
                p = RedditPost(
                    id=str(item.get('id')),
                    subreddit=subreddit,
                    author=item.get('author'),
                    title=item.get('title'),
                    body=item.get('selftext'),
                    score=item.get('score'),
                    created=created,
                    raw=item
                )
                posts.append(p)
            if posts:
                upsert_reddit(posts)
                logger.info("Inserted %d reddit posts (pushshift) for %s", len(posts), subreddit)
            return
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")
            logger.warning("Pushshift API request failed with status code %s, backing off %s sec (attempt %d). Body: %s", e.response.status_code, backoff, attempt+1, body[:1000])
            time.sleep(backoff)
            backoff *= 2
            continue
        except Exception as e:
            logger.exception("Reddit Pushshift ingestion failed: %s", e)
            time.sleep(backoff)
            backoff *= 2
    logger.info("Reddit Pushshift ingestion finished (attempts exhausted or failed).")

# Simple on-chain scanner (Ethereum) - Alchemy
ALCHEMY_KEY = getattr(settings, 'ALCHEMY_API_KEY', None)
W3 = None
if ALCHEMY_KEY:
    W3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"))

# ensure TRANSFER_TOPIC is a 0x-prefixed hex string (place near W3 init)
try:
    TRANSFER_TOPIC = Web3.toHex(Web3.keccak(text="Transfer(address,address,uint256)")) if W3 else None
except Exception:
    # fallback for older web3 versions
    TRANSFER_TOPIC = Web3.to_hex(Web3.keccak(text="Transfer(address,address,uint256)")) if W3 else None

def get_last_chain_block(chain: str = 'ethereum') -> Optional[int]:
    with TS_ENG.begin() as conn:
        r = conn.execute(text("SELECT last_block FROM chain_state WHERE chain=:chain"), {'chain': chain}).fetchone()
        return int(r[0]) if r else None

def set_last_chain_block(chain: str, blk: int):
    with TS_ENG.begin() as conn:
        conn.execute(text("""
            INSERT INTO chain_state (chain, last_block, last_updated)
            VALUES (:chain, :blk, now())
            ON CONFLICT (chain) DO UPDATE SET last_block = EXCLUDED.last_block, last_updated = now()
        """), {'chain': chain, 'blk': int(blk)})

def scan_eth_transfers(batch_blocks: int = 2000, threshold_wei: int = 10**18, max_blocks_per_call: int = 10):
    """
    Scan blocks from last+1 .. head, calling eth_getLogs in chunks of max_blocks_per_call (Alchemy free tier = 10).
    """
    if not W3:
        logger.warning("Alchemy key not configured; skipping on-chain scan")
        return

    chain = 'ethereum'
    last = get_last_chain_block(chain)
    head = W3.eth.block_number
    if last is None:
        last = max(head - 1000, 0)
    to_block = min(head, last + batch_blocks)
    if to_block < last + 1:
        logger.info("No new blocks to scan (%s -> %s)", last+1, to_block)
        return

    logger.info("Scanning blocks %d -> %d (total %d) with chunk_size=%d", last+1, to_block, to_block - last, max_blocks_per_call)

    alerts = []
    start = last + 1
    while start <= to_block:
        end = min(start + max_blocks_per_call - 1, to_block)
        logger.debug("Requesting logs %d -> %d", start, end)
        try:
            logs = W3.eth.get_logs({'fromBlock': start, 'toBlock': end, 'topics': [TRANSFER_TOPIC]})
        except Exception as e:
            # log provider response / message and skip this chunk (prevents infinite loop)
            try:
                logger.exception("get_logs failed for range %d-%d: %s", start, end, e)
            except Exception:
                logger.error("get_logs failed for range %d-%d (unable to format exception)", start, end)
            # Move on to next chunk after short pause
            start = end + 1
            time.sleep(0.5)
            continue

        for l in logs:
            # topics -> addresses
            topics = l.get('topics', []) if isinstance(l, dict) else []
            if len(topics) < 3:
                continue
            try:
                from_addr = '0x' + topics[1].hex()[-40:]
                to_addr = '0x' + topics[2].hex()[-40:]
            except Exception:
                logger.debug("Skipping log with malformed topics: %s", topics)
                continue

            # robust parsing for the data field (covers str, bytes, HexBytes, bytearray)
            data_field = l.get('data', b'0x0')
            amount = 0
            try:
                # raw bytes-like (including HexBytes)
                if isinstance(data_field, (bytes, bytearray)):
                    amount = int.from_bytes(data_field, byteorder='big')
                elif isinstance(data_field, str):
                    hexstr = data_field
                    if hexstr.startswith(('0x', '0X')):
                        amount = int(hexstr, 16)
                    else:
                        amount = int(hexstr, 16)
                else:
                    # objects like HexBytes typically expose .hex()
                    if hasattr(data_field, 'hex'):
                        hexstr = data_field.hex()
                        amount = int(hexstr, 16) if hexstr else 0
                    else:
                        amount = int(str(data_field), 16)
            except Exception as e:
                logger.debug("Failed to parse log data (skipping). data_field=%r error=%s", data_field, e)
                continue

            if amount >= threshold_wei:
                try:
                    blk = W3.eth.get_block(l['blockNumber'])
                    t = datetime.fromtimestamp(blk['timestamp'], tz=timezone.utc)
                    tx_hash_obj = l.get('transactionHash')
                    tx_hash = tx_hash_obj.hex() if hasattr(tx_hash_obj, 'hex') else str(tx_hash_obj)
                    wa = WhaleAlert(
                        time=t,
                        tx_hash=tx_hash,
                        chain=chain,
                        from_address=from_addr,
                        to_address=to_addr,
                        amount=amount,
                        asset=None,
                        raw=dict(l) if isinstance(l, dict) else {}
                    )
                    alerts.append(wa)
                except Exception:
                    logger.exception("Failed to create WhaleAlert from log: %s", l)
                    continue

        time.sleep(0.05)
        start = end + 1

    if alerts:
        try:
            upsert_whale_alerts(alerts)
            logger.info("Inserted %d whale alerts", len(alerts))
        except Exception:
            logger.exception("Failed to upsert whale alerts")
    # persist last scanned block as to_block
    try:
        set_last_chain_block(chain, to_block)
    except Exception:
        logger.exception("Failed to set last chain block to %s", to_block)


# Convenience CLI helpers
def run_quick_backfill_symbols(symbols: List[Dict]):
    """
    symbols: list of dicts: {'coingecko_id':'bitcoin', 'label':'BTC/USDT', 'use_ccxt_symbol':'BTC/USDT', 'exchange':'binance'}
    """
    for s in symbols:
        try:
            backfill_ohlcv_coingecko(s['coingecko_id'], days='365', interval='1d', symbol_label=s.get('label'))
        except Exception:
            logger.exception("backfill coingecko failed for %s", s)

def run_all_once():
    # sample quick run: seed tokens, backfill 1 symbol, ingest news/reddit/fng, scan chain once
    run_quick_backfill_symbols([{'coingecko_id':'bitcoin', 'label':'BTC/USDT'}])
    ingest_cryptopanic()
    ingest_reddit_praw('cryptocurrency', limit=100)
    ingest_fng()
    scan_eth_transfers(batch_blocks=500, threshold_wei=10**18)
