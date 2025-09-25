import json
from typing import Optional, List, Dict
from decimal import Decimal

from sqlalchemy import text

from core.database import get_timescale_engine, get_metadata_engine
from data.validation import OHLCV, Trade, NewsArticle, RedditPost, WhaleAlert, OnchainMetric, IngestionJob, ChainState

import logging

logger = logging.getLogger(__name__)

META_ENG = get_metadata_engine() 
TS_ENG = get_timescale_engine()   

def get_token(symbol: str) -> Optional[Dict]:
    try:
        with META_ENG.connect() as conn:
            r = conn.execute(text("SELECT * FROM tokens WHERE symbol = :symbol"), {'symbol': symbol}).fetchone()
            return r._asdict() if r else None
    except Exception as e:
        logger.error(f"Error fetching token {symbol}: {e}")
        return None

def upsert_ohlcv(rows: List[OHLCV]):
    if not rows:
        return
    insert_sql = text("""
        INSERT INTO ohlcv (time, symbol, exchange, interval, open, high, low, close, volume, raw)
        VALUES (:time, :symbol, :exchange, :interval, :open, :high, :low, :close, :volume, :raw)
        ON CONFLICT (time, symbol, exchange, interval) DO NOTHING
    """)
    params = [
        {
            'time': r.time,
            'symbol': r.symbol,
            'exchange': r.exchange,
            'interval': r.interval,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
            'raw': json.dumps(r.raw or {})
        }
        for r in rows
    ]
    try:
        with TS_ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Upserted {len(rows)} OHLCV rows")
    except Exception as e:
        logger.error(f"Error upserting OHLCV: {e}")

def upsert_trades(rows: List[Trade]):
    if not rows:
        return
    insert_sql = text("""
        INSERT INTO trades (time, exchange, symbol, trade_id, price, amount, side, raw)
        VALUES (:time, :exchange, :symbol, :trade_id, :price, :amount, :side, :raw)
        ON CONFLICT (time, exchange, symbol, trade_id) DO NOTHING
    """)
    params = [
        {
            'time': r.time,
            'exchange': r.exchange,
            'symbol': r.symbol,
            'trade_id': r.trade_id,
            'price': r.price,
            'amount': r.amount,
            'side': r.side,
            'raw': json.dumps(r.raw or {})
        }
        for r in rows
    ]
    try:
        with TS_ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Upserted {len(rows)} trades")
    except Exception as e:
        logger.error(f"Error upserting trades: {e}")

def upsert_news(articles: List[NewsArticle]):
    if not articles:
        return
    insert_sql = text("""
        INSERT INTO news_articles (id, title, source, url, published, text, raw)
        VALUES (:id, :title, :source, :url, :published, :text, :raw)
        ON CONFLICT (id) DO NOTHING
    """)
    params = [
        {
            'id': a.id,
            'title': a.title,
            'source': a.source,
            'url': a.url,
            'published': a.published,
            'text': a.text,
            'raw': json.dumps(a.raw or {})
        }
        for a in articles
    ]
    try:
        with TS_ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Upserted {len(articles)} news articles")
    except Exception as e:
        logger.error(f"Error upserting news: {e}")

def upsert_reddit(posts: List[RedditPost]):
    if not posts:
        return
    insert_sql = text("""
        INSERT INTO reddit_posts (id, subreddit, author, title, body, score, created, raw)
        VALUES (:id, :subreddit, :author, :title, :body, :score, :created, :raw)
        ON CONFLICT (id) DO NOTHING
    """)
    params = [
        {
            'id': p.id,
            'subreddit': p.subreddit,
            'author': p.author,
            'title': p.title,
            'body': p.body,
            'score': p.score,
            'created': p.created,
            'raw': json.dumps(p.raw or {})
        }
        for p in posts
    ]
    try:
        with TS_ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Upserted {len(posts)} Reddit posts")
    except Exception as e:
        logger.error(f"Error upserting Reddit posts: {e}")

def upsert_whale_alerts(alerts: List[WhaleAlert]):
    if not alerts:
        return
    insert_sql = text("""
        INSERT INTO whale_alerts (time, tx_hash, chain, from_address, to_address, amount, asset, raw)
        VALUES (:time, :tx_hash, :chain, :from_address, :to_address, :amount, :asset, :raw)
        ON CONFLICT (time, tx_hash) DO NOTHING
    """)
    params = [
        {
            'time': a.time,
            'tx_hash': a.tx_hash,
            'chain': a.chain,
            'from_address': a.from_address,
            'to_address': a.to_address,
            'amount': a.amount,  
            'asset': a.asset,
            'raw': json.dumps(a.raw or {})
        }
        for a in alerts
    ]
    try:
        with TS_ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Upserted {len(alerts)} whale alerts")
    except Exception as e:
        logger.error(f"Error upserting whale alerts: {e}")

def upsert_onchain_metrics(metrics: List[OnchainMetric]):
    if not metrics:
        return
    insert_sql = text("""
        INSERT INTO onchain_metrics (time, chain, metric, value, raw)
        VALUES (:time, :chain, :metric, :value, :raw)
        ON CONFLICT (time, chain, metric) DO NOTHING
    """)
    params = [
        {
            'time': m.time,
            'chain': m.chain,
            'metric': m.metric,
            'value': m.value,  
            'raw': json.dumps(m.raw or {})
        }
        for m in metrics
    ]
    try:
        with TS_ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Upserted {len(metrics)} onchain metrics")
    except Exception as e:
        logger.error(f"Error upserting onchain metrics: {e}")

def update_ingestion_job(job: IngestionJob):
    update_sql = text("""
        INSERT INTO ingestion_jobs (pipeline, last_run, last_success, details)
        VALUES (:pipeline, :last_run, :last_success, :details)
        ON CONFLICT (pipeline) DO UPDATE SET
            last_run = EXCLUDED.last_run,
            last_success = EXCLUDED.last_success,
            details = EXCLUDED.details
    """)
    try:
        with TS_ENG.begin() as conn:
            conn.execute(update_sql, {
                'pipeline': job.pipeline,
                'last_run': job.last_run,
                'last_success': job.last_success,
                'details': json.dumps(job.details or {})
            })
        logger.info(f"Updated ingestion job for pipeline {job.pipeline}")
    except Exception as e:
        logger.error(f"Error updating ingestion job: {e}")

def update_chain_state(state: ChainState):
    update_sql = text("""
        INSERT INTO chain_state (chain, last_block, last_updated)
        VALUES (:chain, :last_block, :last_updated)
        ON CONFLICT (chain) DO UPDATE SET
            last_block = EXCLUDED.last_block,
            last_updated = EXCLUDED.last_updated
    """)
    try:
        with TS_ENG.begin() as conn:
            conn.execute(update_sql, {
                'chain': state.chain,
                'last_block': state.last_block,
                'last_updated': state.last_updated
            })
        logger.info(f"Updated chain state for {state.chain}")
    except Exception as e:
        logger.error(f"Error updating chain state: {e}")

