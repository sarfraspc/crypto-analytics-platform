import json
from typing import Optional, List, Dict

from sqlalchemy import text

from core.database import get_timescale_engine
from data.validation import OHLCV, Trade, NewsArticle, RedditPost, WhaleAlert

TS_ENG = get_timescale_engine()

def get_token(symbol: str) -> Optional[Dict]:
    with TS_ENG.begin() as conn:
        r = conn.execute(text("SELECT * FROM tokens WHERE symbol = :symbol"), {'symbol': symbol}).fetchone()
        return r._asdict() if r else None

def upsert_ohlcv(rows: List[OHLCV]):
    insert_sql = text("""
        INSERT INTO ohlcv (time, symbol, exchange, interval, open, high, low, close, volume, raw)
        VALUES (:time, :symbol, :exchange, :interval, :open, :high, :low, :close, :volume, :raw)
        ON CONFLICT (time, symbol, exchange, interval) DO NOTHING
    """)
    with TS_ENG.begin() as conn:
        for r in rows:
            conn.execute(insert_sql, {
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
            })

def upsert_trades(rows: List[Trade]):
    insert_sql = text("""
        INSERT INTO trades (time, exchange, symbol, trade_id, price, amount, side, raw)
        VALUES (:time, :exchange, :symbol, :trade_id, :price, :amount, :side, :raw)
        ON CONFLICT (exchange, symbol, trade_id) DO NOTHING
    """)
    with TS_ENG.begin() as conn:
        for r in rows:
            conn.execute(insert_sql, {
                'time': r.time,
                'exchange': r.exchange,
                'symbol': r.symbol,
                'trade_id': r.trade_id,
                'price': r.price,
                'amount': r.amount,
                'side': r.side,
                'raw': json.dumps(r.raw or {})
            })

def upsert_news(articles: List[NewsArticle]):
    insert_sql = text("""
        INSERT INTO news_articles (id, title, source, url, published, text, raw)
        VALUES (:id, :title, :source, :url, :published, :text, :raw)
        ON CONFLICT (id) DO NOTHING
    """)
    with TS_ENG.begin() as conn:
        for a in articles:
            conn.execute(insert_sql, {
                'id': a.id,
                'title': a.title,
                'source': a.source,
                'url': a.url,
                'published': a.published,
                'text': a.text,
                'raw': json.dumps(a.raw or {})
            })

def upsert_reddit(posts: List[RedditPost]):
    insert_sql = text("""
        INSERT INTO reddit_posts (id, subreddit, author, title, body, score, created, raw)
        VALUES (:id, :subreddit, :author, :title, :body, :score, :created, :raw)
        ON CONFLICT (id) DO NOTHING
    """)
    with TS_ENG.begin() as conn:
        for p in posts:
            conn.execute(insert_sql, {
                'id': p.id,
                'subreddit': p.subreddit,
                'author': p.author,
                'title': p.title,
                'body': p.body,
                'score': p.score,
                'created': p.created,
                'raw': json.dumps(p.raw or {})
            })

def upsert_whale_alerts(alerts: List[WhaleAlert]):
    insert_sql = text("""
        INSERT INTO whale_alerts (time, tx_hash, chain, from_address, to_address, amount, asset, raw)
        VALUES (:time, :tx_hash, :chain, :from_address, :to_address, :amount, :asset, :raw)
        ON CONFLICT (time, tx_hash) DO NOTHING
    """)
    with TS_ENG.begin() as conn:
        for a in alerts:
            conn.execute(insert_sql, {
                'time': a.time,
                'tx_hash': a.tx_hash,
                'chain': a.chain,
                'from_address': a.from_address,
                'to_address': a.to_address,
                'amount': a.amount,
                'asset': a.asset,
                'raw': json.dumps(a.raw or {})
            })
