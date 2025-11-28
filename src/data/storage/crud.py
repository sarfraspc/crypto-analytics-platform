"""CRUD operations for data storage in TimescaleDB and PostgreSQL."""

import logging
from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from data.storage.models import (
    IngestionJob as IngestionJobModel,
    NewsArticle as NewsArticleModel,
    OHLCV as OHLCVModel,
    OnchainMetric as OnchainMetricModel,
    RedditPost as RedditPostModel,
    TASignal as TASignalModel,
    TASignalHistory as TASignalHistoryModel,
    Token as TokenModel,
    Trade as TradeModel,
    WhaleAlert as WhaleAlertModel,
)
from data.validation import (
    IngestionJob,
    NewsArticle,
    OHLCV,
    OnchainMetric,
    RedditPost,
    TASignal,
    TASignalHistory,
    Trade,
    WhaleAlert,
)

logger = logging.getLogger(__name__)


def get_token(db: Session, symbol: str):
    """Fetch token metadata by symbol from database."""
    try:
        token = db.execute(select(TokenModel).where(TokenModel.symbol == symbol)).scalar_one_or_none()
        return token.__dict__ if token else None
    except Exception as e:
        logger.error(f"Error fetching token {symbol}: {e}")
        return None


def upsert_ohlcv(db: Session, rows: List[OHLCV]):
    """Insert OHLCV rows, skipping duplicates based on composite key."""
    if not rows:
        return
    try:
        inserted_count = 0
        for row in rows:
            exists = db.execute(
                select(OHLCVModel).where(
                    OHLCVModel.time == row.time,
                    OHLCVModel.symbol == row.symbol,
                    OHLCVModel.exchange == row.exchange,
                    OHLCVModel.interval == row.interval
                )
            ).scalar_one_or_none()
            if not exists:
                ohlcv = OHLCVModel(
                    time=row.time, symbol=row.symbol, exchange=row.exchange, interval=row.interval,
                    open=row.open, high=row.high, low=row.low, close=row.close, volume=row.volume,
                    raw=row.raw or {}
                )
                db.add(ohlcv)
                inserted_count += 1
        db.commit()
        logger.info(f"Committed {inserted_count} OHLCV rows (attempted {len(rows)})")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting OHLCV: {e}")
        raise


def upsert_trades(db: Session, rows: List[Trade]):
    """Insert trade rows, skipping duplicates based on composite key."""
    if not rows:
        return
    try:
        inserted_count = 0
        for row in rows:
            exists = db.execute(
                select(TradeModel).where(
                    TradeModel.time == row.time,
                    TradeModel.exchange == row.exchange,
                    TradeModel.symbol == row.symbol,
                    TradeModel.trade_id == row.trade_id
                )
            ).scalar_one_or_none()
            if not exists:
                trade = TradeModel(
                    time=row.time, exchange=row.exchange, symbol=row.symbol, trade_id=row.trade_id,
                    price=row.price, amount=row.amount, side=row.side, raw=row.raw or {}
                )
                db.add(trade)
                inserted_count += 1
        db.commit()
        logger.info(f"Committed {inserted_count} trades (attempted {len(rows)})")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting trades: {e}")
        raise


def upsert_news(db: Session, articles: List[NewsArticle]):
    """Insert news articles, skipping duplicates by ID."""
    if not articles:
        return
    
    try:
        inserted_count = 0
        for article in articles:
            exists = db.execute(select(NewsArticleModel).where(NewsArticleModel.id == article.id)).scalar_one_or_none()
            if not exists:
                db.add(NewsArticleModel(
                    id=article.id, title=article.title, source=article.source, url=article.url,
                    published=article.published, text=article.text,
                    raw=article.raw or {}
                ))
                inserted_count += 1
        
        db.commit()
        logger.info(f"Committed {inserted_count} news articles (attempted {len(articles)})")
    except Exception as e:
        db.rollback()
        logger.error(f"Upsert failed for news articles: {str(e)} | Type: {type(e).__name__} | Full: {repr(e)}")
        raise


def upsert_reddit(db: Session, posts: List[RedditPost]):
    """Insert Reddit posts, skipping duplicates by ID."""
    if not posts:
        return
    
    try:
        inserted_count = 0
        for post in posts:
            exists = db.execute(select(RedditPostModel).where(RedditPostModel.id == post.id)).scalar_one_or_none()
            if not exists:
                db.add(RedditPostModel(
                    id=post.id, subreddit=post.subreddit, author=post.author, title=post.title,
                    body=post.body, score=post.score, created=post.created,
                    raw=post.raw or {}
                ))
                inserted_count += 1
        
        db.commit()
        logger.info(f"Committed {inserted_count} Reddit posts (attempted {len(posts)})")
    except Exception as e:
        db.rollback()
        logger.error(f"Upsert failed for Reddit posts: {str(e)} | Type: {type(e).__name__} | Full: {repr(e)}")
        raise


def upsert_whale_alerts(db: Session, alerts: List[WhaleAlert], chunk_size: int = 500):
    """Bulk upsert whale alerts in chunks with conflict handling."""
    if not alerts:
        return

    try:
        total_inserted = 0
        for i in range(0, len(alerts), chunk_size):
            batch = alerts[i:i + chunk_size]

            values = [
                {
                    'time': a.time,
                    'tx_hash': a.tx_hash,
                    'chain': a.chain,
                    'from_address': a.from_address,
                    'to_address': a.to_address,
                    'amount': a.amount,
                    'usd_value': a.usd_value,  # Explicitly include usd_value
                    'asset': a.asset,
                    'raw': a.raw or {}
                }
                for a in batch
            ]

            stmt = insert(WhaleAlertModel).values(values)
            stmt = stmt.on_conflict_do_nothing(index_elements=['time', 'tx_hash', 'asset'])
            result = db.execute(stmt)
            total_inserted += result.rowcount  # Accurate count from DB

        db.commit()  # Commit after all chunks
        logger.info(f"Committed {total_inserted} whale alerts (attempted {len(alerts)} in chunks)")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting whale alerts: {e}")
        raise


def upsert_onchain_metrics(db: Session, metrics: List[OnchainMetric]):
    """Bulk upsert on-chain metrics with conflict handling."""
    if not metrics:
        return
    
    values = [
        {
            'time': metric.time,
            'chain': metric.chain,
            'metric': metric.metric,
            'value': metric.value,
            'raw': metric.raw or {}
        }
        for metric in metrics
    ]
    
    stmt = insert(OnchainMetricModel).values(values)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['time', 'chain', 'metric']
    )
    
    try:
        result = db.execute(stmt)
        db.commit()  # Commit
        logger.info(f"Committed {result.rowcount} onchain metrics")
    except Exception as e:
        db.rollback()
        logger.error(f"Upsert failed for onchain metrics: {str(e)} | Type: {type(e).__name__} | Full: {repr(e)}")
        raise


def get_last_success(db: Session, pipeline: str):
    """Get last successful run timestamp for a pipeline, defaults to 1 hour ago."""
    try:
        job = db.execute(
            select(IngestionJobModel).where(IngestionJobModel.pipeline == pipeline)
            .order_by(IngestionJobModel.last_success.desc())
        ).scalar_one_or_none()
        return job.last_success if job else (datetime.now(timezone.utc) - timedelta(hours=1))
    except Exception as e:
        logger.error(f"Error fetching last success for {pipeline}: {e}")
        return datetime.now(timezone.utc) - timedelta(hours=1)


def update_ingestion_job(db: Session, job: IngestionJob):
    """Update or insert ingestion job tracking record."""
    try:
        job_model = IngestionJobModel(
            pipeline=job.pipeline, last_run=job.last_run, last_success=job.last_success,
            details=job.details or {}
        )
        db.merge(job_model)
        db.commit()  # Commit
        logger.info(f"Committed ingestion job for pipeline {job.pipeline}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating ingestion job: {e}")
        raise


def upsert_ta_signals(db: Session, signals: List[TASignal]):
    """Upsert TA signal snapshots with conflict update on composite key."""
    if not signals:
        return
    try:
        values = [
            {
                'symbol': s.symbol,
                'exchange': s.exchange,
                'interval': s.interval,
                'time': s.time or datetime.now(timezone.utc),
                'signal': s.signal,
                'rsi': s.rsi,
                'macd_hist': s.macd_hist,
                'pattern': s.pattern
            }
            for s in signals
        ]
        stmt = insert(TASignalModel).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=['symbol', 'exchange', 'interval'],
            set_={
                'time': stmt.excluded.time,
                'signal': stmt.excluded.signal,
                'rsi': stmt.excluded.rsi,
                'macd_hist': stmt.excluded.macd_hist,
                'pattern': stmt.excluded.pattern
            }
        )
        result = db.execute(stmt)
        db.commit()  # Commit
        logger.info(f"Committed {result.rowcount} TA snapshot rows")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting ta_signals: {e}")
        raise


def insert_ta_signals_history(db: Session, signals: List[TASignalHistory]):
    """Insert TA signal history records with conflict update."""
    if not signals:
        return
    try:
        values = [
            {
                'time': s.time,
                'symbol': s.symbol,
                'exchange': s.exchange,
                'interval': s.interval,
                'signal': s.signal,
                'rsi': s.rsi,
                'macd_hist': s.macd_hist,
                'pattern': s.pattern
            }
            for s in signals
        ]
        stmt = insert(TASignalHistoryModel).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=['time', 'symbol', 'exchange', 'interval'],
            set_= {
                'signal': stmt.excluded.signal,
                'rsi': stmt.excluded.rsi,
                'macd_hist': stmt.excluded.macd_hist,
                'pattern': stmt.excluded.pattern
            }
        )
        result = db.execute(stmt)
        db.commit()  # Commit
        logger.info(f"Committed {result.rowcount} TA history rows")
    except Exception as e:
        db.rollback()
        logger.error(f"Error inserting ta_signals_history: {e}")
        raise
