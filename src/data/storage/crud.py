from typing import List
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select
from data.validation import (
    OHLCV, Trade, NewsArticle, RedditPost, WhaleAlert, OnchainMetric, IngestionJob, ChainState
)
from data.storage.models import (
    Token as TokenModel, OHLCV as OHLCVModel, Trade as TradeModel, WhaleAlert as WhaleAlertModel,
    OnchainMetric as OnchainMetricModel, NewsArticle as NewsArticleModel,
    RedditPost as RedditPostModel, IngestionJob as IngestionJobModel, ChainState as ChainStateModel
)
import logging

logger = logging.getLogger(__name__)


def get_token(db: Session, symbol: str):
    try:
        token = db.execute(select(TokenModel).where(TokenModel.symbol == symbol)).scalar_one_or_none()
        return token.__dict__ if token else None
    except Exception as e:
        logger.error(f"Error fetching token {symbol}: {e}")
        return None

def upsert_ohlcv(db: Session, rows: List[OHLCV]):
    if not rows:
        return
    try:
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
        logger.info(f"Inserted {len(rows)} OHLCV rows")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting OHLCV: {e}")
        raise

def upsert_trades(db: Session, rows: List[Trade]):
    if not rows:
        return
    try:
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
        logger.info(f"Inserted {len(rows)} trades")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting trades: {e}")
        raise

def upsert_news(db: Session, articles: List[NewsArticle]):
    if not articles:
        return
    try:
        for article in articles:
            exists = db.execute(
                select(NewsArticleModel).where(
                    NewsArticleModel.id == article.id
                )
            ).scalar_one_or_none()
            if not exists:
                news = NewsArticleModel(
                    id=article.id, title=article.title, source=article.source, url=article.url,
                    published=article.published, text=article.text,
                    raw=article.raw or {}
                )
                db.add(news)
        logger.info(f"Inserted {len(articles)} news articles")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting news: {e}")
        raise

def upsert_reddit(db: Session, posts: List[RedditPost]):
    if not posts:
        return
    try:
        for post in posts:
            exists = db.execute(
                select(RedditPostModel).where(
                    RedditPostModel.id == post.id
                )
            ).scalar_one_or_none()
            if not exists:
                reddit = RedditPostModel(
                    id=post.id, subreddit=post.subreddit, author=post.author, title=post.title,
                    body=post.body, score=post.upvote_score, created=post.created,
                    raw=post.raw or {}
                )
                db.add(reddit)
        logger.info(f"Inserted {len(posts)} Reddit posts")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting Reddit posts: {e}")
        raise

def upsert_whale_alerts(db: Session, alerts: List[WhaleAlert]):
    if not alerts:
        return
    try:
        for alert in alerts:
            exists = db.execute(
                select(WhaleAlertModel).where(
                    WhaleAlertModel.tx_hash == alert.tx_hash
                )
            ).scalar_one_or_none()
            if not exists:
                whale = WhaleAlertModel(
                    time=alert.time, tx_hash=alert.tx_hash, chain=alert.chain,
                    from_address=alert.from_address, to_address=alert.to_address,
                    amount=alert.amount, asset=alert.asset, raw=alert.raw or {}
                )
                db.add(whale)
        logger.info(f"Inserted {len(alerts)} whale alerts")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting whale alerts: {e}")
        raise

def upsert_onchain_metrics(db: Session, metrics: List[OnchainMetric]):
    if not metrics:
        return
    try:
        for metric in metrics:
            exists = db.execute(
                select(OnchainMetricModel).where(
                    OnchainMetricModel.time == metric.time,
                    OnchainMetricModel.chain == metric.chain,
                    OnchainMetricModel.metric == metric.metric
                )
            ).scalar_one_or_none()
            if not exists:
                onchain = OnchainMetricModel(
                    time=metric.time, chain=metric.chain, metric=metric.metric,
                    value=metric.value, raw=metric.raw or {}
                )
                db.add(onchain)
        logger.info(f"Inserted {len(metrics)} onchain metrics")
    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting onchain metrics: {e}")
        raise

def get_last_success(db: Session, pipeline: str):
    try:
        job = db.execute(
            select(IngestionJobModel).where(IngestionJobModel.pipeline == pipeline)
            .order_by(IngestionJobModel.last_success.desc())
        ).scalar_one_or_none()
        return job.last_success if job else (datetime.now(timezone.utc) - timedelta(hours=1))
    except Exception as e:
        db.rollback()  
        logger.error(f"Error fetching last success for {pipeline}: {e}")
        return datetime.now(timezone.utc) - timedelta(hours=1)
        raise

def update_ingestion_job(db: Session, job: IngestionJob):
    try:
        job_model = IngestionJobModel(
            pipeline=job.pipeline, last_run=job.last_run, last_success=job.last_success,
            details=job.details or {}
        )
        db.merge(job_model)
        logger.info(f"Updated ingestion job for pipeline {job.pipeline}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating ingestion job: {e}")
        raise

def update_chain_state(db: Session, state: ChainState):
    try:
        chain_state = ChainStateModel(
            chain=state.chain, last_block=state.last_block, last_updated=state.last_updated
        )
        db.merge(chain_state)
        logger.info(f"Updated chain state for {state.chain}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating chain state: {e}")
        raise