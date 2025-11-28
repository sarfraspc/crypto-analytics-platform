"""Pydantic validation models for data ingestion and storage."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import BaseModel


class Token(BaseModel):
    """Token metadata validation model."""
    symbol: str
    coingecko_id: Optional[str] = None
    name: Optional[str] = None
    decimals: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class OHLCV(BaseModel):
    """OHLCV candlestick data validation model."""

    time: datetime
    symbol: str
    interval: str
    exchange: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

class Trade(BaseModel):
    """Trade execution data validation model."""

    time: datetime
    exchange: str
    symbol: str
    trade_id: Optional[str] = None
    price: Optional[float] = None
    amount: Optional[float] = None
    side: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

class WhaleAlert(BaseModel):
    """Whale transfer alert validation model."""

    time: datetime
    tx_hash: str
    chain: Optional[str] = None
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    amount: Optional[Decimal] = None
    usd_value: Optional[Decimal] = None
    asset: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

class OnchainMetric(BaseModel):
    """On-chain metric data validation model."""

    time: datetime
    chain: str
    metric: str
    value: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = None

class NewsArticle(BaseModel):
    """News article validation model for CryptoPanic data."""

    id: str
    title: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    published: Optional[datetime] = None
    text: Optional[str] = None
    score: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

class RedditPost(BaseModel):
    """Reddit post validation model."""

    id: str
    subreddit: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None
    upvote_score: Optional[int] = None
    score: Optional[float] = None
    created: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None

class IngestionJob(BaseModel):
    """Ingestion job tracking validation model."""

    pipeline: str
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

class TASignal(BaseModel):
    """Technical analysis signal snapshot validation model."""

    symbol: str
    exchange: str
    interval: str
    time: Optional[datetime] = None
    signal: Optional[str] = None
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    pattern: Optional[str] = None

class TASignalHistory(BaseModel):
    """Technical analysis signal history validation model."""

    time: datetime
    symbol: str
    exchange: str
    interval: str
    signal: Optional[str] = None
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    pattern: Optional[str] = None
