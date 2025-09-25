from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any, Dict
from decimal import Decimal

class Token(BaseModel):
    symbol: str
    coingecko_id: Optional[str] = None
    name: Optional[str] = None
    decimals: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class OHLCV(BaseModel):
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
    time: datetime
    exchange: str
    symbol: str
    trade_id: Optional[str] = None
    price: Optional[float] = None
    amount: Optional[float] = None
    side: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

class WhaleAlert(BaseModel):
    time: datetime
    tx_hash: str
    chain: Optional[str] = None
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    amount: Optional[Decimal] = None
    asset: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

class OnchainMetric(BaseModel):
    time: datetime
    chain: str
    metric: str
    value: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = None

class NewsArticle(BaseModel):
    id: str
    title: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    published: Optional[datetime] = None
    text: Optional[str] = None
    sentiment_score: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

class RedditPost(BaseModel):
    id: str
    subreddit: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None
    score: Optional[int] = None
    sentiment_score: Optional[float] = None
    created: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None

class IngestionJob(BaseModel):
    pipeline: str
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

class ChainState(BaseModel):
    chain: str
    last_block: Optional[int] = None
    last_updated: Optional[datetime] = None