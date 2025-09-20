from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any, Dict

class OHLCV(BaseModel):
    time: datetime
    symbol: str
    interval: Optional[str] = "1m"
    exchange: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float
    raw: Optional[Dict[str, Any]] = None

class Trade(BaseModel):
    time: datetime
    exchange: str
    symbol: str
    trade_id: Optional[str]
    price: float
    amount: float
    side: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

class NewsArticle(BaseModel):
    id: str
    title: str
    source: Optional[str]
    url: Optional[str]
    published: Optional[datetime]
    text: Optional[str]
    raw: Optional[Dict[str, Any]] = None

class RedditPost(BaseModel):
    id: str
    subreddit: str
    author: Optional[str]
    title: str
    body: Optional[str]
    score: Optional[int]
    created: Optional[datetime]
    raw: Optional[Dict[str, Any]] = None

class WhaleAlert(BaseModel):
    time: datetime
    tx_hash: str
    chain: Optional[str]
    from_address: Optional[str]
    to_address: Optional[str]
    amount: float
    asset: Optional[str]
    raw: Optional[Dict[str, Any]] = None
