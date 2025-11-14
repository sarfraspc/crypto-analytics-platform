from sqlalchemy import (
    Column, String, Integer, Numeric, JSON, TEXT, BIGINT, DateTime, Float
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Token(Base):
    __tablename__ = "tokens"

    symbol = Column(String, primary_key=True)
    coingecko_id = Column(String)
    name = Column(String)
    decimals = Column(Integer)
    token_metadata = Column('metadata', JSON) 
    
class OHLCV(Base):  
    __tablename__ = "ohlcv"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    exchange = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    open = Column(Float(precision=53))
    high = Column(Float(precision=53))
    low = Column(Float(precision=53))
    close = Column(Float(precision=53))
    volume = Column(Float(precision=53))
    raw = Column(JSON)

class Trade(Base):
    __tablename__ = "trades"

    time = Column(DateTime(timezone=True), primary_key=True)
    exchange = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    trade_id = Column(String, primary_key=True)
    price = Column(Float(precision=53))
    amount = Column(Float(precision=53))
    side = Column(String)
    raw = Column(JSON)

class WhaleAlert(Base):
    __tablename__ = "whale_alerts"

    time = Column(DateTime(timezone=True), primary_key=True)
    tx_hash = Column(String, primary_key=True)
    chain = Column(String)
    from_address = Column(String)
    to_address = Column(String)
    amount = Column(Numeric)
    usd_value = Column(Numeric)
    asset = Column(String, primary_key=True)
    raw = Column(JSON)

class OnchainMetric(Base):
    __tablename__ = "onchain_metrics"

    time = Column(DateTime(timezone=True), primary_key=True)
    chain = Column(String, primary_key=True)
    metric = Column(String, primary_key=True)
    value = Column(Numeric)
    raw = Column(JSON)

class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(String, primary_key=True)
    title = Column(String)
    source = Column(String)
    url = Column(String)
    published = Column(DateTime(timezone=True)) 
    text = Column(TEXT)
    raw = Column(JSON)

class RedditPost(Base):
    __tablename__ = "reddit_posts"

    id = Column(String, primary_key=True)
    subreddit = Column(String)
    author = Column(String)
    title = Column(String)
    body = Column(TEXT)
    score = Column(Integer)
    created = Column(DateTime(timezone=True)) 
    raw = Column(JSON)

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    pipeline = Column(String, primary_key=True)
    last_run = Column(DateTime(timezone=True)) 
    last_success = Column(DateTime(timezone=True))  
    details = Column(JSON)

class ChainState(Base):
    __tablename__ = "chain_state"

    chain = Column(String, primary_key=True)
    last_block = Column(BIGINT)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())  

class OHLCVFeature(Base):
    __tablename__ = "ohlcv_features"

    time = Column(DateTime(timezone=True), primary_key=True) 
    symbol = Column(String, primary_key=True)
    exchange = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    open = Column(Float(precision=53))
    high = Column(Float(precision=53))
    low = Column(Float(precision=53))
    close = Column(Float(precision=53))
    volume = Column(Float(precision=53))
    returns = Column(Float(precision=53))
    close_lag1 = Column(Float(precision=53))
    volatility = Column(Float(precision=53))
    log_return = Column(Float(precision=53))
    vol_7 = Column(Float(precision=53))
    vol_30 = Column(Float(precision=53))
    sma_7 = Column(Float(precision=53))
    sma_21 = Column(Float(precision=53))
    ema_8 = Column(Float(precision=53))
    ema_20 = Column(Float(precision=53))
    volume_pct_change = Column(Float(precision=53))
    volume_zscore_30 = Column(Float(precision=53))
    hour = Column(Integer)
    dayofweek = Column(Integer)
    month = Column(Integer)
    is_month_start = Column(Integer)

class OHLCVFeaturePanel(Base):
    __tablename__ = "ohlcv_features_panel"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    exchange = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    feature_name = Column(String, primary_key=True)
    feature_value = Column(Float(precision=53))

class TASignal(Base):
    __tablename__ = "ta_signals"

    symbol = Column(String, primary_key=True)
    exchange = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    time = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    signal = Column(String)                   
    rsi = Column(Float(precision=53))
    macd_hist = Column(Float(precision=53))
    pattern = Column(String)                   

class TASignalHistory(Base):
    __tablename__ = "ta_signals_history"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    exchange = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    signal = Column(String)
    rsi = Column(Float(precision=53))
    macd_hist = Column(Float(precision=53))
    pattern = Column(String)
