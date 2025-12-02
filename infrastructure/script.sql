-- METADATA_DB (reference + configs + MLflow)

-- Create metadata user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'metadata_user') THEN
      CREATE ROLE metadata_user LOGIN PASSWORD '123';
   END IF;
END
$$;

-- Create metadata_db if not exists 
SELECT 'CREATE DATABASE metadata_db OWNER metadata_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metadata_db')
\gexec

\connect metadata_db

-- Create dedicated schema for metadata
CREATE SCHEMA IF NOT EXISTS metadata AUTHORIZATION metadata_user;

-- Set search_path so queries don’t need schema prefix
ALTER ROLE metadata_user SET search_path = metadata, public;

-- Privileges
GRANT ALL PRIVILEGES ON DATABASE metadata_db TO metadata_user;
GRANT ALL PRIVILEGES ON SCHEMA metadata TO metadata_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA metadata
   GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO metadata_user;

-- Tokens metadata (reference for all modules)
CREATE TABLE IF NOT EXISTS tokens (
    symbol TEXT PRIMARY KEY,
    coingecko_id TEXT,
    name TEXT,
    decimals INT,
    metadata JSONB
);

ALTER TABLE IF EXISTS tokens OWNER TO metadata_user;
GRANT ALL PRIVILEGES ON TABLE tokens TO metadata_user;

-- CRYPTO_DB (timeseries + onchain + ingestion)

-- Create crypto user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'crypto_user') THEN
      CREATE ROLE crypto_user LOGIN PASSWORD '123';
   END IF;
END
$$;

-- Create crypto_db if not exists
SELECT 'CREATE DATABASE crypto_db OWNER crypto_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'crypto_db')
\gexec

\connect crypto_db

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create dedicated schema for crypto
CREATE SCHEMA IF NOT EXISTS crypto AUTHORIZATION crypto_user;

-- Set search_path so queries don’t need schema prefix
ALTER ROLE crypto_user SET search_path = crypto, public;

-- Privileges
GRANT ALL PRIVILEGES ON DATABASE crypto_db TO crypto_user;
GRANT ALL PRIVILEGES ON SCHEMA crypto TO crypto_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA crypto
   GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO crypto_user;

-- OHLCV (candlesticks)
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    interval TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    raw JSONB,
    PRIMARY KEY (time, symbol, exchange, interval)
);
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS ohlcv OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ohlcv TO crypto_user;

-- Trades
CREATE TABLE IF NOT EXISTS trades (
    time TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    trade_id TEXT,
    price DOUBLE PRECISION,
    amount DOUBLE PRECISION,
    side TEXT,
    raw JSONB,
    PRIMARY KEY (time, exchange, symbol, trade_id)
);
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS trades OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE trades TO crypto_user;

-- Whale alerts
CREATE TABLE IF NOT EXISTS whale_alerts (
    time TIMESTAMPTZ NOT NULL,
    tx_hash TEXT NOT NULL,
    chain TEXT,
    from_address TEXT,
    to_address TEXT,
    amount NUMERIC,
    usd_value NUMERIC,
    asset TEXT NOT NULL,
    raw JSONB,
    PRIMARY KEY (time, tx_hash, asset)
);
SELECT create_hypertable('whale_alerts', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS whale_alerts OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE whale_alerts TO crypto_user;

SELECT add_retention_policy('whale_alerts', INTERVAL '60 days');

-- On-chain metrics (aggregated flows/stats)
CREATE TABLE IF NOT EXISTS onchain_metrics (
    time TIMESTAMPTZ NOT NULL,
    chain TEXT NOT NULL,
    metric TEXT NOT NULL,
    value NUMERIC,
    raw JSONB,
    PRIMARY KEY (time, chain, metric)
);
SELECT create_hypertable('onchain_metrics', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS onchain_metrics OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE onchain_metrics TO crypto_user;

SELECT add_retention_policy('onchain_metrics', INTERVAL '90 days');

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    id TEXT NOT NULL,
    title TEXT,
    source TEXT,
    url TEXT,
    published TIMESTAMPTZ NOT NULL,
    text TEXT,
    raw JSONB,
    PRIMARY KEY (id, published)
);
SELECT create_hypertable('news_articles', 'published', if_not_exists => TRUE);

ALTER TABLE IF EXISTS news_articles OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE news_articles TO crypto_user;

SELECT add_retention_policy('news_articles', INTERVAL '90 days');

-- Reddit posts
CREATE TABLE IF NOT EXISTS reddit_posts (
    id TEXT NOT NULL,
    subreddit TEXT,
    author TEXT,
    title TEXT,
    body TEXT,
    score INT,
    created TIMESTAMPTZ NOT NULL,
    raw JSONB,
    PRIMARY KEY (id, created)
);
SELECT create_hypertable('reddit_posts', 'created', if_not_exists => TRUE);

ALTER TABLE IF EXISTS reddit_posts OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE reddit_posts TO crypto_user;

SELECT add_retention_policy('reddit_posts', INTERVAL '90 days');

-- Ingestion jobs tracker
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  pipeline TEXT PRIMARY KEY,
  last_run TIMESTAMPTZ,
  last_success TIMESTAMPTZ,
  details JSONB
);

ALTER TABLE IF EXISTS ingestion_jobs OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ingestion_jobs TO crypto_user;


CREATE TABLE IF NOT EXISTS ohlcv_features (
  time TIMESTAMP NOT NULL,
  symbol TEXT NOT NULL,
  exchange TEXT NOT NULL,
  interval TEXT NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  returns DOUBLE PRECISION,
  close_lag1 DOUBLE PRECISION,
  volatility DOUBLE PRECISION,
  log_return DOUBLE PRECISION,
  vol_7 DOUBLE PRECISION,
  vol_30 DOUBLE PRECISION,
  sma_7 DOUBLE PRECISION,
  sma_21 DOUBLE PRECISION,
  ema_8 DOUBLE PRECISION,
  ema_20 DOUBLE PRECISION,
  volume_pct_change DOUBLE PRECISION,
  volume_zscore_30 DOUBLE PRECISION,
  hour INT,
  dayofweek INT,
  month INT,
  is_month_start INT,
  PRIMARY KEY (time, symbol, exchange, interval)
);
SELECT create_hypertable('ohlcv_features', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS ohlcv_features OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ohlcv_features TO crypto_user;


CREATE TABLE IF NOT EXISTS ohlcv_features_panel (
  time TIMESTAMP NOT NULL,
  symbol TEXT NOT NULL,
  exchange TEXT NOT NULL,
  interval TEXT NOT NULL,
  feature_name TEXT NOT NULL,
  feature_value DOUBLE PRECISION,
  PRIMARY KEY (time, symbol, exchange, interval, feature_name)
);
SELECT create_hypertable('ohlcv_features_panel', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS ohlcv_features_panel OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ohlcv_features_panel TO crypto_user;


CREATE TABLE IF NOT EXISTS ta_signals (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    interval TEXT NOT NULL,
    signal TEXT,                       
    rsi DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    pattern TEXT,                     
    PRIMARY KEY (symbol, exchange, interval)
);

ALTER TABLE IF EXISTS ta_signals OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ta_signals TO crypto_user;


CREATE TABLE IF NOT EXISTS ta_signals_history (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    interval TEXT NOT NULL,
    signal TEXT,
    rsi DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    pattern TEXT,
    PRIMARY KEY (time, symbol, exchange, interval)
);

SELECT create_hypertable('ta_signals_history', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS ta_signals_history OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ta_signals_history TO crypto_user;

SELECT add_retention_policy('ta_signals_history', INTERVAL '180 days', if_not_exists => TRUE);



-- Forecast cache table (for fast dashboard loading)
CREATE TABLE IF NOT EXISTS forecast_cache (
    symbol TEXT PRIMARY KEY,
    model_used TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    horizon_hours INT,
    forecast_points JSONB,
    last_point JSONB,
    raw_text TEXT
);

ALTER TABLE IF EXISTS forecast_cache OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE forecast_cache TO crypto_user;

-- Sentiment cache table (for fast dashboard loading)
CREATE TABLE IF NOT EXISTS sentiment_cache (
    symbol TEXT PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    top_sentiment TEXT,
    top_confidence DOUBLE PRECISION,
    bullish_score DOUBLE PRECISION,
    bearish_score DOUBLE PRECISION,
    neutral_score DOUBLE PRECISION,
    sources JSONB,
    response TEXT
);

ALTER TABLE IF EXISTS sentiment_cache OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE sentiment_cache TO crypto_user;

-- Indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_forecast_cache_generated ON forecast_cache(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_cache_generated ON sentiment_cache(generated_at DESC);
