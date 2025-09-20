-- Create role if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'crypto_user') THEN
      CREATE ROLE crypto_user LOGIN PASSWORD ''; -- put orginal password only first time to create
   END IF;
END
$$;

-- Create database if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'crypto_db') THEN
      CREATE DATABASE crypto_db OWNER crypto_user;
   END IF;
END
$$;

\connect crypto_db

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Schema privileges
GRANT ALL ON SCHEMA public TO crypto_user;

-- OHLCV (time-series)
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

-- Orderbook snapshots (optional)
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
  time TIMESTAMPTZ NOT NULL,
  exchange TEXT NOT NULL,
  symbol TEXT NOT NULL,
  seq BIGINT,
  best_bid DOUBLE PRECISION,
  best_ask DOUBLE PRECISION,
  raw JSONB,
  PRIMARY KEY (time, exchange, symbol, seq)
);
SELECT create_hypertable('orderbook_snapshots', 'time', if_not_exists => TRUE);

-- Whale alerts (on-chain)
CREATE TABLE IF NOT EXISTS whale_alerts (
    time TIMESTAMPTZ NOT NULL,
    tx_hash TEXT NOT NULL,
    chain TEXT,
    from_address TEXT,
    to_address TEXT,
    amount NUMERIC,
    asset TEXT,
    raw JSONB,
    PRIMARY KEY (time, tx_hash)
);
SELECT create_hypertable('whale_alerts', 'time', if_not_exists => TRUE);

-- News articles (for RAG)
CREATE TABLE IF NOT EXISTS news_articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    url TEXT,
    published TIMESTAMPTZ,
    text TEXT,
    raw JSONB
);

-- Reddit posts
CREATE TABLE IF NOT EXISTS reddit_posts (
    id TEXT PRIMARY KEY,
    subreddit TEXT,
    author TEXT,
    title TEXT,
    body TEXT,
    score INT,
    created TIMESTAMPTZ,
    raw JSONB
);

-- Tokens metadata
CREATE TABLE IF NOT EXISTS tokens (
    symbol TEXT PRIMARY KEY,
    coingecko_id TEXT,
    name TEXT,
    decimals INT,
    metadata JSONB
);

-- Ingestion state & jobs
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  pipeline TEXT PRIMARY KEY,
  last_run TIMESTAMPTZ,
  last_success TIMESTAMPTZ,
  details JSONB
);

-- Chain state (saving last scanned block per chain)
CREATE TABLE IF NOT EXISTS chain_state (
  chain TEXT PRIMARY KEY,
  last_block BIGINT,
  last_updated TIMESTAMPTZ DEFAULT now()
);

