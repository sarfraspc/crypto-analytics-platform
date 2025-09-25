-- METADATA_DB (reference + configs + MLflow)

-- Create metadata user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'metadata_user') THEN
      CREATE ROLE metadata_user LOGIN PASSWORD ''; -- password here
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
      CREATE ROLE crypto_user LOGIN PASSWORD ''; -- password here
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
    asset TEXT,
    raw JSONB,
    PRIMARY KEY (time, tx_hash)
);
SELECT create_hypertable('whale_alerts', 'time', if_not_exists => TRUE);

ALTER TABLE IF EXISTS whale_alerts OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE whale_alerts TO crypto_user;

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

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    url TEXT,
    published TIMESTAMPTZ,
    text TEXT,
    raw JSONB
);

ALTER TABLE IF EXISTS news_articles OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE news_articles TO crypto_user;

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

ALTER TABLE IF EXISTS reddit_posts OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE reddit_posts TO crypto_user;

-- Ingestion jobs tracker
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  pipeline TEXT PRIMARY KEY,
  last_run TIMESTAMPTZ,
  last_success TIMESTAMPTZ,
  details JSONB
);

ALTER TABLE IF EXISTS ingestion_jobs OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE ingestion_jobs TO crypto_user;

-- Chain state (last scanned block per chain)
CREATE TABLE IF NOT EXISTS chain_state (
  chain TEXT PRIMARY KEY,
  last_block BIGINT,
  last_updated TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE IF EXISTS chain_state OWNER TO crypto_user;
GRANT ALL PRIVILEGES ON TABLE chain_state TO crypto_user;
