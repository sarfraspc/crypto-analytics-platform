-- Create role if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'crypto_user') THEN
      CREATE ROLE crypto_user LOGIN PASSWORD '123';
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

-- Tables (created only if missing)
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS whale_alerts (
    time TIMESTAMPTZ NOT NULL,
    tx_hash TEXT PRIMARY KEY,
    from_address TEXT,
    to_address TEXT,
    amount NUMERIC,
    asset TEXT
);
