-- Migration: Add forecast and sentiment cache tables for fast dashboard loading


-- Now connect to correct database
\connect crypto_db

-- Forecast cache table
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

-- Sentiment cache table
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
