\connect crypto_db

CREATE TABLE IF NOT EXISTS ohlcv_features (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    interval TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    returns DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    log_return DOUBLE PRECISION,
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
    time TIMESTAMPTZ NOT NULL,
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

