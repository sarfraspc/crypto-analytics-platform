CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMP,
    symbol TEXT,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);

CREATE TABLE IF NOT EXISTS whale_alerts (
    time TIMESTAMP,
    tx_hash TEXT PRIMARY KEY,
    from_address TEXT,
    to_address TEXT,
    amount NUMERIC,
    asset TEXT
);

INSERT INTO ohlcv (time, symbol, open, high, low, close, volume)
VALUES
  (NOW() - INTERVAL '5 minutes', 'BTC/USDT', 27000, 27100, 26900, 27050, 120.5),
  (NOW() - INTERVAL '4 minutes', 'BTC/USDT', 27050, 27200, 27000, 27150, 250.8);

INSERT INTO whale_alerts (time, tx_hash, from_address, to_address, amount, asset)
VALUES
  (NOW(), '0xabc123', 'wallet1', 'wallet2', 1000, 'ETH')
ON CONFLICT (tx_hash) DO NOTHING;
