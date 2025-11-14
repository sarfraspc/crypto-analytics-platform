"""
Hybrid strategy: Composite signals + risk adjustment (Agent 1) with SMA/sentiment/on-chain fusion (Agent 2).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import talib  # For fallback SMA; prefer precomputed
from utils.cache import RedisCache  # Cache signals
cache = RedisCache(expire_seconds=300)  # 5min

def hybrid_signal(df: pd.DataFrame, forecast: Dict, sentiment: Dict, onchain: Dict, symbol: str = "BTC", query_hash: str = "", days: int = 30) -> Dict[str, Any]:
    # Use precomputed if available (from ohlcv_features)
    if 'sma_7' in df.columns and 'sma_21' in df.columns:
        tech = np.where(df['sma_7'] > df['sma_21'], 1, np.where(df['sma_7'] < df['sma_21'], -1, 0))
    else:  # Fallback TA-Lib
        df['sma_short'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_long'] = talib.SMA(df['close'], timeperiod=30)
        tech = np.where(df['sma_short'] > df['sma_long'], 1, np.where(df['sma_short'] < df['sma_long'], -1, 0))

    # Sentiment
    agg = sentiment.get('aggregated', {})
    sent_score = agg.get('bullish_score', 0.5) - agg.get('bearish_score', 0.5)
    sent_sig = 1 if sent_score > 0.2 else -1 if sent_score < -0.2 else 0

    # Forecast trend
    last_close = df['close'].iloc[-1]
    next_pred = forecast.get('predicted_close', [last_close])[-1]
    fc_sig = 1 if next_pred > last_close * 1.01 else -1 if next_pred < last_close * 0.99 else 0

    # On-chain (enhanced with corr)
    pressure = onchain.get('market_pressure_index', 0.5)
    onch_sig = 1 if pressure > 0.6 else -1 if pressure < 0.4 else 0
    corr = onchain.get('price_whale_corr_7d', 0.0)  # Default 0
    onch_sig += corr * 0.1  # Leverage full metrics

    # Composite (average scores)
    composite = (pd.Series(tech, index=df.index).iloc[-1] + sent_sig + fc_sig + onch_sig) / 4.0
    signal = "BUY" if composite > 0.3 else "SELL" if composite < -0.3 else "HOLD"
    pos_size = abs(composite) * 0.8  # Cap 80%

    # Risk adjust (vol/pressure)
    vol = df['close'].pct_change().std() if 'close' in df and len(df) > 1 else 0.02
    pos_size = risk_adjust_size(pos_size, vol, pressure)

    result = {
        'signal': signal,
        'position_size': pos_size,
        'composite_score': float(composite),
        'rationale': f"Hybrid: Tech={tech[-1]}, Sent={sent_sig}, FC={fc_sig}, OnCh={onch_sig} (Corr: {corr:.2f})",
        'vol_adjusted': vol > 0.05
    }

    # Cache
    key = f"strategy:{symbol}:{query_hash}:{days}"
    if cached := cache.get_json(key): return cached
    cache.set_json(key, result)
    return result

def risk_adjust_size(size: float, vol: float, pressure: float) -> float:
    adjusted = size * (1 - vol * 10) * (0.5 if pressure > 0.7 else 1.0)
    return max(0.1, min(1.0, adjusted))