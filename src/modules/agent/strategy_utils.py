import pandas as pd
import numpy as np
from typing import Dict, Any

def sma_crossover(df: pd.DataFrame, short: int = 10, long: int = 30) -> pd.Series:
    df["short"] = df["close"].rolling(short).mean()
    df["long"] = df["close"].rolling(long).mean()
    signal = np.where(df["short"] > df["long"], 1, np.where(df["short"] < df["long"], -1, 0))
    return pd.Series(signal, index=df.index, name="signal")

def sentiment_signal(score: float) -> int:
    if score > 0.65: return 1
    if score < 0.35: return -1
    return 0

def onchain_signal(pressure: Any) -> int:
    if isinstance(pressure, (int, float)):
        if pressure > 0.6: return 1
        if pressure < 0.4: return -1
        return 0
    if isinstance(pressure, str):
        if pressure.lower() == "buy": return 1
        if "sell" in pressure.lower(): return -1
    return 0

def get_hybrid_signals(df: pd.DataFrame, forecast: Dict, sentiment: Dict, onchain: Dict) -> pd.Series:
    tech = sma_crossover(df)

    sent_sig = sentiment_signal(sentiment.get("bullish_score", 0.5))

    last_close = df["close"].iloc[-1]
    next_pred = forecast.get("predicted_close", [last_close])[0] if isinstance(forecast.get("predicted_close"), list) else forecast.get("predicted_close", last_close)
    forecast_trend = 1 if next_pred > last_close * 1.01 else -1 if next_pred < last_close * 0.99 else 0

    onch_sig = onchain_signal(onchain.get("market_pressure", "") or onchain.get("pressure_index", ""))

    combined = tech + pd.Series([sent_sig + forecast_trend + onch_sig] * len(df), index=df.index)
    final = np.where(combined >= 2, 1, np.where(combined <= -2, -1, 0))
    return pd.Series(final, index=df.index, name="hybrid_signal")