"""Technical analysis pattern detection using TA-Lib indicators."""

import json
import logging
from typing import Optional

import pandas as pd
import talib
from sqlalchemy import desc, select

from core.config import settings
from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.storage.crud import insert_ta_signals_history, upsert_ta_signals
from data.storage.models import OHLCV as OHLCVModel
from data.validation import TASignal, TASignalHistory
from utils.cache import RedisCache

setup_logging()
logger = logging.getLogger(__name__)

redis_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    expire_seconds=300
)


def load_recent_ohlcv(symbol: str, exchange: str | None = None, interval: str = "1h", lookback: int = 100):
    """Load recent OHLCV candles from database for TA computation."""
    exchange = exchange or getattr(settings, "MARKET_EXCHANGE_ID", "binance")
    with get_timescale_db() as db:
        try:
            query = select(
                OHLCVModel.time,
                OHLCVModel.open,
                OHLCVModel.high,
                OHLCVModel.low,
                OHLCVModel.close,
                OHLCVModel.volume
            ).where(
                OHLCVModel.symbol == symbol,
                OHLCVModel.exchange == exchange,
                OHLCVModel.interval == interval
            ).order_by(desc(OHLCVModel.time)).limit(lookback)
            result = db.execute(query).fetchall()
            if not result:
                logger.warning(f"No OHLCV data found for {symbol}:{exchange}:{interval}")
                return None

            df = pd.DataFrame([
                {
                    'time': row.time,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume
                }
                for row in result
            ])
            df = df.sort_values('time').reset_index(drop=True)  
            df['time'] = pd.to_datetime(df['time'])
            logger.info(f"Loaded {len(df)} candles for {symbol}:{exchange}:{interval}")
            return df
        except Exception as e:
            logger.error(f"Error loading OHLCV for {symbol}: {e}")
            return None


def compute_ta_indicators(df: pd.DataFrame):
    """Compute RSI, MACD, Bollinger Bands, and other TA indicators."""
    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        volume = df['volume'].values.astype(float) 
        df['sma_7'] = talib.SMA(close, timeperiod=7)
        df['sma_21'] = talib.SMA(close, timeperiod=21)
        df['ema_8'] = talib.EMA(close, timeperiod=8)
        df['ema_20'] = talib.EMA(close, timeperiod=20)

        df['rsi'] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        df['obv'] = talib.OBV(close, volume)
        df['volume_pct_change'] = df['volume'].pct_change() * 100
        df['volume_pct_change'] = df['volume_pct_change'].fillna(0)

        logger.info("Computed TA indicators successfully")
        return df
    except Exception as e:
        logger.error(f"Error computing TA indicators: {e}")
        raise


def detect_candlestick_patterns(df: pd.DataFrame):
    """Detect candlestick patterns using TA-Lib pattern recognition."""
    try:
        open_prices = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        df['cdl_engulfing'] = talib.CDLENGULFING(open_prices, high, low, close)
        df['cdl_harami'] = talib.CDLHARAMI(open_prices, high, low, close)
        df['cdl_hammer'] = talib.CDLHAMMER(open_prices, high, low, close)
        df['cdl_shootingstar'] = talib.CDLSHOOTINGSTAR(open_prices, high, low, close)
        df['cdl_doji'] = talib.CDLDOJI(open_prices, high, low, close)
        df['cdl_invertedhammer'] = talib.CDLINVERTEDHAMMER(open_prices, high, low, close)
        df['cdl_spinningtop'] = talib.CDLSPINNINGTOP(open_prices, high, low, close)
        df['cdl_marubozu'] = talib.CDLMARUBOZU(open_prices, high, low, close)

        logger.info("Detected candlestick patterns successfully")
        return df
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        raise


def _generate_signal(rsi: float, macd_hist: float, pattern: Optional[str], pattern_direction: Optional[str]):
    """Generate trading signal from RSI, MACD, and candlestick patterns."""
    signal = "neutral"
    explanation_parts = []

    if rsi < 30:
        signal = "bullish"
        explanation_parts.append("RSI below 30 indicates oversold")
    elif rsi > 70:
        signal = "bearish"
        explanation_parts.append("RSI above 70 indicates overbought")

    if macd_hist > 0:
        if signal == "neutral":
            signal = "bullish"
        explanation_parts.append("MACD histogram > 0 suggests bullish bias")
    elif macd_hist < 0:
        if signal == "neutral":
            signal = "bearish"
        explanation_parts.append("MACD histogram < 0 suggests bearish bias")

    if pattern is not None and pattern_direction:
        signal = pattern_direction
        explanation_parts.append(f"{pattern} pattern confirms reversal")

    explanation = "; ".join(explanation_parts)
    if not explanation:
        explanation = "No strong signals detected"

    return signal, explanation


def generate_ta_signal(symbol: str, exchange: str | None = None, interval: str = "1h", use_cache: bool = True):
    """Generate complete TA signal with indicators, patterns, and confidence."""
    exchange = exchange or getattr(settings, "MARKET_EXCHANGE_ID", "binance")
    key = f"ta_signal:{symbol}:{exchange}:{interval}"
    if use_cache:
        cached = redis_cache.get_json(key)
        if cached:
            logger.info(f"Returning cached TA signal for {symbol}")
            return cached

    lookback = 500 if interval == "1d" else 100
    source_interval = "1h" if interval == "1d" else interval
    df = load_recent_ohlcv(symbol, exchange, source_interval, lookback=lookback)
    if df is None or len(df) < 30:
        logger.warning(f"Insufficient data for {symbol} ({len(df) if df is not None else 0} candles)")
        return None

    if interval == "1d":
        df = (
            df.set_index("time")
            .resample("1D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )

    try:
        df = compute_ta_indicators(df)
        df = detect_candlestick_patterns(df)

        last_row = df.iloc[-1]
        current_time = last_row['time']
        rsi = last_row['rsi']
        macd_hist = last_row['macd_hist']

        pattern: Optional[str] = None
        pattern_direction: Optional[str] = None
        recent_candles = df.iloc[-3:]
        for idx in range(len(recent_candles) - 1, -1, -1):
            row = recent_candles.iloc[idx]
            if not pd.isna(row['cdl_engulfing']) and row['cdl_engulfing'] != 0:
                pattern = "bullish_engulfing" if row['cdl_engulfing'] > 0 else "bearish_engulfing"
                pattern_direction = "bullish" if row['cdl_engulfing'] > 0 else "bearish"
                break
            if not pd.isna(row['cdl_harami']) and row['cdl_harami'] != 0:
                pattern = "bullish_harami" if row['cdl_harami'] > 0 else "bearish_harami"
                pattern_direction = "bullish" if row['cdl_harami'] > 0 else "bearish"
                break
            if not pd.isna(row['cdl_hammer']) and row['cdl_hammer'] != 0:
                pattern = "hammer"
                pattern_direction = "bullish"
                break
            if not pd.isna(row['cdl_shootingstar']) and row['cdl_shootingstar'] != 0:
                pattern = "shooting_star"
                pattern_direction = "bearish"
                break
            if not pd.isna(row['cdl_invertedhammer']) and row['cdl_invertedhammer'] != 0:
                pattern = "inverted_hammer"
                pattern_direction = "bullish"
                break
            if not pd.isna(row['cdl_doji']) and row['cdl_doji'] != 0:
                pattern = "doji"
                pattern_direction = None
                break
            if not pd.isna(row['cdl_spinningtop']) and row['cdl_spinningtop'] != 0:
                pattern = "spinning_top"
                pattern_direction = None
                break
            if not pd.isna(row['cdl_marubozu']) and abs(row['cdl_marubozu']) == 100:
                pattern = "marubozu"
                pattern_direction = "bullish" if row['cdl_marubozu'] > 0 else "bearish"
                break

        signal, explanation = _generate_signal(rsi, macd_hist, pattern, pattern_direction)

        confidence = 0.5
        if macd_hist is not None:
            abs_macd = abs(macd_hist)
            if abs_macd > 1:
                confidence += 0.3
            elif abs_macd > 0.01:
                confidence += 0.15
        if rsi is not None and (rsi < 30 or rsi > 70):
            confidence += 0.2
        if pattern is not None:
            confidence += 0.25
        confidence = min(1.0, confidence)

        result = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "timestamp": current_time.isoformat(),
            "rsi": float(rsi) if not pd.isna(rsi) else None,
            "macd_hist": float(macd_hist) if not pd.isna(macd_hist) else None,
            "pattern": pattern,
            "signal": signal,
            "explanation": explanation,
            "confidence": confidence,
        }

        ta_signal = TASignal(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            time=current_time,
            signal=signal,
            rsi=result["rsi"],
            macd_hist=result["macd_hist"],
            pattern=pattern
        )
        ta_signal_history = TASignalHistory(
            time=current_time,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            signal=signal,
            rsi=result["rsi"],
            macd_hist=result["macd_hist"],
            pattern=pattern
        )

        with get_timescale_db() as db:
            upsert_ta_signals(db, [ta_signal])
            insert_ta_signals_history(db, [ta_signal_history])

        redis_cache.set_json(key, result)
        logger.info(f"Generated and cached TA signal for {symbol}: {signal}")

        return result
    except Exception as e:
        logger.error(f"Error generating TA signal for {symbol}: {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate TA signals manually")
    parser.add_argument("--symbol", required=True, help="Symbol e.g., BTCUSDT")
    parser.add_argument("--exchange", default="binance", help="Exchange")
    parser.add_argument("--interval", default="1h", help="Interval")
    parser.add_argument("--no-cache", action="store_true", help="Force recompute without using cache")
    args = parser.parse_args()

    result = generate_ta_signal(args.symbol, args.exchange, args.interval, use_cache=not args.no_cache)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Failed to generate signal")
