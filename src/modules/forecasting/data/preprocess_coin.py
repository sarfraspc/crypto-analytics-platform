import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import types as satypes
from sqlalchemy import text

from core.database import get_timescale_engine
from modules.forecasting.data.scaler_utils import (
    _scaler_path_for,
    load_scaler_with_meta,
    save_scaler_with_meta,
)
from utils.cache import RedisCache

logger = logging.getLogger(__name__)


def normalize_time(df, col="time"):
    if col not in df.columns:
        raise ValueError("normalize_time: column 'time' not found")
    df[col] = pd.to_datetime(df[col])
    if df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize("UTC")
    else:
        df[col] = df[col].dt.tz_convert("UTC")
    df[col] = df[col].dt.tz_localize(None)
    return df


_FREQ_MAP = {
    "1m": "T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "D",
    "1w": "W",
}


class CoinPreprocessor:
    def __init__(
        self,
        table: str = "ohlcv",
        engine=None,
        scaler_dir: Union[str, Path] = "src/modules/forecasting/models/scalers",
        global_scaler_name: str = "scaler_global.pkl",
        default_target_freq: str = "D",
        use_cache: bool = True,
        cache_expire: int = 600,
    ):
        self.table = table
        self.engine = engine or get_timescale_engine()
        self.scaler_dir = Path(scaler_dir)
        self.global_scaler_name = global_scaler_name
        self.default_target_freq = default_target_freq
        self.cache = RedisCache(expire_seconds=cache_expire) if use_cache else None

    def load_data(
        self,
        symbol: str,
        exchange: str = "binance",
        interval: str = "1h",
        lookback_days: Optional[int] = None,
    ):
        base_symbol = (
            symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        )

        if lookback_days is None:
            span_query = f'''
                SELECT (NOW()::date - MIN(time)::date) AS days_span
                FROM {self.table}
                WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s;
            '''
            days_span_df = pd.read_sql(
                span_query,
                self.engine,
                params={
                    "symbol": base_symbol,
                    "exchange": exchange,
                    "interval": interval,
                },
            )
            lookback_days = int(days_span_df.iloc[0, 0]) + 1
            logger.info(
                "Using lookback_days=%s for %s/%s/%s",
                lookback_days,
                base_symbol,
                exchange,
                interval,
            )

        cache_key = f"ohlcv:{base_symbol}:{exchange}:{interval}:{lookback_days}"
        if self.cache:
            cached_df = self.cache.get_dataframe(cache_key)
            if cached_df is not None:
                logger.info("Loaded %s from Redis cache", cache_key)
                return cached_df

        start_ts = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
        q = f'''
            SELECT time, open, high, low, close, volume
            FROM {self.table}
            WHERE symbol = %(symbol)s AND exchange = %(exchange)s
            AND interval = %(interval)s AND time >= %(start)s
            ORDER BY time ASC;
        '''
        df = pd.read_sql(
            q,
            self.engine,
            params={
                "symbol": base_symbol,
                "exchange": exchange,
                "interval": interval,
                "start": start_ts,
            },
            parse_dates=["time"],
        )

        if df.empty:
            raise ValueError(f"No data found for {base_symbol}/{exchange}/{interval}")

        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")

        df = (
            df.set_index(pd.DatetimeIndex(df["time"]))
            .drop(columns=["time"])
            .sort_index()
        )
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if self.cache:
            self.cache.set_dataframe(cache_key, df)
        return df

    def remove_duplicates(self, df: pd.DataFrame):
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]
        return df

    def ensure_continuous_range(
        self,
        df: pd.DataFrame,
        freq_alias: str = "H",
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        method: str = "ffill",
    ):
        if start is None:
            start = df.index.min() if not df.empty else pd.Timestamp('2017-01-01')  # Crypto-safe default
        if end is None:
            end = df.index.max()
        freq_alias = freq_alias.lower()  # 'h' for deprecation
        full_idx = pd.date_range(start=start, end=end, freq=freq_alias, tz="UTC")
        df = df.reindex(full_idx)
        if method == "ffill":
            df = df.ffill().bfill()
        elif method == "bfill":
            df = df.bfill().ffill()
        elif method == "interpolate":
            df = df.interpolate(method="time").ffill().bfill()
        else:
            raise ValueError("method must be ffill|bfill|interpolate")
        return df

    def add_advanced_features(
        self,
        df: pd.DataFrame,
        sma_windows=None,
        ema_windows=None,
        vol_windows=None,
        target_freq: str = "D",
    ):
        df = df.copy()
        if sma_windows is None or ema_windows is None or vol_windows is None:
            if str(target_freq).upper().startswith("D"):
                sma_windows = sma_windows or (7, 21)
                ema_windows = ema_windows or (8, 20)
                vol_windows = vol_windows or (7, 30)  
            else: 
                sma_windows = sma_windows or (24, 168)
                ema_windows = ema_windows or (24, 168)
                vol_windows = vol_windows or (24, 168)  

        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        for w in vol_windows:
            df[f"vol_{w}"] = (
                df["log_return"].rolling(window=w, min_periods=1).std().fillna(0)
            )
        for w in sma_windows:
            df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
        for w in ema_windows:
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        df["volume_pct_change"] = df["volume"].pct_change().fillna(0)
        roll = df["volume"].rolling(30, min_periods=1)
        df["volume_zscore_30"] = (
            (df["volume"] - roll.mean()) / roll.std().replace(0, np.nan)
        ).fillna(0)
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_month_start"] = df.index.is_month_start.astype(int)
        return df

    def _freq_alias(self, interval: str) -> str:
        return _FREQ_MAP.get(interval, interval)

    def preprocess(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        input_interval: str = "1h",
        target_freq: Optional[str] = None,
        cols_to_scale: Optional[Sequence[str]] = None,
        fit_scaler: bool = False,
        save_scaler: bool = True,
        scaler_scope: str = "per_symbol",
        volatility_window_days: Optional[int] = None,
        fill_method: str = "ffill",
        drop_initial_na: bool = True,
        return_numpy: bool = False,
    ):
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        target_freq = target_freq or self.default_target_freq
        freq_alias = self._freq_alias(input_interval)

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df.index))
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        if freq_alias == "H":
            df = self.remove_duplicates(df)
            df = self.ensure_continuous_range(
                df, freq_alias="H", start=df.index.min(), end=df.index.max(), method=fill_method
            )

        if freq_alias == "H" and target_freq.upper() in ["D", "1D"]:
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            df_resampled = df.resample(target_freq).agg(agg)
        else:
            df_resampled = df.resample(target_freq).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            )

        if fill_method == "ffill":
            df_resampled = df_resampled.ffill().bfill()
        elif fill_method == "bfill":
            df_resampled = df_resampled.bfill().ffill()
        elif fill_method == "interpolate":
            df_resampled = df_resampled.interpolate(method="time").ffill().bfill()
        elif fill_method == "drop":
            df_resampled = df_resampled.dropna()
        else:
            raise ValueError(
                "fill_method must be one of: 'ffill', 'bfill', 'drop', 'interpolate'"
            )

        if drop_initial_na:
            df_resampled = df_resampled.dropna(subset=["open", "close"])

        df = df_resampled.copy()
        df["returns"] = df["close"].pct_change().fillna(0)
        df["close_lag1"] = df["close"].shift(1).bfill()
        volatility_window_days = volatility_window_days or 7
        df["volatility"] = (
            df["returns"].rolling(window=volatility_window_days, min_periods=1).std().fillna(0)
        )
        df["volume"] = df["volume"].replace(0, np.nan).ffill().fillna(0)

        df = self.add_advanced_features(df, target_freq=target_freq)

        default_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "volatility",
        ]
        cols_to_scale = (
            list(cols_to_scale) if cols_to_scale is not None else default_cols
        )
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]

        scaler = None
        meta_cols = None
        if scaler_scope == "global":
            path = _scaler_path_for(self.scaler_dir, None, self.global_scaler_name)
            scaler, meta_cols = load_scaler_with_meta(path)
        elif scaler_scope == "per_symbol" and symbol:
            path = _scaler_path_for(self.scaler_dir, symbol)
            scaler, meta_cols = load_scaler_with_meta(path)

        if fit_scaler:
            scaler = MinMaxScaler()
            scaler.fit(df[cols_to_scale].fillna(0))
            cols_order = cols_to_scale
            meta_cols = cols_order
            if save_scaler:
                path = _scaler_path_for(
                    self.scaler_dir,
                    symbol if scaler_scope == "per_symbol" else None,
                    self.global_scaler_name,
                )
                save_scaler_with_meta(path, scaler, cols_order)
                logger.info(
                    "Saved scaler+meta for %s (scope: %s)",
                    symbol or "global",
                    scaler_scope,
                )
            else:
                logger.info(
                    "Fitted scaler in-memory (no save) for EDA: %s",
                    symbol or "global",
                )

        if scaler is not None:
            df_scaled = df.copy()
            transform_cols = meta_cols or cols_to_scale
            df_scaled[transform_cols] = scaler.transform(df[transform_cols].fillna(0))
        else:
            df_scaled = df.copy()

        if return_numpy:
            numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
            return df_scaled, df_scaled[numeric_cols].values
        return df_scaled, None

    def split_by_dates(self, df: pd.DataFrame, train_end: str, val_end: str):
        train_end = pd.to_datetime(train_end).tz_localize('UTC') if pd.to_datetime(train_end).tzinfo is None else pd.to_datetime(train_end).tz_convert('UTC')
        val_end = pd.to_datetime(val_end).tz_localize('UTC') if pd.to_datetime(val_end).tzinfo is None else pd.to_datetime(val_end).tz_convert('UTC')
        train = df[df.index <= train_end].copy()
        val = df[(df.index > train_end) & (df.index <= val_end)].copy()
        test = df[df.index > val_end].copy()
        return train, val, test

    def create_sequences(self, df: pd.DataFrame, seq_length: int = 7, forecast_horizon: int = 7, feature_cols: Optional[Sequence[str]] = None, target_col: str = 'close', as_numpy: bool = True):
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in df.columns:
            raise ValueError('Target column not found')
        features = df[list(feature_cols)].values
        target = df[target_col].values
        X, y = [], []
        for i in range(seq_length, len(df) - forecast_horizon + 1):
            X.append(features[i - seq_length:i])
            y.append(target[i:i + forecast_horizon])
        X = np.asarray(X)
        y = np.asarray(y)
        return (X, y) if as_numpy else (X.tolist(), y.tolist())

    def save_to_timescaledb(self, df: pd.DataFrame, table_name: str):
        if isinstance(df.index, pd.DatetimeIndex):
            df_to_write = df.reset_index().rename(columns={'index': 'time'})
        else:
            df_to_write = df.copy()
            if 'time' not in df_to_write.columns:
                raise ValueError("DataFrame must have a 'time' column or a DatetimeIndex")

        df_to_write = normalize_time(df_to_write) 
        df_to_write.to_sql(
            table_name, con=self.engine, if_exists='append', index=False,
            method='multi', chunksize=5000,
            dtype={"time": satypes.DateTime(), "symbol": satypes.Text(),
                "exchange": satypes.Text(), "interval": satypes.Text()}
        )

    def update_features(self, symbol: str, exchange: str = "binance",
                        interval: str = "1h", target_freq: str = "D"):
        from sqlalchemy import text
        q = """
            SELECT MAX(time) FROM ohlcv_features
            WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s;
        """
        last_processed_result = pd.read_sql(q, self.engine, params={
            "symbol": symbol.upper(), "exchange": exchange, "interval": interval
        })
        last_processed = last_processed_result.iloc[0, 0] if not last_processed_result.empty and not pd.isna(last_processed_result.iloc[0, 0]) else None

        fit_scaler = False
        if last_processed is None:
            logger.info("No existing features, running full preprocessing for %s", symbol)
            # Check if any features exist to avoid re-full
            q_check = """
                SELECT COUNT(*) FROM ohlcv_features 
                WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s;
            """
            count = pd.read_sql(q_check, self.engine, params={
                "symbol": symbol.upper(), "exchange": exchange, "interval": interval
            }).iloc[0, 0]
            if count > 0:
                logger.warning("Features already exist for %s; skipping full to avoid duplicates", symbol)
                return pd.DataFrame()

            df_raw = self.load_data(symbol, exchange, interval, lookback_days=None)
            fit_scaler = True
        else:
            logger.info("Incremental update from %s onwards for %s", last_processed, symbol)
            q_new = f"""
                SELECT time, open, high, low, close, volume
                FROM {self.table}
                WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s
                AND time > %(last_processed)s
                ORDER BY time ASC;
            """
            df_raw = pd.read_sql(q_new, self.engine, params={
                "symbol": symbol.upper(), "exchange": exchange, "interval": interval,
                "last_processed": pd.Timestamp(last_processed)
            }, parse_dates=["time"])

        if df_raw.empty:
            logger.info("No new rows to process for %s", symbol)
            return pd.DataFrame()

        df_proc, _ = self.preprocess(
            df_raw, symbol=symbol, input_interval=interval, target_freq=target_freq,
            fit_scaler=fit_scaler, save_scaler=fit_scaler
        )
        df_proc['symbol'] = symbol.upper()
        df_proc['exchange'] = exchange
        df_proc['interval'] = interval

        df_proc = df_proc[~df_proc.index.duplicated(keep='last')]

        if not df_proc.empty:
            if last_processed is None:  
                delete_q = """
                    DELETE FROM ohlcv_features
                    WHERE symbol = :symbol AND exchange = :exchange AND interval = :interval;
                    """
                with self.engine.connect() as connection:
                    connection.execute(text(delete_q), {
                        "symbol": symbol.upper(), "exchange": exchange, "interval": interval
                    })

            self.save_to_timescaledb(df_proc, "ohlcv_features")
            logger.info("Upserted %d rows for %s", len(df_proc), symbol)
        else:
            logger.info("No rows to save for %s", symbol)

        return df_proc
    
    def load_features_series(self, symbol: str, exchange: str = 'binance', interval: str = '1h', start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None):
        params = {"symbol": symbol.upper(), "exchange": exchange, "interval": interval}
        q = f"SELECT * FROM ohlcv_features WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s"
        if start is not None:
            q += " AND time >= %(start)s"
            params['start'] = pd.to_datetime(start)
        if end is not None:
            q += " AND time <= %(end)s"
            params['end'] = pd.to_datetime(end)
        q += " ORDER BY time ASC;"


        df = pd.read_sql(q, self.engine, params=params, parse_dates=['time'])
        if df.empty:
            raise ValueError(f"No features found for {symbol}/{exchange}/{interval}")


        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        else:
            df['time'] = df['time'].dt.tz_convert('UTC')


        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time']).sort_index()
        return df