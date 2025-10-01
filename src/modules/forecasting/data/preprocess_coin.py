import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sqlalchemy import types as satypes
from sqlalchemy import text

from core.database import get_timescale_engine
from modules.forecasting.data.preprocess_utils import (
    normalize_time,
    normalize_single_time,
    clean_and_resample,
    add_features,
    scale_features,
    load_scaler_with_meta,
    _scaler_path_for,
)
from utils.cache import RedisCache

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_WINDOWS = {
    "D": {"sma": (7, 21), "ema": (8, 20), "vol": (7, 30), "z_score": 30},
    "H": {"sma": (24, 168), "ema": (24, 168), "vol": (24, 168), "z_score": 30},
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

    def get_coin_start(self, symbol: str, exchange: str = "binance", interval: str = "1h"):
        q = f"""
            SELECT MIN(time) AS start_time
            FROM {self.table}
            WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s;
        """
        df_start = pd.read_sql(q, self.engine, params={"symbol": symbol.upper(), "exchange": exchange, "interval": interval})
        start_time = df_start.iloc[0, 0]
        if pd.isna(start_time):
            raise ValueError(f"No OHLCV data found for {symbol}")
        return normalize_single_time(start_time)

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

        df = normalize_time(df, col="time")

        df = (
            df.set_index(pd.DatetimeIndex(df["time"]))
            .drop(columns=["time"])
            .sort_index()
        )

        coin_start = self.get_coin_start(base_symbol, exchange, interval)
        df = df[df.index >= coin_start]

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if self.cache:
            self.cache.set_dataframe(cache_key, df)
        return df

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
        fill_method: str = "ffill",
        drop_initial_na: bool = True,
        return_numpy: bool = False,
        feature_config: Optional[Dict] = None,
    ):
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        target_freq = target_freq or self.default_target_freq

        df = clean_and_resample(
            df, input_interval, target_freq, fill_method, drop_initial_na
        )
        
        df = add_features(df, target_freq, feature_config, DEFAULT_FEATURE_WINDOWS)

        if cols_to_scale is None:
            cols_to_scale = [
                "open", "high", "low", "close", "volume", "returns"
            ] + [c for c in df.columns if c.startswith('volatility_')]

        df_scaled = scale_features(
            df, self.scaler_dir, self.global_scaler_name, symbol, cols_to_scale, fit_scaler, save_scaler, scaler_scope
        )

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

        df_to_write['time'] = df_to_write['time'].dt.tz_localize(None)
        df_to_write.to_sql(
            table_name, con=self.engine, if_exists='append', index=False,
            method='multi', chunksize=5000,
            dtype={"time": satypes.DateTime(), "symbol": satypes.Text(),
                "exchange": satypes.Text(), "interval": satypes.Text()}
        )

    def update_features(self, symbol: str, exchange: str = "binance",
                        interval: str = "1h", target_freq: str = "D",
                        refit_scaler: bool = False):
        if self.cache:
            logger.info("Invalidating cache for symbol %s", symbol.upper())
            self.cache.delete_by_pattern(f"ohlcv:{symbol.upper()}:*")
            self.cache.delete_by_pattern(f"ohlcv_features:{symbol.upper()}:*")

        freq_type = "D" if str(target_freq).upper().startswith("D") else "H"
        windows = DEFAULT_FEATURE_WINDOWS[freq_type]
        all_windows = windows['sma'] + windows['ema'] + windows['vol'] + (windows.get('z_score', 30),)
        max_window = max(all_windows)

        if freq_type == 'H':
            overlap_days = (max_window // 24) + 2
        else:
            overlap_days = max_window + 1

        q = """
            SELECT MAX(time) FROM ohlcv_features
            WHERE symbol = %(symbol)s AND exchange = %(exchange)s AND interval = %(interval)s;
        """
        last_processed_result = pd.read_sql(q, self.engine, params={
            "symbol": symbol.upper(), "exchange": exchange, "interval": interval
        })
        last_processed = last_processed_result.iloc[0, 0] if not last_processed_result.empty and not pd.isna(last_processed_result.iloc[0, 0]) else None

        fit_scaler = True if last_processed is None else refit_scaler

        if last_processed is None:
            logger.info("No existing features, running full preprocessing for %s", symbol)
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
        else:
            logger.info("Incremental update from %s onwards for %s", last_processed, symbol)
            
            last_processed_ts = pd.Timestamp(last_processed)
            if last_processed_ts.tzinfo is None:
                last_processed_ts = last_processed_ts.tz_localize('UTC')

            start_date_for_load = last_processed_ts - pd.Timedelta(days=overlap_days)
            
            lookback_days = (pd.Timestamp.utcnow() - start_date_for_load).days
            
            df_raw = self.load_data(symbol, exchange, interval, lookback_days=lookback_days)

        if df_raw.empty:
            logger.info("No new rows to process for %s", symbol)
            return pd.DataFrame()

        if not fit_scaler:
            path = _scaler_path_for(self.scaler_dir, symbol, None)
            if path.exists():
                _, meta_cols = load_scaler_with_meta(path)
                if meta_cols:
                    expected_vol_cols = {c for c in meta_cols if 'volatility' in c}
                    current_vol_cols = {f"volatility_{w}" for w in windows['vol']}
                    if expected_vol_cols != current_vol_cols:
                        fit_scaler = True
                        logger.info("Scaler meta mismatch for %s; forcing refit", symbol)

        df_proc, _ = self.preprocess(
            df_raw, symbol=symbol, input_interval=interval, target_freq=target_freq,
            fit_scaler=fit_scaler, save_scaler=fit_scaler
        )
        df_proc['symbol'] = symbol.upper()
        df_proc['exchange'] = exchange
        df_proc['interval'] = interval

        if last_processed:
            last_processed_ts = pd.Timestamp(last_processed)
            if last_processed_ts.tzinfo is None:
                last_processed_ts = last_processed_ts.tz_localize('UTC')
            df_proc = df_proc[df_proc.index > last_processed_ts]

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
        cache_key = f"ohlcv_features:{symbol.upper()}:{exchange}:{interval}:{start.isoformat() if start else 'None'}:{end.isoformat() if end else 'None'}"
        if self.cache:
            cached_df = self.cache.get_dataframe(cache_key)
            if cached_df is not None:
                logger.info("Loaded %s from Redis cache", cache_key)
                return cached_df

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

        df = normalize_time(df, col="time")

        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time']).sort_index()

        if self.cache:
            self.cache.set_dataframe(cache_key, df)

        return df