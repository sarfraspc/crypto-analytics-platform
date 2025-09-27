import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from core.database import get_timescale_engine

logger = logging.getLogger(__name__)

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
    ):
        self.table = table
        self.engine = engine or get_timescale_engine()
        self.scaler_dir = Path(scaler_dir)
        self.global_scaler_path = self.scaler_dir / global_scaler_name
        self.default_target_freq = default_target_freq

    def load_data(self, symbol: str, exchange: str = "binance", interval: str = "1h", lookback_days: Optional[int] = None):
        base_symbol = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        if lookback_days is None:
            span_query = f"""
            SELECT (NOW()::date - MIN(time)::date) AS days_span
            FROM ohlcv
            WHERE symbol = '{base_symbol}' AND exchange = '{exchange}';
            """
            days_span_df = pd.read_sql(span_query, self.engine)
            lookback_days = int(days_span_df.iloc[0, 0]) + 1
            logger.info("Using lookback_days=%s for %s", lookback_days, base_symbol)

        q = f"""
        SELECT time, open, high, low, close, volume, interval, symbol, exchange
        FROM {self.table}
        WHERE symbol = '{base_symbol}' AND exchange = '{exchange}' AND time >= NOW() - INTERVAL '{lookback_days} days'
        ORDER BY time ASC, interval;
        """
        df = pd.read_sql(q, self.engine, parse_dates=["time"]) if self.engine else pd.DataFrame()
        if df.empty:
            raise ValueError(f"No data found for {base_symbol}/{exchange}")

        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        else:
            df['time'] = df['time'].dt.tz_convert('UTC')

        df = df.set_index(pd.DatetimeIndex(df['time']))
        df.index.name = 'time'
        df = df.drop(columns=['time'])
        df = df.sort_index()

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def _scaler_path_for(self, symbol: Optional[str]):
        if symbol:
            safe = symbol.replace('/', '_').upper()
            return self.scaler_dir / f"scaler_{safe}.pkl"
        return self.global_scaler_path

    def _meta_path_for(self, symbol: Optional[str]):
        return self._scaler_path_for(symbol).with_suffix('.json')

    def save_scaler_with_meta(self, scaler: MinMaxScaler, symbol: Optional[str], cols_order: List[str]):
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        p = self._scaler_path_for(symbol)
        joblib.dump(scaler, p)
        meta = {"cols_order": cols_order, "scaler_type": type(scaler).__name__, "symbol": symbol, "saved_at": pd.Timestamp.utcnow().isoformat()}
        self._meta_path_for(symbol).write_text(json.dumps(meta))
        logger.info("Saved scaler+meta: %s", p)

    def load_scaler_with_meta(self, symbol: Optional[str] = None):
        p = self._scaler_path_for(symbol)
        meta_p = self._meta_path_for(symbol)
        if not p.exists() or not meta_p.exists():
            return None, None
        scaler = joblib.load(p)
        meta = json.loads(meta_p.read_text())
        return scaler, meta.get('cols_order')

    def remove_duplicates(self, df: pd.DataFrame):
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        return df

    def ensure_continuous_range(self, df: pd.DataFrame, freq_alias: str = 'H', start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, method: str = 'ffill'):
        if start is None:
            start = df.index.min()
        if end is None:
            end = df.index.max()
        full_idx = pd.date_range(start=start, end=end, freq=freq_alias, tz='UTC')
        df = df.reindex(full_idx)
        if method == 'ffill':
            df = df.ffill().bfill()
        elif method == 'bfill':
            df = df.bfill().ffill()
        elif method == 'interpolate':
            df = df.interpolate(method='time').ffill().bfill()
        else:
            raise ValueError('method must be ffill|bfill|interpolate')
        return df

    def add_advanced_features(self, df: pd.DataFrame, sma_windows=(7, 21), ema_windows=(8, 20), vol_windows=(24, 168)):
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        for w in vol_windows:
            df[f'vol_{w}'] = df['log_return'].rolling(window=w, min_periods=1).std().fillna(0)
        for w in sma_windows:
            df[f'sma_{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
        for w in ema_windows:
            df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
        df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
        df['volume_zscore_30'] = (df['volume'] - df['volume'].rolling(30, min_periods=1).mean()) / (df['volume'].rolling(30, min_periods=1).std().replace(0, np.nan))
        df['volume_zscore_30'] = df['volume_zscore_30'].fillna(0)
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_start'] = df.index.is_month_start.astype(int)
        return df

    def _freq_alias(self, interval: str) -> str:
        return _FREQ_MAP.get(interval, interval)

    def preprocess(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        input_interval: str = "1h",
        target_freq: Optional[str] = None,  # FIXED: Proper typing
        cols_to_scale: Optional[Sequence[str]] = None,
        fit_scaler: bool = False,
        save_scaler: bool = True,  # NEW: Controls saving (False for EDA)
        scaler_scope: str = "per_symbol",
        volatility_window_days: Optional[int] = None,
        fill_method: str = "ffill",
        drop_initial_na: bool = True,
        return_numpy: bool = False,
    ):
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        target_freq = target_freq or self.default_target_freq
        freq_alias = self._freq_alias(input_interval)  # FIXED: Now uses method

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df.index))
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        if freq_alias == 'H':
            df = self.remove_duplicates(df)
            df = self.ensure_continuous_range(df, freq_alias='H', start=df.index.min(), end=df.index.max(), method=fill_method)

        if freq_alias == 'H' and target_freq.upper() in ['D', '1D']:
            agg = { 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum' }
            df_resampled = df.resample(target_freq).agg(agg)
        else:
            df_resampled = df.resample(target_freq).agg({ 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum' })

        if fill_method == 'ffill':
            df_resampled = df_resampled.ffill().bfill()
        elif fill_method == 'bfill':
            df_resampled = df_resampled.bfill().ffill()
        elif fill_method == 'interpolate':
            df_resampled = df_resampled.interpolate(method='time').ffill().bfill()
        elif fill_method == 'drop':
            df_resampled = df_resampled.dropna()
        else:
            raise ValueError("fill_method must be one of: 'ffill', 'bfill', 'drop', 'interpolate'")

        if drop_initial_na:
            df_resampled = df_resampled.dropna(subset=['open', 'close'])

        df = df_resampled.copy()
        df['returns'] = df['close'].pct_change().fillna(0)
        df['close_lag1'] = df['close'].shift(1).fillna(method='bfill')
        volatility_window_days = volatility_window_days or 7
        df['volatility'] = df['returns'].rolling(window=volatility_window_days, min_periods=1).std().fillna(0)
        df['volume'] = df['volume'].replace(0, np.nan).ffill().fillna(0)

        df = self.add_advanced_features(df)

        default_cols = ["open", "high", "low", "close", "volume", "returns", "volatility"]
        cols_to_scale = list(cols_to_scale) if cols_to_scale is not None else default_cols
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]

        scaler = None
        meta_cols = None
        if scaler_scope == 'global':
            scaler, meta_cols = self.load_scaler_with_meta(None)
        elif scaler_scope == 'per_symbol' and symbol:
            scaler, meta_cols = self.load_scaler_with_meta(symbol)

        if fit_scaler:  # FIXED: Simplified—only if explicitly True
            scaler = MinMaxScaler()
            scaler.fit(df[cols_to_scale])
            cols_order = cols_to_scale
            meta_cols = cols_order
            if save_scaler:  # NEW: Conditional save
                self.save_scaler_with_meta(scaler, symbol if scaler_scope == 'per_symbol' else None, cols_order)
                logger.info("Saved scaler+meta for %s (scope: %s)", symbol or "global", scaler_scope)
            else:
                logger.info("Fitted scaler in-memory (no save) for EDA: %s", symbol or "global")

        if scaler is not None:
            df_scaled = df.copy()
            df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
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

    def save_parquet(self, df: pd.DataFrame, out_path: Union[str, Path]):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)

    def save_to_timescaledb(self, df: pd.DataFrame, table_name: str):
        if isinstance(df.index, pd.DatetimeIndex):
            df_to_write = df.reset_index().rename(columns={'index': 'time'})
        else:
            df_to_write = df.copy()
            if 'time' not in df_to_write.columns:
                raise ValueError("DataFrame must have a 'time' column or a DatetimeIndex")
        df_to_write['time'] = pd.to_datetime(df_to_write['time']).dt.tz_convert('UTC').dt.tz_localize(None)
        df_to_write.to_sql(table_name, con=self.engine, if_exists='append', index=False, method='multi', chunksize=5000)