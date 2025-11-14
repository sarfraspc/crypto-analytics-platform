import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sqlalchemy import types as satypes
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from core.database import get_timescale_engine
from data.storage.models import OHLCV, OHLCVFeature
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
    "H": {"sma": (7, 21), "ema": (8, 20), "vol": (7, 30), "z_score": 30},
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
        cache_expire: int = 3600,  # Default for less volatile data
    ):
        self.table = table
        self.engine = engine or get_timescale_engine()
        logger.info(f"[PREPROCESSOR] Engine created with URL: {self.engine.url}")  # NEW: Log URL at init
        self.scaler_dir = Path(scaler_dir)
        self.global_scaler_name = global_scaler_name
        self.default_target_freq = default_target_freq
        self.cache = RedisCache(expire_seconds=cache_expire) if use_cache else None
        self.volatile_ttl = 300  # 5 minutes for volatile data

    def get_coin_start(self, symbol: str, exchange: str = "binance", interval: str = "1h"):
        Session = sessionmaker(bind=self.engine)
        with Session() as session:
            start_time = session.query(func.min(OHLCV.time)).filter(
                OHLCV.symbol == symbol.upper(),
                OHLCV.exchange == exchange,
                OHLCV.interval == interval
            ).scalar()

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

        Session = sessionmaker(bind=self.engine)
        with Session() as session:
            if lookback_days is None:
                days_span = session.query(func.max(OHLCV.time) - func.min(OHLCV.time)).filter(
                    OHLCV.symbol == base_symbol,
                    OHLCV.exchange == exchange,
                    OHLCV.interval == interval
                ).scalar()
                lookback_days = days_span.days + 1 if days_span else 1
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
            
            query = session.query(OHLCV.time, OHLCV.open, OHLCV.high, OHLCV.low, OHLCV.close, OHLCV.volume).filter(
                OHLCV.symbol == base_symbol,
                OHLCV.exchange == exchange,
                OHLCV.interval == interval,
                OHLCV.time >= start_ts
            ).order_by(OHLCV.time.asc())

            df = pd.read_sql(query.statement, self.engine, parse_dates=["time"])

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
            self.cache.set_dataframe(cache_key, df, expire_seconds=self.volatile_ttl)
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
            logger.info("Invalidating cache for symbol %s and global keys", symbol.upper())
            self.cache.delete_by_pattern(f"ohlcv:{symbol.upper()}:*")
            self.cache.delete_by_pattern(f"ohlcv_features:{symbol.upper()}:*")
            self.cache.delete_by_pattern("strategy:*")

        freq_type = "D" if str(target_freq).upper().startswith("D") else "H"
        windows = DEFAULT_FEATURE_WINDOWS[freq_type]
        all_windows = windows['sma'] + windows['ema'] + windows['vol'] + (windows.get('z_score', 30),)
        max_window = max(all_windows)

        if freq_type == 'H':
            overlap_days = (max_window // 24) + 2
        else:
            overlap_days = max_window + 1

        Session = sessionmaker(bind=self.engine)
        with Session() as session:
            last_processed = session.query(func.max(OHLCVFeature.time)).filter(
                OHLCVFeature.symbol == symbol.upper(),
                OHLCVFeature.exchange == exchange,
                OHLCVFeature.interval == interval
            ).scalar()

            fit_scaler = True if last_processed is None else refit_scaler

            if last_processed is None:
                logger.info("No existing features, running full preprocessing for %s", symbol)
                
                count = session.query(func.count(OHLCVFeature.time)).filter(
                    OHLCVFeature.symbol == symbol.upper(),
                    OHLCVFeature.exchange == exchange,
                    OHLCVFeature.interval == interval
                ).scalar()

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
                    session.query(OHLCVFeature).filter(
                        OHLCVFeature.symbol == symbol.upper(),
                        OHLCVFeature.exchange == exchange,
                        OHLCVFeature.interval == interval
                    ).delete(synchronize_session=False)
                    session.commit()

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

        Session = sessionmaker(bind=self.engine)
        with Session() as session:
            # NEW LOGS
            logger.info(f"[PREPROCESSOR] Engine URL in load: {self.engine.url}")  # Full URL (redact in prod)
            logger.info(f"[PREPROCESSOR] Query filters: symbol={symbol.upper()}, exchange={exchange}, interval={interval}")
            
            query = session.query(OHLCVFeature).filter(
                OHLCVFeature.symbol == symbol.upper(),
                OHLCVFeature.exchange == exchange,
                OHLCVFeature.interval == interval
            )
            if start is not None:
                query = query.filter(OHLCVFeature.time >= pd.to_datetime(start))
            if end is not None:
                query = query.filter(OHLCVFeature.time <= pd.to_datetime(end))
            query = query.order_by(OHLCVFeature.time.asc())

            # Compiled SQL log
            compiled_sql = str(query.statement.compile(dialect=self.engine.dialect, compile_kwargs={"literal_binds": True}))
            logger.info(f"[PREPROCESSOR] Compiled SQL: {compiled_sql[:200]}...")

            # Raw count
            raw_count = session.execute(select(func.count()).select_from(query.subquery())).scalar()
            logger.info(f"[PREPROCESSOR] Raw row count: {raw_count}")

            df = pd.read_sql(query.statement, self.engine, parse_dates=['time'])
            logger.info(f"[PREPROCESSOR] pd.read_sql rows: {len(df)}")

            if df.empty:
                logger.error(f"[PREPROCESSOR] Empty DF - check engine connect or filters. Raw count was {raw_count}")
                raise ValueError(f"No features found for {symbol}/{exchange}/{interval} (raw_count: {raw_count})")

        if df.empty:
            raise ValueError(f"No features found for {symbol}/{exchange}/{interval}")

        df = normalize_time(df, col="time")

        df = df.set_index(pd.DatetimeIndex(df['time'])).drop(columns=['time']).sort_index()

        if self.cache:
            self.cache.set_dataframe(cache_key, df, expire_seconds=self.volatile_ttl)

        return df