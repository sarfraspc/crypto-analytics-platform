from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, List, Dict, Sequence
import logging

logger = logging.getLogger(__name__)


_FREQ_MAP = {
    "1m": "T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "h",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "D",
    "1w": "W",
}

_OHLCV_AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

def remove_duplicates(df: pd.DataFrame):
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df


def ensure_continuous_range(
    df: pd.DataFrame,
    freq_alias: str = "H",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    method: str = "ffill",
):
    if df.empty:
        raise ValueError("Input DataFrame to ensure_continuous_range cannot be empty.")
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()
    freq_alias = freq_alias.lower()
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


def clean_and_resample(
    df: pd.DataFrame,
    input_interval: str,
    target_freq: str,
    fill_method: str,
    drop_initial_na: bool = True,
):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df.index))

    freq_alias = _FREQ_MAP.get(input_interval, input_interval)
    df = remove_duplicates(df)
    df = ensure_continuous_range(
        df, freq_alias=freq_alias, start=df.index.min(), end=df.index.max(), method=fill_method
    )

    missing_cols = set(_OHLCV_AGG_RULES.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required columns for aggregation: {missing_cols}"
        )

    df_resampled = df.resample(target_freq).agg(_OHLCV_AGG_RULES)

    fill_methods = {
        "ffill": lambda d: d.ffill().bfill(),
        "bfill": lambda d: d.bfill().ffill(),
        "interpolate": lambda d: d.interpolate(method="time").ffill().bfill(),
        "drop": lambda d: d.dropna(),
    }
    if fill_method in fill_methods:
        df_resampled = fill_methods[fill_method](df_resampled)
    else:
        raise ValueError(f"Invalid fill_method: {fill_method}")

    if drop_initial_na:
        df_resampled = df_resampled.dropna(subset=["open", "close"])
    
    df_resampled["volume"] = df_resampled["volume"].replace(0, np.nan).ffill().fillna(0)
    return df_resampled


def add_features(
    df: pd.DataFrame,
    target_freq: str,
    feature_config: Optional[Dict] = None,
    DEFAULT_FEATURE_WINDOWS: dict = None
) -> pd.DataFrame:
    df = df.copy()
    
    freq_type = "D" if str(target_freq).upper().startswith("D") else "H"
    windows = DEFAULT_FEATURE_WINDOWS[freq_type]

    default_config = {
        'returns_type': 'log',
        'sma_windows': windows['sma'],
        'ema_windows': windows['ema'],
        'vol_windows': windows['vol'],
        'zscore_window': windows['z_score']
    }
    
    config = default_config
    if feature_config:
        config.update(feature_config)

    if config['returns_type'] == 'log':
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
    else: 
        df["returns"] = df["close"].pct_change()
    df["returns"] = df["returns"].fillna(0)

    for w in config['vol_windows']:
        df[f"volatility_{w}"] = df["returns"].rolling(window=w, min_periods=1).std().fillna(0)

    for w in config['sma_windows']:
        df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
    for w in config['ema_windows']:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    df["volume_pct_change"] = df["volume"].pct_change().fillna(0)
    z_window = config['zscore_window']
    roll = df["volume"].rolling(z_window, min_periods=1)
    df[f"volume_zscore_{z_window}"] = ((df["volume"] - roll.mean()) / roll.std().replace(0, np.nan)).fillna(0)
    
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["close_lag1"] = df["close"].shift(1).bfill()

    return df


def scale_features(
    df: pd.DataFrame,
    scaler_dir: Path,
    global_scaler_name: str,
    symbol: Optional[str],
    cols_to_scale: Sequence[str],
    fit: bool,
    save: bool,
    scope: str,
):
    
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]
    if not cols_to_scale:
        return df

    scaler, meta_cols = None, None
    path = _scaler_path_for(
        scaler_dir,
        symbol if scope == "per_symbol" else None,
        global_scaler_name if scope == "global" else None,
    )

    if path:
        scaler, meta_cols = load_scaler_with_meta(path)

    if meta_cols and set(meta_cols) != set(cols_to_scale):
        if fit:
            logger.warning("Meta mismatch; refitting scaler")
        else:
            raise ValueError(f"Scaler meta mismatch: expected {meta_cols}, got {cols_to_scale}")

    if fit:
        scaler = MinMaxScaler()
        scaler.fit(df[cols_to_scale].fillna(0))
        if save and path:
            save_scaler_with_meta(path, scaler, cols_to_scale)
            logger.info("Saved scaler for %s (scope: %s)", symbol or "global", scope)
        else:
            logger.info("Fitted scaler in-memory for %s", symbol or "global")
    
    if scaler:
        df_scaled = df.copy()
        if fit:
            transform_cols = cols_to_scale
        else:
            transform_cols = meta_cols or cols_to_scale
        df_scaled[transform_cols] = scaler.transform(df[transform_cols].fillna(0))
        return df_scaled
    
    return df


def _scaler_path_for(base_dir: Path, symbol: Optional[str], global_name: str = 'scaler_global.pkl'):
    base_dir = Path(base_dir)
    if symbol:
        safe = symbol.replace('/', '_').upper()
        return base_dir / f'scaler_{safe}.pkl'
    return base_dir / global_name




def save_scaler_with_meta(path: Path, scaler: MinMaxScaler, cols_order: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    meta = {'cols_order': cols_order, 'scaler_type': type(scaler).__name__, 'saved_at': pd.Timestamp.utcnow().isoformat()}
    path.with_suffix('.json').write_text(json.dumps(meta))




def load_scaler_with_meta(path: Path):
    meta_p = path.with_suffix('.json')
    if not path.exists() or not meta_p.exists():
        return None, None
    scaler = joblib.load(path)
    meta = json.loads(meta_p.read_text())
    return scaler, meta.get('cols_order')




def normalize_time(df, col="time"):
    if col not in df.columns:
        raise ValueError("normalize_time: column 'time' not found")
    df[col] = pd.to_datetime(df[col])
    if df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize("UTC")
    else:
        df[col] = df[col].dt.tz_convert("UTC")
    return df



def normalize_single_time(time_value):
    ts = pd.Timestamp(time_value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts