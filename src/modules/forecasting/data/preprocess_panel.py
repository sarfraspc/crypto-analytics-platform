"""Multi-coin panel data preprocessing for cross-asset ML models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from modules.forecasting.data.preprocess_coin import CoinPreprocessor, normalize_time
from modules.forecasting.data.preprocess_utils import (
    _scaler_path_for,
    load_scaler_with_meta,
    save_scaler_with_meta,
)

logger = logging.getLogger(__name__)


class PanelPreprocessor:
    """Preprocessor for multi-coin panel data with global scaling."""
    def __init__(
        self,
        scaler_dir: Union[str, Path] = "src/modules/forecasting/models/scalers",
        global_scaler_name: str = "panel_global_scaler.pkl",
    ):
        self.scaler_dir = Path(scaler_dir)
        self.global_scaler_name = global_scaler_name
        self.coin_pre = CoinPreprocessor(scaler_dir=self.scaler_dir)
        self.engine = self.coin_pre.engine 

    def preprocess_panel(
        self,
        df_dict: Dict[str, pd.DataFrame],
        symbol_col: str = "symbol",
        keep_cols: Optional[List[str]] = None,
        fit_global_scaler: bool = False,
        save_scaler: bool = True,
        global_cols: Optional[List[str]] = None,
    ):
        """Combine multiple coin DataFrames into a unified panel with global scaling."""
        if not df_dict:
            return pd.DataFrame(), None

        global_start = min([df.index.min() for df in df_dict.values()])
        global_end = max([df.index.max() for df in df_dict.values()])

        for sym, df in df_dict.items():
            df_dict[sym] = df.reindex(
                pd.date_range(global_start, global_end, freq='h', tz='UTC')  
            ).ffill().bfill()

        panels = []
        for sym, df in df_dict.items():
            df2 = df.copy()
            if isinstance(df2.index, pd.DatetimeIndex):
                df2 = df2.reset_index().rename(columns={'index': 'time'})
            df2[symbol_col] = sym
            df2[symbol_col] = df2[symbol_col].astype(str)
            panels.append(df2)
        
        panel = pd.concat(panels, axis=0, ignore_index=True)
        
        panel["time"] = pd.to_datetime(panel["time"])
        if panel["time"].dt.tz is None:
            panel["time"] = panel["time"].dt.tz_localize("UTC")
        else:
            panel["time"] = panel["time"].dt.tz_convert("UTC")

        if keep_cols is not None:
            cols = ["time", symbol_col] + [
                c for c in keep_cols if c in panel.columns
            ]
            panel = panel[cols]

        scaler = None
        scaler_path = _scaler_path_for(
            self.scaler_dir, None, self.global_scaler_name
        )
        
        if fit_global_scaler:
            if global_cols is None:
                global_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
                global_cols = [c for c in global_cols if c not in ("time",)]
            
            if global_cols:
                # Replace infinity values with NaN, then fill with 0
                panel[global_cols] = panel[global_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                scaler = MinMaxScaler()
                scaler.fit(panel[global_cols])
                if save_scaler:
                    save_scaler_with_meta(scaler_path, scaler, global_cols)
                    logger.info("Saved global panel scaler -> %s", scaler_path)
                panel[global_cols] = scaler.transform(panel[global_cols])
        else:
            scaler, global_cols = load_scaler_with_meta(scaler_path)
            if scaler is not None and global_cols:
                panel[global_cols] = panel[global_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                panel[global_cols] = scaler.transform(panel[global_cols])

        return panel, scaler

    def save_panel_parquet(self, panel: pd.DataFrame, out_path: Union[str, Path]):
        """Save panel DataFrame to parquet file."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)
        logger.info(f"Saved panel data to {out_path}")

    def _convert_to_long_format(self, panel: pd.DataFrame, exchange: str, interval: str):
        """Convert wide panel to long format for TimescaleDB storage."""
        feature_cols = [col for col in panel.columns 
                       if col not in ['time', 'symbol', 'exchange', 'interval'] 
                       and pd.api.types.is_numeric_dtype(panel[col])]
        
        long_panel = panel.melt(
            id_vars=['time', 'symbol'],
            value_vars=feature_cols,
            var_name='feature_name',
            value_name='feature_value'
        )
        long_panel['exchange'] = exchange
        long_panel['interval'] = interval
        
        return long_panel

    def save_panel_to_timescaledb(
        self,
        panel: pd.DataFrame,
        table_name: str,
        exchange: str = "kraken",
        interval: str = "1h",
    ):
        """Save panel data to TimescaleDB in long format."""
        df_to_write = self._convert_to_long_format(panel, exchange, interval)
        df_to_write = normalize_time(df_to_write)
        
        df_to_write.to_sql(
            table_name,
            con=self.engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

    def update_panel(
        self,
        symbols: List[str],
        exchange: str = "kraken",
        interval: str = "1h",
    ):
        """Load features for symbols and create/update panel in database."""
        df_dict = {}
        successful_symbols = []
        
        for sym in symbols:
            try:
                logger.info(f"Loading features for {sym} for panel creation...")
                
                df_features = self.coin_pre.load_features_series(sym, exchange, interval)
                
                if df_features is not None and not df_features.empty:
                    df_dict[sym] = df_features
                    successful_symbols.append(sym)
                    logger.info(f"Loaded features for {sym}: {len(df_features)} rows")
                else:
                    logger.warning(f"No features found for {sym} in database")
                            
            except Exception as e:
                logger.warning(f"Failed to load features for {sym}: {e}")
                continue

        if not df_dict:
            logger.error("No features data available for any symbols in panel")
            return pd.DataFrame(), {}

        logger.info(f"Successfully loaded features for {len(successful_symbols)} symbols: {successful_symbols}")
        
        panel, _ = self.preprocess_panel(
            df_dict, 
            keep_cols=None, 
            fit_global_scaler=True  
        )
        
        if not panel.empty:
            self.save_panel_to_timescaledb(
                panel,
                "ohlcv_features_panel",
                exchange=exchange,
                interval=interval,
            )
            logger.info(f"Inserted {len(panel)} rows into ohlcv_features_panel")
        else:
            logger.error("Panel creation resulted in empty DataFrame")
        
        return panel, df_dict