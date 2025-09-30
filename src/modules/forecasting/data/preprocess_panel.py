import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from modules.forecasting.data.preprocess_coin import CoinPreprocessor, normalize_time
from modules.forecasting.data.scaler_utils import (
    _scaler_path_for,
    load_scaler_with_meta,
    save_scaler_with_meta,
)

logger = logging.getLogger(__name__)


class PanelPreprocessor:
    def __init__(
        self,
        scaler_dir: Union[str, Path] = "src/modules/forecasting/models/scalers",
        global_scaler_name: str = "panel_global_scaler.pkl",
    ):
        self.scaler_dir = Path(scaler_dir)
        self.global_scaler_name = global_scaler_name
        self.coin_pre = CoinPreprocessor(scaler_dir=self.scaler_dir)

    def preprocess_panel(
        self,
        df_dict: Dict[str, pd.DataFrame],
        symbol_col: str = "symbol",
        keep_cols: Optional[List[str]] = None,
        fit_global_scaler: bool = False,
        save_scaler: bool = True,
        global_cols: Optional[List[str]] = None,
    ):
        panels = []
        for sym, df in df_dict.items():
            df2 = df.copy()
            if isinstance(df2.index, pd.DatetimeIndex):
                df2 = df2.reset_index()
            df2[symbol_col] = sym
            df2[symbol_col] = df2[symbol_col].astype(str)
            panels.append(df2)
        panel = pd.concat(panels, axis=0, ignore_index=True)
        if "time" not in panel.columns and "index" in panel.columns:
            panel = panel.rename(columns={"index": "time"})
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
            scaler = MinMaxScaler()
            scaler.fit(panel[global_cols].fillna(0))
            if save_scaler:
                save_scaler_with_meta(scaler_path, scaler, global_cols)
                logger.info("Saved global panel scaler -> %s", scaler_path)
            panel[global_cols] = scaler.transform(panel[global_cols].fillna(0))
        else:
            scaler, global_cols = load_scaler_with_meta(scaler_path)
            if scaler is not None and global_cols:
                panel[global_cols] = scaler.transform(panel[global_cols].fillna(0))

        return panel, scaler

    def save_panel_parquet(self, panel: pd.DataFrame, out_path: Union[str, Path]):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)

    def save_panel_to_timescaledb(
        self,
        panel: pd.DataFrame,
        table_name: str,
        engine,
        exchange: str = "binance",
        interval: str = "1h",
    ):
        df_to_write = panel.copy()
        df_to_write["exchange"] = exchange
        df_to_write["interval"] = interval
        df_to_write = normalize_time(df_to_write)
        df_to_write.to_sql(
            table_name,
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

    def update_panel(
        self,
        symbols: List[str],
        exchange: str = "binance",
        interval: str = "1h",
        target_freq: str = "D",
    ):
        df_dict = {}
        for sym in symbols:
            df_proc = self.coin_pre.update_features(sym, exchange, interval, target_freq)
            if df_proc is not None:
                df_dict[sym] = df_proc

        if not df_dict:
            logger.info("No updates for any symbols")
            return pd.DataFrame(), {}

        panel, _ = self.preprocess_panel(
            df_dict, keep_cols=None, fit_global_scaler=False
        )
        self.save_panel_to_timescaledb(
            panel,
            "ohlcv_features_panel",
            self.coin_pre.engine,
            exchange=exchange,
            interval=interval,
        )
        logger.info("Inserted %d rows into ohlcv_features_panel", len(panel))
        return panel, df_dict