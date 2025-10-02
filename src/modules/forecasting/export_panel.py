import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd

from core.database import get_timescale_engine
from modules.forecasting.data.preprocess_panel import PanelPreprocessor  

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_all_symbols(exchange: str = "binance", min_records: int = 100):
    engine = get_timescale_engine()
    q = """
        SELECT DISTINCT symbol 
        FROM ohlcv 
        WHERE exchange = %(exchange)s AND interval = '1h' 
        GROUP BY symbol 
        HAVING COUNT(*) > %(min_records)s;
    """
    df = pd.read_sql(q, engine, params={"exchange": exchange, "min_records": min_records})
    return df["symbol"].tolist()

def export_panel_data(
    symbols: Optional[List[str]] = None,
    exchange: str = "binance",
    interval: str = "1h",
    keep_cols: Optional[List[str]] = None,
    fit_global_scaler: bool = True,
    save_to_db: bool = False,  
    table_name: str = "ohlcv_features_panel",  
    min_records: int = 100,
):
    if symbols is None:
        symbols = get_all_symbols(exchange=exchange, min_records=min_records)
    
    logger.info(f"Processing {len(symbols)} symbols: {symbols[:5]}...")

    panel_pre = PanelPreprocessor()
    df_dict = {}
    failed_symbols = []
    
    for sym in symbols:
        try:
            df = panel_pre.coin_pre.load_features_series(sym, exchange=exchange, interval=interval)
            if not df.empty:
                df_dict[sym] = df
                logger.info(f"Loaded features for {sym}: {df.shape}")
            else:
                failed_symbols.append(sym)
        except Exception as e:
            logger.warning(f"Skipping {sym}: {e}")
            failed_symbols.append(sym)
    
    if not df_dict:
        raise ValueError("No valid symbol data loaded. Ensure features are updated via retrain_all")
    
    if failed_symbols:
        logger.warning(f"Failed to load {len(failed_symbols)} symbols: {failed_symbols}")

    panel_df, scaler = panel_pre.preprocess_panel(
        df_dict=df_dict,
        keep_cols=keep_cols,
        fit_global_scaler=fit_global_scaler,
        save_scaler=True,
    )
    
    logger.info(f"Final panel shape: {panel_df.shape}")
    logger.info(f"Date range: {panel_df['time'].min()} to {panel_df['time'].max()}")

    if save_to_db:
        panel_pre.save_panel_to_timescaledb(
            panel_df, 
            table_name=table_name,
            exchange=exchange,
            interval=interval
        )
        logger.info(f"Saved {len(panel_df)} rows to {table_name}")

    return panel_df, scaler

if __name__ == "__main__":
    panel_df, scaler = export_panel_data(
        keep_cols=["close", "returns", "sma_7", "ema_8", "volatility_7", "volume_zscore_30", "dayofweek"],
        fit_global_scaler=True,
        save_to_db=True,  
        table_name="ohlcv_features_panel",
    )
