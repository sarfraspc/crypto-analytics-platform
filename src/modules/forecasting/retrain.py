import logging
from core.database import get_timescale_engine
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.models.sarimax import train_and_forecast as sarimax_train_and_forecast
from modules.forecasting.models.prophet import train_and_forecast as prophet_train_and_forecast
import pandas as pd

logger = logging.getLogger(__name__)

def get_all_symbols(exchange: str = "binance"):
    engine = get_timescale_engine()
    q = f"SELECT DISTINCT symbol FROM ohlcv WHERE exchange = '{exchange}' AND interval = '1h' GROUP BY symbol HAVING COUNT(*) > 100;"
    df = pd.read_sql(q, engine)
    return df["symbol"].tolist()


def retrain_all(
    exchange: str = "binance", 
    interval: str = "1h",
    forecast_steps: int = 7,
    models: list = ["sarimax", "prophet"], 
    retrain_if_exists: bool = False
):
    symbols = get_all_symbols(exchange=exchange)
    print(f"Found {len(symbols)} data-rich symbols: {symbols}")

    results = {"sarimax": {}, "prophet": {}}
    
    for sym in symbols:
        try:
            preproc = CoinPreprocessor()
            preproc.update_features(symbol=sym, exchange=exchange, interval=interval)

            if "sarimax" in models:
                sar_res = sarimax_train_and_forecast(
                    symbol=sym,
                    exchange=exchange,
                    interval=interval,
                    forecast_steps=forecast_steps,
                    retrain_if_exists=retrain_if_exists,
                    ensure_features=False
                )
                results["sarimax"][sym] = sar_res["forecast"]
                print(f"Trained/Loaded SARIMAX for {sym}")

            if "prophet" in models:
                prop_res = prophet_train_and_forecast(
                    symbol=sym,
                    exchange=exchange,
                    interval=interval,
                    forecast_steps=forecast_steps,
                    retrain_if_exists=retrain_if_exists,
                    ensure_features=False
                )
                results["prophet"][sym] = prop_res["forecast"]
                print(f"Trained/Loaded Prophet for {sym}")
                
        except Exception as e:
            logger.exception(f"Failed for {sym}: {e}")
    
    return results

if __name__ == "__main__":
    retrain_all(exchange="binance", interval="1h", models=["sarimax", "prophet"])