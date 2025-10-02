import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from modules.forecasting.data.preprocess_coin import CoinPreprocessor

logger = logging.getLogger(__name__)


class SarimaxModel:
    def __init__(
        self,
        symbol: str,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (0, 0, 0, 0),
        model_dir: str = r"D:\python_projects\crypto-analytics-platform\src\modules\forecasting\models\saved\sarimax",
    ):
        self.symbol = symbol.upper()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"sarimax_{self.symbol}.pkl"

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        logger.info(
            f"Training SARIMAX for {self.symbol} with order={self.order}, seasonal_order={self.seasonal_order}"
        )
        self.model = SARIMAX(
            df[target_col],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = self.model.fit(disp=False)
        logger.info(f"Finished training SARIMAX for {self.symbol}")

    def forecast(self, steps: int = 7, last_date: Optional[pd.Timestamp] = None, freq: str = 'D'):
        if self.model_fit is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        forecast = self.model_fit.forecast(steps=steps)

        if last_date is not None:
            start_date = last_date + pd.to_timedelta(1, unit=freq)
            forecast.index = pd.date_range(
                start=start_date,
                periods=steps,
                freq=freq,
            )
        return forecast

    def save(self):
        if self.model_fit is None:
            raise RuntimeError("No trained model to save")
        joblib.dump(self.model_fit, self.model_path)
        logger.info(f"Saved SARIMAX model for {self.symbol} -> {self.model_path}")

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model found at {self.model_path}")
        self.model_fit = joblib.load(self.model_path)
        logger.info(f"Loaded SARIMAX model for {self.symbol} from {self.model_path}")

def train_and_forecast(symbol: str, df: pd.DataFrame = None, exchange: str = 'binance', interval: str = '1h', forecast_steps: int = 7, retrain_if_exists: bool = False, ensure_features: bool = True):
    if df is None:
        coin_pre = CoinPreprocessor()
        if ensure_features:
            coin_pre.update_features(symbol, exchange=exchange, interval=interval, target_freq='H')  
        df = coin_pre.load_features_series(symbol, exchange=exchange, interval=interval)
    
    model = SarimaxModel(symbol)
    if model.model_path.exists() and not retrain_if_exists:
        model.load()
    else:
        model.train(df, target_col='close')
        model.save()
    
    forecast = model.forecast(steps=forecast_steps, last_date=df.index[-1], freq='H')
    return {'forecast': forecast, 'history': df['close']}
