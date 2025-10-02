import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from prophet import Prophet

from modules.forecasting.data.preprocess_coin import CoinPreprocessor

logger = logging.getLogger(__name__)


class ProphetModel:
    def __init__(
        self,
        symbol: str,
        model_dir: str = r"D:\python_projects\crypto-analytics-platform\src\modules\forecasting\models\saved\prophet",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_range: float = 0.8,
    ):
        self.symbol = symbol.upper()
        self.model = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"prophet_{self.symbol}.pkl"
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_range = changepoint_range

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ):
        if df.empty:
            raise ValueError(f"Empty DataFrame for {self.symbol}")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        df_prophet = df[[target_col]].copy()
        df_prophet = df_prophet.reset_index()
        date_col = [col for col in df_prophet.columns if col != target_col][0]
        df_prophet = df_prophet.rename(columns={date_col: "ds", target_col: "y"})

        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)

        logger.info(
            f"Training Prophet for {self.symbol} with changepoint_prior_scale={self.changepoint_prior_scale}"
        )
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_range=self.changepoint_range,
        )
        self.model.fit(df_prophet)
        logger.info(f"Finished training Prophet for {self.symbol}")

    def forecast(self, steps: int = 7, last_date: Optional[pd.Timestamp] = None, freq: str = 'D'):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future)

        forecast_future = forecast[["ds", "yhat"]].tail(steps).set_index("ds")["yhat"]

        if last_date is not None:
            start_date = last_date + pd.to_timedelta(1, unit=freq)
            forecast_future.index = pd.date_range(
                start=start_date,
                periods=steps,
                freq=freq,
            )
        return forecast_future

    def save(self):
        if self.model is None:
            raise RuntimeError("No trained model to save")
        joblib.dump(self.model, self.model_path)
        logger.info(f"Saved Prophet model for {self.symbol} -> {self.model_path}")

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded Prophet model for {self.symbol} from {self.model_path}")

def train_and_forecast(symbol: str, df: pd.DataFrame = None, exchange: str = 'binance', interval: str = '1h', forecast_steps: int = 7, retrain_if_exists: bool = False, ensure_features: bool = True):
    if df is None:
        coin_pre = CoinPreprocessor()
        if ensure_features:
            coin_pre.update_features(symbol, exchange=exchange, interval=interval, target_freq='H')  # Match freq
        df = coin_pre.load_features_series(symbol, exchange=exchange, interval=interval)
    
    model = ProphetModel(symbol)
    if model.model_path.exists() and not retrain_if_exists:
        model.load()
    else:
        model.train(df, target_col='close')
        model.save()
    
    forecast = model.forecast(steps=forecast_steps, last_date=df.index[-1], freq='H')
    return {'forecast': forecast, 'history': df['close']}
