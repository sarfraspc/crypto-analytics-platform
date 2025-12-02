"""SARIMAX time series forecasting model for crypto prices."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_utils import _scaler_path_for, load_scaler_with_meta

logger = logging.getLogger(__name__)


class SarimaxModel:
    """SARIMAX model wrapper for crypto price forecasting on log returns."""
    def __init__(
        self,
        symbol: str,
        # Changed default order to (1,0,1) because we will manually calculate returns (d=1 implicit)
        order: tuple = (1, 0, 1), 
        seasonal_order: tuple = (0, 0, 0, 0),
        MODEL_DIR = Path("src/modules/forecasting/models/saved/sarimax"),
    ):
        self.symbol = symbol.upper()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.model_dir = Path(MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"sarimax_{self.symbol}.pkl"

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "log_return",
    ):
        """Train SARIMAX model on log returns."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Drop NaNs created by return calculation
        train_data = df[target_col].dropna()
        
        # Ensure we aren't passing infinite values
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(
            f"Training SARIMAX for {self.symbol} on {target_col} with order={self.order}"
        )
        
        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,  # CHANGED: Set to True for stability
            enforce_invertibility=True, # CHANGED: Set to True
            concentrate_scale=True      # OPTIONAL: Helps with convergence
        )
        self.model_fit = self.model.fit(disp=False)
        logger.info(f"Finished training SARIMAX for {self.symbol}")

    def forecast(self, steps: int = 7) -> pd.Series:
        """Forecast log returns for specified number of steps."""
        if self.model_fit is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        
        # Forecast returns
        forecast_returns = self.model_fit.forecast(steps=steps)
        return forecast_returns

    def save(self):
        """Save trained model to disk."""
        if self.model_fit is None:
            raise RuntimeError("No trained model to save")
        joblib.dump(self.model_fit, self.model_path)
        logger.info(f"Saved SARIMAX model for {self.symbol} -> {self.model_path}")

    def load(self):
        """Load trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model found at {self.model_path}")
        self.model_fit = joblib.load(self.model_path)
        logger.info(f"Loaded SARIMAX model for {self.symbol} from {self.model_path}")


def _inverse_transform_close(symbol: str, df: pd.DataFrame, preprocessor: CoinPreprocessor) -> pd.Series:
    """Helper to recover actual prices from the MinMax scaled DB data."""
    scaler_path = _scaler_path_for(preprocessor.scaler_dir, symbol, None)
    scaler, cols = load_scaler_with_meta(scaler_path)
    
    if scaler is None or 'close' not in cols:
        logger.warning(f"Scaler not found for {symbol}, using raw values (might be scaled)")
        return df['close']

    # Create a dummy matrix to inverse transform
    close_idx = cols.index('close')
    matrix = np.zeros((len(df), len(cols)))
    
    # Fill the columns we have
    for i, col_name in enumerate(cols):
        if col_name in df.columns:
            matrix[:, i] = df[col_name].values

    # Inverse transform
    inversed_matrix = scaler.inverse_transform(matrix)
    return pd.Series(inversed_matrix[:, close_idx], index=df.index)


def train_and_forecast(
    symbol: str,
    df: pd.DataFrame = None,
    exchange: str = 'kraken',
    interval: str = '1h',
    forecast_steps: int = 7,
    retrain_if_exists: bool = False,
    ensure_features: bool = True
):
    """Train SARIMAX model and generate price forecast from log returns."""
    coin_pre = CoinPreprocessor()
    
    # 1. Load Data
    if df is None:
        if ensure_features:
            coin_pre.update_features(symbol, exchange=exchange, interval=interval, target_freq='h') 
        df = coin_pre.load_features_series(symbol, exchange=exchange, interval=interval)
    
    # 2. Inverse Transform to get REAL prices
    # The DB has scaled data (0-1), we need real prices ($94,000) for accurate math
    raw_close = _inverse_transform_close(symbol, df, coin_pre)
    
    # 3. Calculate Log Returns (Stationary data for ARIMA)
    # Log returns are better than percentage returns for time-series additivity
    df['raw_close'] = raw_close
    df['log_return'] = np.log(df['raw_close'] / df['raw_close'].shift(1))
    df = df.dropna(subset=['log_return'])

    # 4. Train Model on Returns
    model = SarimaxModel(symbol, order=(1, 0, 1)) # d=0 because log_return is already differenced
    
    # Check if we should load or train
    if retrain_if_exists: # Case A: Training mode
        model.train(df, target_col='log_return')
        model.save()
    else: # Case B: Inference mode (load if exists, otherwise train and save)
        if model.model_path.exists():
            try:
                model.load()
            except Exception:
                logger.warning(f"Failed to load model for {symbol}, retraining...")
                model.train(df, target_col='log_return')
                model.save()
        else:
            logger.warning(f"Model not found for {symbol}, training new one.")
            model.train(df, target_col='log_return')
            model.save()
    
    # 5. Forecast Returns
    predicted_log_returns = model.forecast(steps=forecast_steps)
    
    # 6. Reconstruct Price Path
    # Price_t = Price_{t-1} * exp(return_t)
    last_real_price = df['raw_close'].iloc[-1]
    
    forecast_prices = []
    current_price = last_real_price
    
    for log_ret in predicted_log_returns:
        next_price = current_price * np.exp(log_ret)
        forecast_prices.append(next_price)
        current_price = next_price

    # 7. Create timestamps for forecast
    last_date = df.index[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.to_timedelta(1, unit='h'),
        periods=forecast_steps,
        freq='h'
    )
    
    forecast_series = pd.Series(forecast_prices, index=forecast_index)

    # Return the forecast series (Real Prices) and history (Real Prices)
    return {'forecast': forecast_series, 'history': df['raw_close']}