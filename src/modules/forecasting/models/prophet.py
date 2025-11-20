import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from prophet import Prophet
from typing import Optional, Dict

# --- CRITICAL: Suppress Prophet/Stan Logging ---
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_utils import load_scaler_with_meta, _scaler_path_for

logger = logging.getLogger(__name__)

class ProphetModel:
    def __init__(
        self,
        symbol: str,
        model_dir: str = "src/modules/forecasting/models/saved/prophet",
        # Tuned defaults for Crypto
        changepoint_prior_scale: float = 0.05, 
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_range: float = 0.9, # Look for trend changes in 90% of data
    ):
        self.symbol = symbol.upper()
        self.model = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"prophet_{self.symbol}.pkl"
        
        self.params = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "changepoint_range": changepoint_range
        }

    def train(self, df: pd.DataFrame, target_col: str = "close"):
        if df.empty:
            raise ValueError(f"Empty DataFrame for {self.symbol}")

        # Prophet requires columns ['ds', 'y']
        df_prophet = df.reset_index().rename(columns={'time': 'ds', 'index': 'ds', target_col: 'y'})
        df_prophet = df_prophet[['ds', 'y']].copy()
        
        # Ensure timezone-naive for Prophet
        if pd.api.types.is_datetime64_any_dtype(df_prophet['ds']):
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

        logger.info(f"Training Prophet for {self.symbol}...")
        
        self.model = Prophet(**self.params)
        # Add hourly seasonality for crypto
        self.model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
        
        self.model.fit(df_prophet)
        logger.info(f"Finished training Prophet for {self.symbol}")

    def forecast(self, steps: int = 24, freq: str = 'h') -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained")

        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future)
        
        # --- SAFETY FIX: Clamp negative predictions to 0 ---
        cols = ['yhat', 'yhat_lower', 'yhat_upper']
        for col in cols:
            forecast[col] = forecast[col].clip(lower=0)
        # ---------------------------------------------------

        # Return only future part
        return forecast.tail(steps)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def save(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Saved Prophet model -> {self.model_path}")

    def load(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded Prophet model -> {self.model_path}")
            return True
        return False

# --- Helper to get REAL prices before training ---
def _get_real_price_data(symbol: str, days: int = 60):
    """Loads data and Inverse Transforms it from 0-1 to Real Price"""
    coin_pre = CoinPreprocessor()
    start_date = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    
    # 1. Load Scaled Data
    df = coin_pre.load_features_series(symbol, interval="1h", start=start_date)
    
    # 2. Get Scaler
    scaler_path = _scaler_path_for(coin_pre.scaler_dir, symbol, None)
    scaler, cols = load_scaler_with_meta(scaler_path)
    
    # 3. Inverse Transform 'close'
    if scaler and 'close' in cols:
        close_idx = cols.index('close')
        matrix = np.zeros((len(df), len(cols)))
        # Fill known columns
        for i, c in enumerate(cols):
            if c in df.columns:
                matrix[:, i] = df[c].values
        
        real_close = scaler.inverse_transform(matrix)[:, close_idx]
        df['real_close'] = real_close
    else:
        df['real_close'] = df['close'] # Fallback
        
    return df[['real_close']].rename(columns={'real_close': 'close'})

# --- ADD THIS SECTION TO THE BOTTOM OF THE FILE ---

from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_utils import load_scaler_with_meta, _scaler_path_for

def train_and_forecast(
    model: ProphetModel, # Accept model instance
    df: pd.DataFrame = None, 
    exchange: str = 'binance', 
    interval: str = '1h', 
    forecast_steps: int = 24, 
    retrain_if_exists: bool = False
):
    """
    Helper function compatible with retrain_all.py structure.
    Handles data fetching (Inverse Scaling), Training, and Forecasting.
    Assumes model instance is already initialized.
    """
    symbol = model.symbol # Get symbol from model instance
    
    # 1. Load Data if not provided
    if df is None:
        coin_pre = CoinPreprocessor()
        # Look back 180 days for training context
        start_date = pd.Timestamp.utcnow() - pd.Timedelta(days=180)
        
        try:
            # Load features from DB
            df = coin_pre.load_features_series(symbol, exchange, interval, start=start_date)
        except ValueError:
            # If no data found, return empty result to prevent crash
            return {'forecast': None, 'history': None}

        # --- CRITICAL: Inverse Scale to get REAL prices for Prophet ---
        scaler_path = _scaler_path_for(coin_pre.scaler_dir, symbol, None)
        scaler, cols = load_scaler_with_meta(scaler_path)
        
        if scaler and 'close' in cols:
            close_idx = cols.index('close')
            matrix = np.zeros((len(df), len(cols)))
            for i, c in enumerate(cols):
                if c in df.columns:
                    matrix[:, i] = df[c].values
            
            # Replace scaled close with real close
            df['close'] = scaler.inverse_transform(matrix)[:, close_idx]
    
    if df is None or df.empty:
        return {'forecast': None, 'history': None}

    # 2. Train or Load based on retrain_if_exists
    if retrain_if_exists: # Case A: Training mode
        model.train(df, target_col='close')
        model.save()
    else: # Case B: Inference mode (load if exists, otherwise train and save)
        if model.model_path.exists():
            model.load()
        else:
            logger.warning(f"Model not found for {symbol}, training new one.")
            model.train(df, target_col='close')
            model.save()
    
    # 3. Forecast
    # freq='h' assumes hourly data
    forecast_df = model.forecast(steps=forecast_steps, freq='h')
    
    # 4. Format output for retrain_all.py
    # It expects a Pandas Series with DatetimeIndex
    forecast_series = forecast_df.set_index('ds')['yhat']
    
    return {'forecast': forecast_series, 'history': df['close']}