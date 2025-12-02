"""
Portfolio backtesting module for crypto trading strategies.

Provides historical simulation of hybrid trading strategies using
real market data, sentiment analysis, and on-chain metrics.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from sqlalchemy import func, select

# MLflow Import with Safety
try:
    import mlflow
except ImportError:
    mlflow = None

# Core Configuration
from core.config import settings
from core.database import get_timescale_db

# Data Models
from data.storage.models import NewsArticle
from data.storage.models import OnchainMetric as OnchainMetricModel

# Strategy Logic
from modules.agent.strategy_utils import hybrid_signal

# Forecasting Modules (Real Model Logic)
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_utils import _scaler_path_for, load_scaler_with_meta
from modules.forecasting.models.prophet import ProphetModel

logger = logging.getLogger(__name__)


# --- Helper Functions ---

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculates Sharpe, Sortino, Max Drawdown, and Total Return."""
    if returns.empty or len(returns) < 2:
        return {
            "total_return": 0.0, "total_return_pct": 0.0, 
            "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0, 
            "trades_count": 0
        }
    
    cumprod = (1 + returns).cumprod()
    total_return = cumprod.iloc[-1] - 1
    
    # Annualized Metrics (Crypto markets active 365 days)
    ann_ret = returns.mean() * 365
    ann_vol = returns.std() * np.sqrt(365)
    
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    # Max Drawdown
    drawdowns = cumprod / cumprod.cummax() - 1
    max_dd = drawdowns.min()
    
    return {
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100),
        "trades_count": 0,
    }

def _mlflow_server_available() -> bool:
    """Checks if the MLflow tracking server is reachable."""
    if mlflow is None: 
        return False
    
    tracking_uri = getattr(settings, "MLFLOW_TRACKING_URI", None)
    if not tracking_uri:
        return False
        
    try:
        # If local file store, it's always available
        if tracking_uri.startswith("file:"): 
            return True
        # If HTTP server, ping it
        requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=1)
        return True
    except Exception:
        return False


# Main Backtester Class


class PortfolioBacktester:
    """
    Portfolio backtester for hybrid crypto trading strategies.

    Simulates trading using real historical data, AI forecasts,
    sentiment analysis, and on-chain metrics with MLflow tracking.
    """

    def __init__(self, initial_capital: float = 10000, experiment: str = "crypto_backtest_real", enable_mlflow: bool = True):
        """Initialize backtester with capital and MLflow settings."""
        self.initial_capital = initial_capital
        self.fee_rate: float = 0.0005  # 0.05% per trade
        self.experiment = experiment
        
        # MLflow Configuration: Only enable if requested AND server is running
        self.enable_mlflow = False
        if enable_mlflow and _mlflow_server_available():
            try:
                mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(self.experiment)
                self.enable_mlflow = True
                logger.info(f"MLflow logging enabled on {settings.MLFLOW_TRACKING_URI}")
            except Exception as e:
                logger.warning(f"Failed to setup MLflow: {e}")
                self.enable_mlflow = False

    async def _load_historical_sentiment(self, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, Any]:
        """Load historical sentiment from news articles using keyword scoring."""
        try:
            with get_timescale_db() as session:
                # Fetch headlines in this window
                stmt = select(NewsArticle.title).where(
                    NewsArticle.published >= start_ts,
                    NewsArticle.published <= end_ts
                ).limit(50) # Limit for performance
                
                titles = session.execute(stmt).scalars().all()
                
                if not titles:
                    return {"aggregated": {"bullish_score": 0.5, "bearish_score": 0.5}}

                # Simple Keyword Scoring (Deterministic & Fast)
                bullish_count = 0
                for t in titles:
                    title_lower = (t or "").lower()
                    if any(x in title_lower for x in ["soar", "high", "bull", "jump", "gain", "etf", "approval", "record"]):
                        bullish_count += 1
                    elif any(x in title_lower for x in ["crash", "drop", "ban", "sec", "bear", "low", "dump", "risk"]):
                        bullish_count -= 1
                
                # Normalize score: 0.5 baseline, +/- 0.05 per relevant keyword
                score = 0.5 + (bullish_count * 0.05)
                score = max(0.1, min(0.9, score)) # Clamp between 0.1 and 0.9
                
                return {"aggregated": {"bullish_score": score, "bearish_score": 1-score}}

        except Exception as e:
            logger.warning(f"Sentiment DB load failed: {e}")
            return {"aggregated": {"bullish_score": 0.5, "bearish_score": 0.5}}

    async def _load_historical_onchain(self, symbol: str, as_of: pd.Timestamp) -> Dict[str, Any]:
        """Load historical on-chain metrics from database."""
        try:
            chain = "bitcoin" if "BTC" in symbol.upper() else "ethereum"
            
            with get_timescale_db() as session:
                stmt = select(OnchainMetricModel).where(
                    OnchainMetricModel.chain == chain,
                    OnchainMetricModel.time <= as_of
                ).order_by(OnchainMetricModel.time.desc()).limit(1)
                
                metric = session.execute(stmt).scalar_one_or_none()
                
                if not metric or metric.value is None:
                    return {"market_pressure_index": 0.5, "price_whale_corr_7d": 0.0}
                
                val = float(metric.value)
                
                # Return formatted metrics
                return {"market_pressure_index": val, "price_whale_corr_7d": 0.0}

        except Exception as e:
            logger.warning(f"OnChain DB load failed: {e}")
            return {"market_pressure_index": 0.5, "price_whale_corr_7d": 0.0}

    async def _generate_model_forecasts(self, symbol: str, df_bt: pd.DataFrame) -> Dict[pd.Timestamp, float]:
        """Generate AI forecasts using saved Prophet model for backtest timestamps."""
        logger.info(f"Generating AI predictions using saved Prophet model for {symbol}...")
        
        model_wrapper = ProphetModel(symbol)
        
        # Load the pre-trained weights
        if not model_wrapper.load():
            logger.warning(f"No saved model found for {symbol}. Forecasting component will be neutral.")
            return {}

        # Create Future DataFrame for Prophet (The dates we want to test)
        future_dates = pd.DataFrame({'ds': df_bt.index})
        if future_dates['ds'].dt.tz is not None:
             future_dates['ds'] = future_dates['ds'].dt.tz_localize(None)

        # Generate Predictions
        try:
            forecast = model_wrapper.model.predict(future_dates)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}
        
        # Create Lookup Map (Timestamp -> Predicted Price)
        forecast_map = {}
        for _, row in forecast.iterrows():
            ts = row['ds'].tz_localize('UTC')
            forecast_map[ts] = row['yhat']
            
        return forecast_map

    async def run_hybrid_backtest(self, symbol: str, days: int = 60, cats: List[str] = None, rolling_window: int = 30) -> Dict[str, Any]:
        """
        Run hybrid backtest simulation.

        Loads real data from DB, inverse scales prices, generates AI predictions,
        loads sentiment/on-chain data, and simulates trading with fee accounting.
        """
        pre = CoinPreprocessor()
        exchange = getattr(settings, "MARKET_EXCHANGE_ID", "kraken")
        
        # 1. Load Data from DB (respect configured exchange)
        df_features = pre.load_features_series(symbol, exchange=exchange)
        
        if df_features.empty:
             logger.error("No data found in database for backtest.")
             return {"metrics": {}, "equity_curve": [], "trades": [], "signals": []}

        # 2. Inverse Scaling (CRITICAL)
        # We must convert 0-1 scaled values back to real prices for the strategy to work
        scaler_path = _scaler_path_for(pre.scaler_dir, symbol, None)
        scaler, cols = load_scaler_with_meta(scaler_path)
        
        if scaler and 'close' in cols:
             matrix = np.zeros((len(df_features), len(cols)))
             for i, c in enumerate(cols):
                if c in df_features.columns:
                    matrix[:, i] = df_features[c].values
             
             close_idx = cols.index('close')
             real_prices = scaler.inverse_transform(matrix)[:, close_idx]
             df_features['close'] = real_prices # Overwrite with Real Price
             
             # Re-calculate SMAs on Real Prices
             df_features['sma_7'] = df_features['close'].rolling(7).mean()
             df_features['sma_21'] = df_features['close'].rolling(21).mean()
        else:
             logger.warning("Scaler not found. Backtest might use scaled values (0-1).")

        # Filter Data for requested Period
        end_date = df_features.index[-1]
        start_date = end_date - pd.Timedelta(days=days)
        df_bt = df_features[start_date:end_date].copy()

        if len(df_bt) < rolling_window:
            logger.warning(f"Not enough data ({len(df_bt)} rows). Need > {rolling_window}.")
            return {"metrics": {}, "equity_curve": [], "trades": [], "signals": []}

        # 3. Get Real Context
        # A. AI Forecasts (Prophet)
        fc_map = await self._generate_model_forecasts(symbol, df_bt)
        # B. Sentiment & OnChain (DB Queries)
        sent_ctx = await self._load_historical_sentiment(symbol, start_date, end_date)
        onch_ctx = await self._load_historical_onchain(symbol, end_date)

        # 4. Initialize Portfolio
        portfolio_value = self.initial_capital
        holdings = 0.0
        positions = []
        signals_over_time = []
        trades = [] 
        trades_count = 0

        # 5. Trading Loop
        for i in range(rolling_window, len(df_bt)):
            window_df = df_bt.iloc[i-rolling_window:i]
            current_idx = df_bt.index[i]
            price = df_bt.iloc[i]['close']

            # Lookup Forecast for this specific hour
            pred_price = fc_map.get(current_idx, price) 
            fc_mock = {"predicted_close": [pred_price]}

            # Generate Hybrid Signal
            # This uses your strategy_utils.py logic to combine Tech + Sent + OnChain + Forecast
            signal_dict = hybrid_signal(window_df, fc_mock, sent_ctx, onch_ctx, symbol)
            signals_over_time.append(signal_dict)

            sig = signal_dict['signal']
            size = float(signal_dict.get('position_size', 0.0))
            size = max(0.0, min(size, 0.95)) # Risk management cap

            trade_executed = False
            
            # --- BUY LOGIC ---
            if sig == "BUY" and portfolio_value > 0 and size > 0:
                if holdings == 0: # Simple logic: Buy if not holding
                    trade_notional = portfolio_value * size
                    if trade_notional > 10: # Minimum trade
                        fee = trade_notional * self.fee_rate
                        units_to_buy = trade_notional / price
                        holdings += units_to_buy
                        portfolio_value -= (trade_notional + fee)
                        trades_count += 1
                        trade_executed = True
            
            # --- SELL LOGIC ---
            elif sig == "SELL" and holdings > 0:
                units_to_sell = holdings # Simple logic: Sell all
                trade_notional = units_to_sell * price
                if trade_notional > 0:
                    fee = trade_notional * self.fee_rate
                    portfolio_value += (trade_notional - fee)
                    holdings -= units_to_sell
                    trades_count += 1
                    trade_executed = True
            
            # Snapshot State
            positions.append({
                "date": current_idx,
                "price": price,
                "signal": sig,
                "model_pred": pred_price, # Save for plotting later
                "portfolio_value": portfolio_value + (holdings * price)
            })
            
            if trade_executed:
                trades.append({
                    "time": current_idx.isoformat(),
                    "side": sig,
                    "price": price,
                    "size": size,
                    "portfolio_value": portfolio_value + (holdings * price)
                })

        # 6. Final Metrics
        if not positions:
            return {"metrics": {}, "equity_curve": [], "trades": [], "signals": []}

        pos_df = pd.DataFrame(positions).set_index('date')
        returns = pos_df["portfolio_value"].pct_change().fillna(0)
        metrics = calculate_metrics(returns)
        metrics["trades_count"] = trades_count
        
        logger.info(f"Backtest Finished. ROI: {metrics['total_return_pct']:.2f}%, Trades: {trades_count}")

        # MLFlow logging (Safe Wrapper)
        if self.enable_mlflow:
            try:
                with mlflow.start_run(run_name=f"backtest_{symbol}_{days}d"):
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"MLflow logging failed (run continues): {e}")

        return {"metrics": metrics, "equity_curve": positions, "trades": trades, "signals": signals_over_time}
