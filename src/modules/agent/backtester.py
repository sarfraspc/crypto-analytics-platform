"""
Merged backtester: Agent 1 simulation/MLflow + Agent 2 hybrid signals/metrics (Sharpe/Sortino).
"""

import logging
import mlflow
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List
from sqlalchemy import select, func
from core.database import get_timescale_db
from modules.sentiment.rag.embedder import Embedder  # For historical sentiment
from data.storage.models import TASignalHistory  # Historical TA
from data.validation import OnchainMetric  # For corr
from modules.sentiment.models.sentiment_infer import analyze_sentiment_batch  # Threaded
from modules.agent.strategy_utils import hybrid_signal  # For historical mock

logger = logging.getLogger(__name__)

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns.empty or len(returns) < 2: return {"total_return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0, "trades_count": 0}
    cumprod = (1 + returns).cumprod()
    total_return = cumprod.iloc[-1] - 1
    ann_ret = returns.mean() * 365
    ann_vol = returns.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = returns[returns < 0]
    sortino = ann_ret / (downside.std() * np.sqrt(365)) if len(downside) > 0 and downside.std() > 0 else 0.0
    drawdowns = cumprod / cumprod.cummax() - 1
    max_dd = drawdowns.min()
    return {
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "trades_count": int((returns.diff() != 0).sum())
    }

class PortfolioBacktester:
    def __init__(self, initial_capital: float = 10000, experiment: str = "crypto_backtest_v2"):
        self.initial_capital = initial_capital
        mlflow.set_experiment(experiment)

    async def _historical_mock(self, symbol: str, df: pd.DataFrame, cats: List[str] = None) -> tuple:
        """
        Generates aligned mock data for a given historical window to avoid lookahead bias.
        """
        # Aligned sentiment mock
        texts = [f"{symbol} on {date.strftime('%Y-%m-%d')}: neutral update" for date in df.index]
        sent_mock_scores = np.zeros(len(df))
        if texts:
            sent_results = await asyncio.to_thread(analyze_sentiment_batch, texts)
            sent_mock_scores = np.array([r.get('bullish_score', 0.5) - r.get('bearish_score', 0.5) for r in sent_results])
        
        # Use the mean sentiment over the window as the signal
        mean_sentiment_score = np.mean(sent_mock_scores)
        sent_mock = {"aggregated": {"bullish_score": 0.5 + mean_sentiment_score, "bearish_score": 0.5 - mean_sentiment_score}}

        # Persistence forecast (predicts the last known price) to avoid lookahead
        fc_mock = {"predicted_close": [df['close'].iloc[-1]]}

        # Simplified on-chain mock (randomized)
        onch_mock = {
            "market_pressure_index": np.random.uniform(0.3, 0.7),
            "price_whale_corr_7d": np.random.uniform(-0.2, 0.2)
        }

        return fc_mock, sent_mock, onch_mock

    async def run_hybrid_backtest(self, symbol: str, days: int = 365, cats: List[str] = None, rolling_window: int = 30) -> Dict[str, Any]:
        """
        Runs a walk-forward backtest with rolling window signal generation.
        """
        from modules.forecasting.data.preprocess_coin import CoinPreprocessor
        pre = CoinPreprocessor()
        df = pre.load_features_series(symbol)
        end_date = df.index[-1]
        start_date = end_date - pd.Timedelta(days=days)
        df_bt = df[start_date:end_date].copy()

        if len(df_bt) < rolling_window:
            logger.warning(f"Not enough data for rolling window of {rolling_window}. Have {len(df_bt)} points.")
            return {"metrics": {}, "positions": pd.DataFrame(), "signals": []}

        portfolio_value = self.initial_capital
        holdings = 0.0
        positions = []
        signals_over_time = []

        for i in range(rolling_window, len(df_bt)):
            window_df = df_bt.iloc[i-rolling_window:i]
            
            # Generate mocks and signal for the current time step based on past data
            fc_mock, sent_mock, onch_mock = await self._historical_mock(symbol, window_df, cats)
            
            signal_dict = hybrid_signal(window_df, fc_mock, sent_mock, onch_mock, symbol)
            signals_over_time.append(signal_dict)

            # Execute trade based on the generated signal
            price = df_bt.iloc[i]['close']
            sig = signal_dict['signal']
            size = signal_dict['position_size']

            if sig == "BUY" and portfolio_value > 0:
                units_to_buy = (portfolio_value * size) / price
                holdings += units_to_buy
                portfolio_value -= units_to_buy * price
            elif sig == "SELL" and holdings > 0:
                units_to_sell = holdings * size
                portfolio_value += units_to_sell * price
                holdings -= units_to_sell
            
            positions.append({
                "date": df_bt.index[i],
                "price": price,
                "signal": sig,
                "holdings": holdings,
                "portfolio_value": portfolio_value + holdings * price
            })

        if not positions:
            return {"metrics": {}, "positions": pd.DataFrame(), "signals": []}

        pos_df = pd.DataFrame(positions).set_index('date')
        returns = pos_df["portfolio_value"].pct_change().dropna()
        metrics = calculate_metrics(returns)

        # Log to MLflow
        with mlflow.start_run(run_name=f"v2_walk_forward_backtest_{symbol}_{days}d"):
            mlflow.log_params({
                "symbol": symbol, 
                "days": days, 
                "initial_capital": self.initial_capital,
                "rolling_window": rolling_window
            })
            mlflow.log_metrics(metrics)
            
            # Log positions as artifact
            pos_df.to_csv("positions.csv")
            mlflow.log_artifact("positions.csv", "positions_log")

        logger.info(f"Walk-forward backtest complete for {symbol}. Return: {metrics.get('total_return_pct', 0):.2f}%, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        return {"metrics": metrics, "positions": pos_df, "signals": signals_over_time}