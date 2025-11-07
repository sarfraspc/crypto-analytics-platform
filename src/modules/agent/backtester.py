import pandas as pd
import numpy as np
from typing import Dict, Optional
from modules.forecasting.data.preprocess_coin import CoinPreprocessor

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns.empty or len(returns) < 2:
        return {"total_return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0, "volatility_pct": 0.0}

    cumprod = (1 + returns).cumprod()
    total_return = cumprod.iloc[-1] - 1

    annualized_return = returns.mean() * 365
    annualized_vol = returns.std() * np.sqrt(365)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    downside = returns[returns < 0]
    sortino = annualized_return / (downside.std() * np.sqrt(365)) if len(downside) > 0 and downside.std() > 0 else 0.0

    drawdowns = cumprod / cumprod.cummax() - 1
    max_dd = drawdowns.min()

    return {
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "annual_volatility_pct": float(annualized_vol * 100)
    }

def run_backtest(symbol: str, signals: pd.Series, historical_df: Optional[pd.DataFrame] = None, days: int = 365) -> Dict:
    if historical_df is None:
        pre = CoinPreprocessor()
        historical_df = pre.load_features_series(symbol)

    end_date = historical_df.index[-1]
    start_date = end_date - pd.Timedelta(days=days)
    df = historical_df[start_date:end_date].copy()

    df["signal"] = signals.reindex(df.index).ffill().fillna(0)
    df["market_return"] = df["close"].pct_change()
    df["strategy_return"] = df["market_return"] * df["signal"].shift(1).fillna(0)

    metrics = calculate_metrics(df["strategy_return"].dropna())

    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["market_return"]).cumprod()

    return {
        "period_days": days,
        "metrics": metrics,
        "equity_curve": df["cum_strategy"].to_dict(),
        "benchmark_curve": df["cum_market"].to_dict(),
        "trade_count": int((df["signal"].diff().abs() == 2).sum())
    }