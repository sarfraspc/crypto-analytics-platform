"""
Merged backtester: Agent 1 simulation/MLflow + Agent 2 hybrid signals/metrics (Sharpe/Sortino).
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from sqlalchemy import select

try:
    import mlflow
except Exception:  # pragma: no cover - MLflow optional dependency
    mlflow = None

from core.config import settings
from core.database import get_timescale_db
from data.storage.models import (
    TASignalHistory,  # Historical TA (reserved for future use)
    OnchainMetric as OnchainMetricModel,
    NewsArticle,
    RedditPost,
)
from modules.agent.strategy_utils import hybrid_signal
from modules.sentiment.models.sentiment_infer import (
    analyze_sentiment_batch,  # Threaded
    get_sentiment_classifier,
)

logger = logging.getLogger(__name__)

_SENTIMENT_TOKENIZER = None


def _get_sentiment_tokenizer():
    """
    Lazily load the tokenizer used by the fine-tuned sentiment model, so
    historical texts are truncated in exactly the same token space.
    """
    global _SENTIMENT_TOKENIZER
    if _SENTIMENT_TOKENIZER is None:
        try:
            classifier = get_sentiment_classifier()
            _SENTIMENT_TOKENIZER = getattr(classifier.classifier, "tokenizer", None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load sentiment tokenizer; falling back to naive truncation: %s", exc)
            _SENTIMENT_TOKENIZER = None
    return _SENTIMENT_TOKENIZER

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns.empty or len(returns) < 2:
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "trades_count": 0,
        }
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
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd * 100),
        # trades_count is populated at the backtest level from actual entry/exit events.
        "trades_count": 0,
    }        

def _mlflow_server_available() -> bool:
    if mlflow is None:
        return False
    try:
        tracking_uri = mlflow.get_tracking_uri() or ""
        if tracking_uri.startswith("file:"):
            return True
        requests.get(f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/experiments/list", timeout=1)
        return True
    except Exception:
        return False


class PortfolioBacktester:
    def __init__(
        self,
        initial_capital: float = 10000,
        experiment: str = "crypto_backtest_v2",
        enable_mlflow: bool | None = None,
    ):
        self.initial_capital = initial_capital
        # Flat fee applied to each trade (both entry and exit) as a fraction of notional.
        # 0.0005 = 0.05%
        self.fee_rate: float = 0.0005
        self.experiment = experiment
        env_toggle = os.getenv("ENABLE_MLFLOW", "").lower()
        disable_toggle = os.getenv("DISABLE_MLFLOW", "").lower() in {"1", "true", "yes"}

        if enable_mlflow is None:
            enable_mlflow = (mlflow is not None) and env_toggle not in {"0", "false"} and not disable_toggle

        # Best-effort MLflow tracking URI configuration using app settings when
        # no explicit tracking URI has been set via environment.
        if enable_mlflow and mlflow is not None:
            try:
                current_uri = mlflow.get_tracking_uri() or ""
                if not current_uri and getattr(settings, "MLFLOW_TRACKING_URI", None):
                    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                    logger.info("MLflow tracking URI set from settings: %s", settings.MLFLOW_TRACKING_URI)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not configure MLflow tracking URI for backtester: %s", exc)

        if enable_mlflow and _mlflow_server_available() and os.getenv("CI") != "true":
            self.enable_mlflow = True
            try:
                mlflow.set_experiment(self.experiment)
                logger.info("MLflow enabled for backtester.")
            except Exception as exc:  # pragma: no cover - server issues
                self.enable_mlflow = False
                logger.warning("Disabling MLflow logging; failed to set experiment: %s", exc)
        else:
            self.enable_mlflow = False
            logger.info("MLflow disabled (server not running or env disabled).")

    def _persistence_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple persistence forecast: predicts the last known close.
        This avoids lookahead while staying consistent with the live strategy utils.
        """
        last_close = float(df["close"].iloc[-1])
        return {"predicted_close": [last_close]}

    async def _load_historical_sentiment(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> Dict[str, Any]:
        """
        Load historical sentiment by querying stored news and Reddit posts
        for the given window and aggregating model scores.
        Falls back to neutral sentiment if data or inference is unavailable.
        """

        def _fetch_texts() -> List[str]:
            texts: List[str] = []
            tokenizer = _get_sentiment_tokenizer()
            try:
                with get_timescale_db() as session:
                    news_q = (
                        select(NewsArticle)
                        .where(
                            NewsArticle.published >= start_ts,
                            NewsArticle.published <= end_ts,
                        )
                        .order_by(NewsArticle.published.asc())
                        .limit(200)
                    )
                    reddit_q = (
                        select(RedditPost)
                        .where(
                            RedditPost.created >= start_ts,
                            RedditPost.created <= end_ts,
                        )
                        .order_by(RedditPost.created.asc())
                        .limit(200)
                    )
                    news_rows = session.execute(news_q).scalars().all()
                    reddit_rows = session.execute(reddit_q).scalars().all()

                    def _truncate(text: str) -> str:
                        if not text:
                            return ""
                        if tokenizer is None:
                            # Fallback: rough character-level guardrail.
                            return text[:2000]
                        encoded = tokenizer.encode(
                            text,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=500,
                        )
                        return tokenizer.decode(encoded, skip_special_tokens=True)

                    for row in news_rows:
                        parts = [row.title or "", row.text or ""]
                        combined = " ".join(p for p in parts if p)
                        if combined:
                            texts.append(_truncate(combined))

                    for row in reddit_rows:
                        parts = [row.title or "", row.body or ""]
                        combined = " ".join(p for p in parts if p)
                        if combined:
                            texts.append(_truncate(combined))
            except Exception as exc:
                logger.warning("Historical sentiment DB query failed: %s", exc)
            return texts

        texts = await asyncio.to_thread(_fetch_texts)
        if not texts:
            # Fallback to neutral sentiment if no historical texts are available
            return {"aggregated": {"bullish_score": 0.5, "bearish_score": 0.5}}

        try:
            results = await asyncio.to_thread(analyze_sentiment_batch, texts)
        except Exception as exc:
            logger.warning("Historical sentiment inference failed: %s", exc)
            return {"aggregated": {"bullish_score": 0.5, "bearish_score": 0.5}}

        bull_scores = [r.get("bullish_score", 0.0) for r in results]
        bear_scores = [r.get("bearish_score", 0.0) for r in results]

        if bull_scores:
            bull = float(np.mean(bull_scores))
        else:
            bull = 0.5

        if bear_scores:
            bear = float(np.mean(bear_scores))
        else:
            bear = 0.5

        return {"aggregated": {"bullish_score": bull, "bearish_score": bear}}

    def _infer_chain_from_symbol(self, symbol: str) -> str:
        """
        Map a trading symbol to a blockchain for on-chain metrics.
        """
        base = symbol.upper().split("/")[0]
        if base in {"ETH", "WETH"}:
            return "ethereum"
        if base in {"BTC", "WBTC"}:
            return "bitcoin"
        # Default to Ethereum for other assets
        return "ethereum"

    async def _load_historical_onchain(
        self,
        symbol: str,
        as_of: pd.Timestamp,
    ) -> Dict[str, Any]:
        """
        Load historical on-chain snapshot (market pressure and whale-price correlation)
        from the onchain_metrics table as of a given timestamp.
        Falls back to neutral defaults when data is unavailable.
        """
        chain = self._infer_chain_from_symbol(symbol)

        def _fetch_onchain() -> Dict[str, float]:
            pressure_val = None
            corr_val = None
            try:
                with get_timescale_db() as session:
                    def latest(metric_name: str):
                        stmt = (
                            select(OnchainMetricModel.value)
                            .where(
                                OnchainMetricModel.chain == chain,
                                OnchainMetricModel.metric == metric_name,
                                OnchainMetricModel.time <= as_of,
                            )
                            .order_by(OnchainMetricModel.time.desc())
                            .limit(1)
                        )
                        return session.execute(stmt).scalar_one_or_none()

                    pressure_val = latest("market_pressure_index")
                    corr_val = latest("price_whale_corr_7d")
            except Exception as exc:
                logger.warning("Historical on-chain DB query failed: %s", exc)

            pressure = float(pressure_val) if pressure_val is not None else 0.5
            corr = float(corr_val) if corr_val is not None else 0.0
            return {
                "market_pressure_index": pressure,
                "price_whale_corr_7d": corr,
            }

        return await asyncio.to_thread(_fetch_onchain)

    async def _historical_mock(self, symbol: str, df: pd.DataFrame, cats: List[str] = None) -> tuple:
        """
        Historical context loader for backtesting.

        Replaces the old random/synthetic mocks with:
        - Persistence price forecast derived from the window's close prices.
        - Sentiment aggregated from historical news and Reddit content.
        - On-chain snapshot pulled from onchain_metrics as of the window end.
        """
        if df.empty:
            return {"predicted_close": []}, {"aggregated": {}}, {}

        # Normalize timestamps to UTC for DB queries
        start_ts = pd.Timestamp(df.index[0])
        end_ts = pd.Timestamp(df.index[-1])
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        fc_ctx = self._persistence_forecast(df)
        sent_ctx = await self._load_historical_sentiment(symbol, start_ts, end_ts)
        onch_ctx = await self._load_historical_onchain(symbol, end_ts)

        return fc_ctx, sent_ctx, onch_ctx

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
            return {"metrics": {}, "equity_curve": [], "trades": [], "signals": []}

        # Precompute heavier historical context (sentiment + on-chain) once for the whole backtest window
        # to avoid running expensive inference on every single step.
        _, sent_ctx, onch_ctx = await self._historical_mock(symbol, df_bt, cats)

        portfolio_value = self.initial_capital
        holdings = 0.0
        positions = []
        signals_over_time = []
        trades_count = 0

        for i in range(rolling_window, len(df_bt)):
            window_df = df_bt.iloc[i-rolling_window:i]

            # Lightweight per-step context: recompute persistence forecast on the rolling window,
            # but reuse aggregated sentiment and on-chain context for the full backtest horizon.
            fc_mock = self._persistence_forecast(window_df)
            sent_mock = sent_ctx
            onch_mock = onch_ctx

            signal_dict = hybrid_signal(window_df, fc_mock, sent_mock, onch_mock, symbol)
            signals_over_time.append(signal_dict)

            # Execute trade based on the generated signal
            price = df_bt.iloc[i]['close']
            sig = signal_dict['signal']
            size = float(signal_dict.get('position_size', 0.0))
            # Clamp position size to avoid excessive leverage and negative cash after fees.
            size = max(0.0, min(size, 0.95))

            trade_executed = False
            if sig == "BUY" and portfolio_value > 0 and size > 0:
                trade_notional = portfolio_value * size
                if trade_notional > 0 and price > 0:
                    fee = trade_notional * self.fee_rate
                    units_to_buy = trade_notional / price
                    holdings += units_to_buy
                    portfolio_value -= (trade_notional + fee)
                    trades_count += 1
                    trade_executed = True
            elif sig == "SELL" and holdings > 0 and size > 0:
                units_to_sell = holdings * size
                trade_notional = units_to_sell * price
                if trade_notional > 0:
                    fee = trade_notional * self.fee_rate
                    portfolio_value += (trade_notional - fee)
                    holdings -= units_to_sell
                    trades_count += 1
                    trade_executed = True
            
            positions.append({
                "date": df_bt.index[i],
                "price": price,
                "signal": sig,
                "position_size": size,
                "trade_executed": trade_executed,
                "holdings": holdings,
                "portfolio_value": portfolio_value + holdings * price
            })

        if not positions:
            return {"metrics": {}, "equity_curve": [], "trades": [], "signals": []}

        pos_df = pd.DataFrame(positions).set_index('date')
        returns = pos_df["portfolio_value"].pct_change().dropna()
        metrics = calculate_metrics(returns)
        # Override generic trades_count with the actual number of executed entry/exit events.
        metrics["trades_count"] = trades_count

        # JSON-serializable equity curve and trade list for API consumers.
        equity_curve = []
        trades = []
        for row in positions:
            ts = row.get("date")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            point = {
                "time": ts_str,
                "price": float(row.get("price", 0.0)),
                "signal": row.get("signal"),
                "holdings": float(row.get("holdings", 0.0)),
                "portfolio_value": float(row.get("portfolio_value", 0.0)),
            }
            equity_curve.append(point)
            if row.get("trade_executed"):
                trades.append({
                    "time": ts_str,
                    "side": row.get("signal"),
                    "price": float(row.get("price", 0.0)),
                    "position_size": float(row.get("position_size", 0.0)),
                    "holdings": float(row.get("holdings", 0.0)),
                    "portfolio_value": float(row.get("portfolio_value", 0.0)),
                })

        if self.enable_mlflow:
            try:
                with mlflow.start_run(run_name=f"v2_walk_forward_backtest_{symbol}_{days}d"):
                    mlflow.log_params({
                        "symbol": symbol,
                        "days": days,
                        "initial_capital": self.initial_capital,
                        "rolling_window": rolling_window,
                        "experiment": self.experiment,
                        "categories": ",".join(cats or []),
                        "n_points": len(df_bt),
                        "n_trades": trades_count,
                    })
                    mlflow.log_metrics(metrics)

                    # Log positions as an in-memory CSV artifact without writing to project files.
                    csv_payload = pos_df.to_csv()
                    mlflow.log_text(csv_payload, "positions_log/positions.csv")
            except Exception as exc:
                logger.warning("MLflow logging skipped due to error: %s", exc)

        logger.info(f"Walk-forward backtest complete for {symbol}. Return: {metrics.get('total_return_pct', 0):.2f}%, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        return {"metrics": metrics, "equity_curve": equity_curve, "trades": trades, "signals": signals_over_time}
