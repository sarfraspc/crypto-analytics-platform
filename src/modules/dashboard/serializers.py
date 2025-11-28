"""
Dashboard data serialization module.

Provides formatting functions to convert MCP results and backtest
data into structured JSON for frontend dashboard consumption.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def format_overview(
    agent_data: Dict[str, Any],
    market_sentiment: Dict[str, Any],
    whale_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert mixed MCP results into dashboard overview JSON.
    """
    try:
        # Derive sentiment index from market_sentiment
        bullish = market_sentiment.get("scores", {}).get("bullish", 0.5)
        sentiment_index = round(bullish, 2)

        # Derive whale trend from whale_metrics
        whale_tx = whale_metrics.get("total_whale_txs", 0)
        whale_trend = "rising" if whale_tx > 10 else "stable" if whale_tx > 3 else "quiet"

        # Extract top asset snapshot from agent_data (extend for multiple if needed)
        top_assets = [{
            "symbol": agent_data.get("symbol", "BTC"),
            "sentiment": agent_data.get("sentiment", {}).get("overall", "neutral"),
            "forecast_change_pct": round(
                # Derive change_pct from forecast if available; fallback to 0
                (agent_data.get("forecast", {}).get("predicted_close", [0])[-1] - agent_data.get("forecast", {}).get("current_price", 0)) / agent_data.get("forecast", {}).get("current_price", 1) * 100, 2
            ) if agent_data.get("forecast") else 0.0,
            "pressure": whale_metrics.get("market_pressure", "neutral")
        }]

        return {
            "top_assets": top_assets,
            "market_sentiment_index": sentiment_index,
            "whale_activity_trend": whale_trend,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"format_overview failed: {e}")
        return {"error": f"format_overview failed: {e}"}

def format_portfolio(backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten multiple agent backtest results into one portfolio summary.
    """
    portfolio = []
    for bt in backtest_results:
        # Handle raw_text parsing if needed (consistent with services)
        if isinstance(bt, dict) and "raw_text" in bt:
            # Extract backtest from parsed JSON if available
            bt_text = bt["raw_text"]
            json_match = re.search(r'\{.*\}', bt_text, re.DOTALL)
            if json_match:
                bt = json.loads(json_match.group())
        metrics = bt.get("backtest", {}).get("metrics", {}) if isinstance(bt, dict) else {}
        portfolio.append({
            "symbol": bt.get("symbol", "N/A"),
            "strategy_return_pct": round(metrics.get("total_return_pct", 0), 2),
            "sharpe_ratio": round(metrics.get("sharpe_ratio", 0), 2),
            "sortino_ratio": round(metrics.get("sortino_ratio", 0), 2),
            "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2)
        })

    # Portfolio-level aggregates
    avg_sharpe = (
        sum(p["sharpe_ratio"] for p in portfolio) / len(portfolio)
        if portfolio else 0
    )

    return {
        "portfolio": portfolio,
        "avg_sharpe": round(avg_sharpe, 2),
        "period_days": 30,
        "timestamp": datetime.now().isoformat()
    }