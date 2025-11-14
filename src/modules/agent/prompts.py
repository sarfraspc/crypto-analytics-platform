"""
Unified prompts with context builders for multi-type responses.
"""

import json
import pandas as pd
import re
import logging
from typing import Dict, Any
from datetime import datetime
import pytz  # For UTC dynamic

logger = logging.getLogger(__name__)

# Core system prompt (shared)
SYSTEM_PROMPT = """
You are CryptoAgent v2.0, an expert AI for crypto analytics and trading strategies.
Access tools via MCP: Forecasting (SARIMAX), On-Chain (whales/flows/TA), Sentiment/RAG (news/Reddit + scoring).

Guidelines:
- Classify intent: forecast/onchain/sentiment/combined/backtest.
- Parallel tools for efficiency.
- Reason step-by-step: Analyze outputs, synthesize, suggest actions (e.g., "BUY 0.5 BTC: Bullish alignment").
- Risk-aware: Adjust for volatility/pressure.
- Output: Summary, metrics (tables), recommendation. Be concise/data-driven.

Current date: {current_date}.
"""

# Type-specific instructions
TYPE_INSTRUCTIONS = {
    "real_time": """
    QUERY: Quick metrics.
    FORMAT: Direct answer + key numbers (1-2 sentences). E.g., "BTC: $45k, bullish (78% conf), net inflow +4k BTC."
    """,
    "reasoning": """
    QUERY: Causal analysis.
    FORMAT: Summary → Evidence (data refs) → Interpretation → Outlook. Use bullets.
    """,
    "long_context": """
    QUERY: Report/backtest.
    FORMAT: Sections: Exec Summary | Price/Forecast | Sentiment | On-Chain | Strategy/Backtest. Tables for metrics.
    """,
    "combined": """
    QUERY: Multi-tool synthesis.
    FORMAT: Align signals (F/O/S scores), generate hybrid strategy. Include risk-adjusted signal.
    """
}

# Unified context template
CONTEXT_TEMPLATE = """
DATA INSIGHTS:
{forecast_ctx}
{sentiment_ctx}
{onchain_ctx}
{backtest_ctx}

QUERY: {query}
INSTRUCTIONS: {type_instructions}
"""

def _robust_parse(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback parser for tool outputs (text/JSON/DF)."""
    if isinstance(raw, dict) and 'raw' in raw:
        text = raw['raw']
        # SARIMAX: Detect header, parse DF
        if "SARIMAX Forecast" in text and "timestamp" in text:
            try:
                # Find DF start
                df_start = text.find('timestamp')
                df_text = text[df_start:]
                lines = df_text.split('\n')[1:]  # Skip header
                df_lines = [line for line in lines if line.strip() and not line.startswith('SARIMAX')]
                if df_lines:
                    # Manual split for irregular spaces
                    data = []
                    for line in df_lines:
                        parts = re.split(r'\s{2,}', line.strip())  # Split on multiple spaces
                        if len(parts) == 2:
                            data.append({'timestamp': parts[0], 'predicted_close': float(parts[1])})
                    if data:
                        df = pd.DataFrame(data)
                        parsed = df.iloc[-1].to_dict()
                        if len(df) > 1:
                            parsed['last_close'] = float(df['predicted_close'].iloc[-2])
                        else:
                            parsed['last_close'] = parsed.get('predicted_close', 0)
                        return parsed
            except Exception as e:
                logger.warning(f"SARIMAX parse failed: {e}")
        # RAG/Pipeline: Extract sections
        if any(k in text for k in ["Generated Answer:", "Aggregated Sentiment:", "Sentiment:"]):
            try:
                sections = {}
                for key in ["Generated Answer:", "Aggregated Sentiment:", "Sentiment:"]:
                    if key in text:
                        start = text.find(key) + len(key)
                        end = text.find('\n\n', start) if '\n\n' in text[start:] else len(text)
                        sections[key.strip(':').lower().replace(' ', '_')] = text[start:end].strip()
                # Parse agg sentiment
                if 'aggregated_sentiment' in sections:
                    agg_text = sections['aggregated_sentiment']
                    match = re.search(r'(\w+) \((Avg Conf: ([\d.]+))', agg_text)
                    if match:
                        sections['top_sentiment'] = match.group(1).upper()
                        sections['top_confidence'] = float(match.group(3))
                # Clean 'sentiment' section (exclude aggregated if duplicate)
                if 'sentiment' in sections and 'aggregated_sentiment' in sections:
                    sections['sentiment'] = sections['sentiment'].split('Aggregated')[0].strip()
                return sections or {"raw": text}
            except: pass
        # Fallback JSON/onchain
        try:
            json_start = text.find(":\n\n") + 3 if ":\n\n" in text else 0
            return json.loads(text[json_start:])
        except: pass
    return raw or {}

def build_forecast_ctx(forecast: Dict[str, Any]) -> str:
    forecast = _robust_parse(forecast)
    if not forecast: return "FORECAST: N/A"
    pred = forecast.get('predicted_close', 0)
    last = forecast.get('last_close', pred)
    trend = "↑ Bullish" if pred > last * 1.01 else "↓ Bearish" if pred < last * 0.99 else "→ Neutral"
    ctx = f"FORECAST: {trend} (Horizon: {forecast.get('horizon', 7)}h, Pred Close: ${pred:.2f}, MAE: {forecast.get('mae_forecast', 0):.2f})"
    if 'shap' in forecast:  # Inject SHAP
        shap_mean = forecast['shap'].get('mean_abs_shap', {})
        top_feat = max(shap_mean, key=shap_mean.get) if shap_mean else 'N/A'
        ctx += f" | Top Driver: {top_feat} (SHAP: {shap_mean.get(top_feat, 0):.3f})"
    return ctx

def build_sentiment_ctx(sentiment: Dict[str, Any]) -> str:
    sentiment = _robust_parse(sentiment)
    if not sentiment: return "SENTIMENT: N/A"
    top = sentiment.get('top_sentiment', 'NEUTRAL')
    conf = sentiment.get('top_confidence', 0)
    bull = sentiment.get('bullish_score', 0)
    bear = sentiment.get('bearish_score', 0)
    return f"SENTIMENT: {top} (Conf: {conf:.1%}, Bull: {bull:.2f}, Bear: {bear:.2f})"

def build_onchain_ctx(onchain: Dict[str, Any]) -> str:
    onchain = _robust_parse(onchain)
    if not onchain: return "ON-CHAIN: N/A"
    bias = onchain.get('market_bias', 'neutral')
    pressure = onchain.get('market_pressure_index', 0)
    return f"ON-CHAIN: {bias.upper()} Bias (Pressure: {pressure:.2f}, Net Flow: ${onchain.get('net_flow_usd', 0):,.0f})"

def build_backtest_ctx(backtest: Dict[str, Any]) -> str:
    if not backtest: return "BACKTEST: N/A"
    m = backtest.get('metrics', {})
    return f"BACKTEST: Return {m.get('total_return', 0):+.1%} | Sharpe {m.get('sharpe_ratio', 0):.2f} | DD {m.get('max_drawdown', 0):+.1%} | Trades {m.get('trades_count', 0)}"

def construct_prompt(query: str, data: Dict[str, Any], qtype: str, current_date: str) -> str:
    # Dynamic UTC date
    utc_now = datetime.now(pytz.UTC).strftime("%Y-%m-%d") if current_date is None else current_date
    parts = {
        "current_date": utc_now,
        "forecast_ctx": build_forecast_ctx(data.get('forecast', {})),
        "sentiment_ctx": build_sentiment_ctx(data.get('sentiment', {})),
        "onchain_ctx": build_onchain_ctx(data.get('onchain', {})),
        "backtest_ctx": build_backtest_ctx(data.get('backtest', {})),
        "query": query,
        "type_instructions": TYPE_INSTRUCTIONS.get(qtype, TYPE_INSTRUCTIONS['reasoning'])
    }
    context = CONTEXT_TEMPLATE.format(**parts)
    return f"{SYSTEM_PROMPT.format(current_date=utc_now)}\n\n{context}\nSYNTHESIZE:"