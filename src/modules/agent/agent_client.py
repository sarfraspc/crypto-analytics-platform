"""
Main v2 orchestrator: Hybrid classify, parallel MCP, multi-LLM synthesis, hybrid backtest.
"""

import asyncio
import hashlib
import json
import logging
import os  # Added by agent
import re
import sys  # Added by agent
import traceback
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Dict, List, Optional
import io

import google.generativeai as genai
import pandas as pd
from httpx import AsyncClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolRequest  # Imported per MCP client spec (unused but kept for parity)

from core.config import settings
from core.logging_config import setup_logging
from modules.agent.backtester import PortfolioBacktester
from modules.agent.constants import LLM_REGISTRY
from modules.agent.prompts import construct_prompt
from modules.agent.query_classifier import HybridClassifier
from modules.agent.strategy_utils import hybrid_signal
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.explainers.xai import explain_model_predictions
from utils.cache import RedisCache

setup_logging()
logger = logging.getLogger(__name__)
cache = RedisCache(expire_seconds=1800)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_arguments(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(args or {})
    options = normalized.get("options")
    if isinstance(options, str):
        try:
            normalized["options"] = json.loads(options)
        except json.JSONDecodeError:
            logger.debug("Failed to deserialize options string; leaving as-is")
    elif options is None:
        normalized["options"] = {}
    return normalized


def _build_cache_key(server: str, tool: str, args: Dict[str, Any]) -> str:
    serialized = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return f"mcp:{server}:{tool}:{digest}"


def _parse_tool_output(text: str) -> Any:
    stripped = text.strip()
    try:
        if "DataFrame" in text or stripped.startswith("timestamp"):
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python', on_bad_lines='skip')
            if not df.empty:
                return df.iloc[-1].to_dict()
        if stripped.startswith('{') or stripped.startswith('['):
            return json.loads(stripped)
        if "Result:" in text:
            json_str = text.split("Result:")[-1].strip()
            return json.loads(json_str)
    except Exception as exc:
        logger.warning(f"Parsing MCP output failed: {exc}. Raw snippet: {text[:200]}")
    return {"raw": text}


def setup_environment_for_subprocesses():
    """
    Ensure every subprocess launched by stdio_client inherits the full
    serialized settings from this process.
    """
    logger.info("Serializing settings for subprocess environment...")
    for key, value in settings.dict().items():
        if value is None:
            env_value = ""
        elif isinstance(value, (str, int, float, bool)):
            env_value = str(value)
        else:
            try:
                env_value = json.dumps(value)
            except TypeError:
                logger.warning(f"Could not serialize setting '{key}', skipping.")
                continue
        os.environ[key.upper()] = env_value


setup_environment_for_subprocesses()

def _log_exception_group(exc: BaseException) -> None:
    """
    ExceptionGroup hides the concrete failure from TaskGroup.
    Surface every nested exception so we can see which server crashed.
    """
    if hasattr(exc, "exceptions"):
        for idx, sub in enumerate(exc.exceptions, start=1):
            logger.error("ExceptionGroup sub-exception %s:\n%s", idx, "".join(traceback.format_exception(sub)))


async def _call_mcp(server: str, tool: str, args: Optional[Dict[str, Any]] = None, *, use_cache: bool = True) -> Dict[str, Any]:
    normalized_args = _normalize_arguments(args)
    cache_key = _build_cache_key(server, tool, normalized_args)

    if use_cache and (cached := cache.get_json(cache_key)):
        return cached
    if not use_cache:
        cache.delete(cache_key)

    server_scripts = {
        "crypto-prophet-server": "src/mcp_servers/price_server.py", 
        
        "crypto-onchain-server": "src/mcp_servers/chain_server.py",
        "crypto-sentiment-server": "src/mcp_servers/sentiment_server.py",
        "crypto-agent-server": "src/mcp_servers/agent_server.py",
    }
    script_path = server_scripts.get(server)
    if not script_path or not os.path.exists(script_path):
        return {"error": f"Server script missing: {script_path}"}

    # Ensure subprocess inherits full environment (DB, Redis, etc.)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[script_path],
        cwd=os.getcwd(),
        env=dict(os.environ),
    )

    try:
        async with AsyncExitStack() as stack:
            read_transport, write_transport = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_transport, write_transport))

            await session.initialize()
            result = await session.call_tool(tool, arguments=normalized_args)

            if result.content and isinstance(result.content, list) and hasattr(result.content[0], 'text'):
                text = result.content[0].text
            else:
                text = str(result)

            parsed = _parse_tool_output(text)
            if isinstance(parsed, dict):
                parsed.setdefault("raw_text", text)
            else:
                parsed = {"raw_text": text, "value": parsed}

            if use_cache:
                cache.set_json(cache_key, parsed)
            return parsed

    except Exception as e:
        if hasattr(e, "exceptions"):
            _log_exception_group(e)
        logger.error(f"MCP {server}.{tool} failed: {e}", exc_info=True)
        return {"error": str(e)}


async def call_mcp_tool(
    server: str,
    tool: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Public helper used by FastAPI services and the agent orchestrator."""
    return await _call_mcp(server, tool, args, use_cache=use_cache)

async def route_tools(
    classify: Dict[str, List[str]],
    symbol: str = "BTC",
    df: Optional[pd.DataFrame] = None,
    query: str = "",
    options: Optional[Dict[str, Any]] = None,
    no_cache: bool = False
) -> Dict[str, Any]:
    tasks = {}
    cats = classify.get("categories", [])
    options = options or {}
    query_lower = (query or "").lower()
    use_cache = not no_cache

    horizon_days = max(1, _safe_int(options.get("horizon"), 7))
    forecast_horizon = max(1, horizon_days * 24)
    window = str(options.get("window") or "24h")
    k_docs = max(1, _safe_int(options.get("k_docs"), 5))
    explain_forecast = bool(options.get("explain_forecast"))

    logger.info(
        "Routing tools for categories=%s horizon_days=%s window=%s k_docs=%s",
        cats,
        horizon_days,
        window,
        k_docs,
    )

    if any(cat in cats for cat in ("forecast", "combined")):
        tasks["forecast"] = call_mcp_tool(
            "crypto-prophet-server",    # CHANGED from crypto-sarimax-server
            "forecast_prophet",         # CHANGED from forecast_sarimax
            {"symbol": symbol, "horizon": forecast_horizon},
            use_cache=use_cache,
        )

    wants_onchain = any(cat in cats for cat in ("onchain", "combined", "patterns"))
    if wants_onchain:
        base_onchain_args = {"window": window}
        wants_patterns = "patterns" in cats or any(
            word in query_lower for word in ["pattern", "rsi", "macd", "ta", "technical"]
        )
        if wants_patterns:
            pattern_args = {
                "exchange": "binance",
                "interval": "1d",
                "limit": 20,
            }
            tasks["onchain"] = call_mcp_tool(
                "crypto-onchain-server",
                "run_patterns_only",
                pattern_args,
                use_cache=use_cache,
            )
        else:
            tasks["onchain"] = call_mcp_tool(
                "crypto-onchain-server",
                "run_metrics_only",
                base_onchain_args,
                use_cache=use_cache,
            )

    if any(cat in cats for cat in ("sentiment", "combined")):
        sentiment_query = f"Current market sentiment and news about {symbol}"
        if any(token in query_lower for token in ['pattern', 'technical', 'ta']):
            sentiment_query += " including technical impact"

        tasks["sentiment"] = call_mcp_tool(
            "crypto-sentiment-server",
            "analyze_with_sources",
            {
                "query": sentiment_query,
                "k": k_docs,
                "include_sources": True,
            },
            use_cache=use_cache,
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    # Process results
    data = {}
    for tool_name, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            logger.error(f"Tool {tool_name} failed: {result}")
            data[tool_name] = {"error": f"Tool execution failed: {str(result)}"}
        else:
            if hasattr(result, "model_dump"):
                data[tool_name] = result.model_dump()
            elif isinstance(result, dict):
                data[tool_name] = result
            else:
                data[tool_name] = {"raw": result}

    if explain_forecast and "forecast" in data and df is not None and "error" not in data["forecast"]:
        try:
            from modules.forecasting.models.sarimax import SarimaxModel
            model = SarimaxModel(symbol)
            await asyncio.to_thread(model.load)
            shap_data = await asyncio.to_thread(
                explain_model_predictions, 
                'SARIMAX', model, CoinPreprocessor(), symbol, df.tail(100)
            )
            data["forecast"]["shap"] = shap_data
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            data["forecast"]["shap"] = {"error": f"SHAP explanation unavailable: {str(e)}"}

    if "sentiment" in data and isinstance(data["sentiment"], dict):
        if "sources" in data["sentiment"] or "response" in data["sentiment"]:
            data["rag"] = {
                "sources": data["sentiment"].get("sources"),
                "response": data["sentiment"].get("response"),
                "aggregated": data["sentiment"].get("aggregated"),
            }
        else:
            data["rag"] = data["sentiment"]

    return data

async def synthesize(provider: str, model: str, temp: float, prompt: str, max_retries: int = 2) -> str:
    """
    Dynamic LLM synthesis with fallback and retry logic to prevent recursion issues.
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting synthesis with {provider}/{model} (attempt {attempt + 1}/{max_retries})")
            
            if provider == "google":
                if not settings.GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY not configured")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                generative_model = genai.GenerativeModel(model)
                response = await asyncio.to_thread(
                    generative_model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temp,
                        max_output_tokens=2048
                    )
                )
                return response.text
            
            elif provider in ["groq", "openrouter"]:
                api_key = settings.GROQ_API_KEY if provider == "groq" else settings.OPENROUTER_API_KEY
                base_url = "https://api.groq.com/openai/v1" if provider == "groq" else "https://openrouter.ai/api/v1"
                
                if not api_key:
                    raise ValueError(f"{provider.upper()}_API_KEY not configured")
                
                async with AsyncClient(timeout=60.0) as client:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                        "max_tokens": 2048
                    }
                    
                    headers = {"Authorization": f"Bearer {api_key}"}
                    if provider == "openrouter":
                        headers["HTTP-Referer"] = "https://github.com/your-repo"
                        headers["X-Title"] = "Crypto Analytics Agent"
                    
                    response = await client.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.warning(f"Synthesis attempt {attempt + 1} failed with {provider}/{model}: {e}")
            if attempt + 1 >= max_retries:
                logger.error("Synthesis failed after all retries.")
                return "Analysis unavailable at the moment. The system may be experiencing high load or an external service is down. Please try again later."
            
            # Switch to fallback for the next attempt
            provider, model, temp = "groq", "llama-3.3-70b-versatile", 0.3
            
    return "Analysis failed due to an unexpected error after multiple retries."

class CryptoAgentV2:
    def __init__(self):
        self.classifier = HybridClassifier()
        self.backtester = PortfolioBacktester()
        self.current_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    async def run(
        self,
        query: str,
        symbol: str = "BTC",
        days: int = 30,
        options: Optional[Dict[str, Any]] = None,
        no_cache: bool = False,
        force_query_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main agent execution with dynamic LLM routing."""

        options = options or {}
        run_backtest_opt = bool(options.get("run_backtest"))
        symbol = symbol.upper()
        sanitized_query = (query or "").strip() or f"{symbol} market overview"
        horizon_days = max(1, _safe_int(options.get("horizon"), days))
        backtest_days = max(days, horizon_days)

        cache_envelope = {
            "symbol": symbol,
            "query": sanitized_query,
            "days": backtest_days,
            "force_query_type": force_query_type,
            "options": options,
        }
        cache_key = f"v2_agent:{hashlib.sha256(json.dumps(cache_envelope, sort_keys=True, default=str).encode()).hexdigest()}"

        if not no_cache and (cached := cache.get_json(cache_key)):
            logger.info("Returning cached response")
            return cached

        logger.info("Processing query='%s' for %s (horizon_days=%s)", sanitized_query, symbol, horizon_days)

        classification = await self.classifier.classify(sanitized_query)
        qtype = classification.get("qtype", "combined")
        categories = classification.get("categories", ["combined"])
        if not categories:
            categories = ["combined"]

        if force_query_type and force_query_type in LLM_REGISTRY:
            logger.info("Overriding classifier qtype %s -> %s", qtype, force_query_type)
            qtype = force_query_type
            classification["qtype"] = qtype

        if qtype == "long_context" and "backtest" not in categories:
            categories = [*categories, "backtest"]

        # Ensure downstream routing sees the updated categories
        categories = list(dict.fromkeys(categories))
        classification["categories"] = categories

        logger.info("Query classified as: %s, categories: %s", qtype, categories)

        preprocessor = CoinPreprocessor()
        df = await asyncio.to_thread(preprocessor.load_features_series, symbol)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        data = await route_tools(classification, symbol, df, sanitized_query, options, no_cache)

        run_backtest = run_backtest_opt or "backtest" in categories or qtype == "long_context"
        backtest = None
        if run_backtest:
            logger.info("Running backtest for %s days", backtest_days)
            backtest = await self.backtester.run_hybrid_backtest(symbol, backtest_days, categories)
            data["backtest"] = backtest

        strategy = None
        if qtype in ["combined", "reasoning", "patterns"]:
            query_hash = hashlib.sha256(sanitized_query.encode()).hexdigest()
            strategy = hybrid_signal(
                df,
                data.get("forecast", {}),
                data.get("sentiment", {}),
                data.get("onchain", {}),
                symbol,
                query_hash,
                backtest_days,
            )
            data["strategy"] = strategy

        provider, model, temperature = LLM_REGISTRY.get(qtype, LLM_REGISTRY["combined"])
        logger.info("Using %s/%s for %s query (temp=%s)", provider, model, qtype, temperature)

        prompt_text = construct_prompt(sanitized_query, data, qtype, self.current_date, categories)
        response = await synthesize(provider, model, temperature, prompt_text)

        result = {
            "symbol": symbol,
            "query_type": qtype,
            "categories": categories,
            "llm_used": f"{provider}/{model}",
            "final_answer": response,
            "response": response,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "data_insights": {
                "forecast_available": "forecast" in data,
                "sentiment_available": "sentiment" in data,
                "onchain_available": "onchain" in data,
                "backtest_available": "backtest" in data,
                "strategy_available": "strategy" in data,
            },
            **data,
        }

        if not no_cache:
            cache.set_json(cache_key, result)
            logger.info("Cached agent response under %s", cache_key)

        return result


_AGENT_SINGLETON: Optional[CryptoAgentV2] = None


def _get_agent() -> CryptoAgentV2:
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = CryptoAgentV2()
    return _AGENT_SINGLETON


_SYMBOL_STOPWORDS = {
    "RUN",
    "PRICE",
    "FORECAST",
    "ONCHAIN",
    "SENTIMENT",
    "MARKET",
    "BACKTEST",
    "REPORT",
    "OVERVIEW",
    "HISTORY",
    "PAST",
    "LAST",
    "DAYS",
    "DAY",
    "WEEK",
    "WEEKS",
    "MONTH",
    "MONTHS",
    "YEAR",
    "YEARS",
    "SIMPLE",
    "STRATEGY",
    "FOR",
    "THE",
    "AND",
    "OR",
    "OF",
    "TO",
    "IN",
}

_SYMBOL_ALIASES = {
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
}


def _infer_symbol_from_text(question: str, fallback: str = "BTC") -> str:
    """
    Lightweight symbol extractor that uses the natural-language question
    as the primary source of truth for the asset ticker.

    - Detects common asset names like 'bitcoin'/'ethereum'.
    - Then looks for ticker-like tokens (2–10 uppercase letters).
    - Ignores obvious non-symbol words via a small stopword set.
    - Falls back to the provided fallback (or BTC) if nothing is found.
    """
    base = (question or "").strip()
    if not base:
        return (fallback or "BTC").upper()

    upper = base.upper()

    # Name-based aliases first (e.g., "bitcoin" -> BTC).
    for name, sym in _SYMBOL_ALIASES.items():
        if name in upper:
            return sym

    # Then look for ticker-like tokens, normalizing to base symbols:
    # - Split pairs like "BTC/USDT" -> "BTC"
    # - Strip common quote suffixes like "BTCUSDT" -> "BTC"
    matches = re.findall(r"\\b[A-Z]{2,10}\\b", upper)
    for token in matches:
        if token in _SYMBOL_STOPWORDS:
            continue

        base_token = token
        if "/" in base_token:
            base_token = base_token.split("/", 1)[0]
        for suffix in ("USDT", "USD", "USDC", "BUSD", "PERP"):
            if base_token.endswith(suffix) and len(base_token) > len(suffix) + 1:
                base_token = base_token[: -len(suffix)]
                break

        return base_token

    return (fallback or "BTC").upper()


async def orchestrate_query(
    symbol: str,
    question: str,
    options: Optional[Dict[str, Any]] = None,
    no_cache: bool = False,
    force_query_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Entry point used by the Agent MCP server."""

    raw_question = (question or "").strip()
    # Use the question text as the primary source of truth for the asset,
    # falling back to the provided symbol or BTC when nothing is detected.
    inferred_symbol = _infer_symbol_from_text(raw_question, fallback=(symbol or "BTC"))
    safe_symbol = inferred_symbol.upper()
    safe_question = raw_question or f"{safe_symbol} market overview"
    safe_options = options or {}

    horizon_hint = _safe_int(safe_options.get("horizon"), 30)
    days_hint = max(horizon_hint, _safe_int(safe_options.get("backtest_days"), horizon_hint or 30))
    return await _get_agent().run(
        query=safe_question,
        symbol=safe_symbol,
        days=days_hint,
        options=safe_options,
        no_cache=no_cache,
        force_query_type=force_query_type,
    )

async def main():
    """CLI interface for testing the agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Analytics Agent v2.1")
    parser.add_argument("--query", required=True, help="User query about crypto markets")
    parser.add_argument("--symbol", default="BTC", help="Cryptocurrency symbol (default: BTC)")
    parser.add_argument("--days", type=int, default=30, help="Backtest period in days (default: 30)")
    
    args = parser.parse_args()
    
    agent = CryptoAgentV2()
    
    try:
        result = await agent.run(args.query, args.symbol, args.days)
        
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        print(f"SYMBOL: {args.symbol}")
        print(f"QUERY TYPE: {result['query_type']}")
        print(f"LLM USED: {result['llm_used']}")
        print("="*80)
        final_answer = result.get('final_answer', result.get('response', ''))
        print(f"\nRESPONSE:\n{final_answer}")
        print("\n" + "="*80)
        
        if result.get('strategy'):
            print(f"\nTRADING STRATEGY: {result['strategy']}")
            
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
