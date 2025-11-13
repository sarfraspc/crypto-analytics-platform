"""
Main v2 orchestrator: Hybrid classify, parallel MCP, multi-LLM synthesis, hybrid backtest.
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import io
import pandas as pd
import os  # Added by agent
import sys  # Added by agent
import re  # Added by agent
import traceback
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import CallToolRequest  # Imported per MCP client spec (unused but kept for parity)
from httpx import AsyncClient
import google.generativeai as genai

from core.config import settings
from core.logging_config import setup_logging
from utils.cache import RedisCache
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.explainers.xai import explain_model_predictions 
from modules.agent.prompts import construct_prompt
from modules.agent.query_classifier import HybridClassifier
from modules.agent.backtester import PortfolioBacktester
from modules.agent.strategy_utils import hybrid_signal
from modules.agent.constants import LLM_REGISTRY

setup_logging()
logger = logging.getLogger(__name__)
cache = RedisCache(expire_seconds=1800)


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


async def call_mcp(server: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    args = args or {}
    key = f"mcp:{server}:{tool}:{hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()}"
    if cached := cache.get_json(key):
        return cached

    server_scripts = {
        "crypto-sarimax-server": "src/mcp_servers/price_server.py",
        "crypto-onchain-server": "src/mcp_servers/chain_server.py",
        "crypto-pipeline-server": "src/mcp_servers/sentiment_server.py"
    }
    script_path = server_scripts.get(server)
    if not script_path or not os.path.exists(script_path):
        return {"error": f"Server script missing: {script_path}"}

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[script_path],
        cwd=os.getcwd()
    )

    try:
        async with AsyncExitStack() as stack:
            read_transport, write_transport = await stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await stack.enter_async_context(
                ClientSession(read_transport, write_transport)
            )

            await session.initialize()
            call_arguments = args if args is not None else {}
            result = await session.call_tool(tool, arguments=call_arguments)

            if result.content and isinstance(result.content, list) and hasattr(result.content[0], 'text'):
                text = result.content[0].text
            else:
                text = str(result)

            try:
                if 'DataFrame' in text or text.startswith('timestamp'):
                    df = pd.read_csv(io.StringIO(text), sep=None, engine='python', on_bad_lines='skip')
                    parsed = df.iloc[-1].to_dict() if not df.empty else {"raw": text}
                elif text.strip().startswith('{') or text.strip().startswith('['):
                    parsed = json.loads(text)
                elif "Result:" in text:
                    json_str = text.split("Result:")[-1].strip()
                    parsed = json.loads(json_str)
                else:
                    parsed = {"raw": text}
            except Exception as e:
                logger.warning(f"Parsing failed: {e}. Raw: {text[:200]}...")
                parsed = {"raw": text}

            cache.set_json(key, parsed)
            return parsed

    except Exception as e:
        if hasattr(e, "exceptions"):
            _log_exception_group(e)
        logger.error(f"MCP {server}.{tool} failed: {e}", exc_info=True)
        return {"error": str(e)}

async def route_tools(classify: Dict[str, List[str]], symbol: str = "BTC", df: pd.DataFrame = None, query: str = "") -> Dict[str, Any]:
    tasks = {}
    cats = classify["categories"]
    
    logger.info(f"Routing tools for categories: {cats}")

    if "forecast" in cats:
        tasks["forecast"] = call_mcp("crypto-sarimax-server", "forecast_sarimax", {"symbol": symbol, "horizon": 7})

    if "onchain" in cats:
        # Determine which on-chain tools to call based on query
        if any(word in query.lower() for word in ['pattern', 'rsi', 'macd', 'ta', 'technical']):
            tasks["onchain"] = call_mcp("crypto-onchain-server", "run_patterns_only", {"symbol": symbol, "limit": 20})
        else:
            tasks["onchain"] = call_mcp("crypto-onchain-server", "run_onchain_pipeline", {"symbol": symbol, "run_steps": "all"})

    if "sentiment" in cats or "combined" in cats:
        # Smart ingest based on query type
        ingest_needed = any(word in query.lower() for word in ['news', 'sentiment', 'reddit', 'article', 'media'])
        if ingest_needed:
            ingest_result = await call_mcp("crypto-pipeline-server", "ingest_documents", {"days_back": 7})
            logger.info("Ingested documents for sentiment analysis")
        
        # Dynamic query formulation
        sentiment_query = f"Current market sentiment and news about {symbol}"
        if "pattern" in query.lower() or "technical" in query.lower():
            sentiment_query += " including technical analysis patterns impact"
        
        tasks["sentiment"] = call_mcp("crypto-pipeline-server", "analyze_with_sources", {
            "query": sentiment_query, 
            "k": 5,
            "include_sources": True
        })

    # Execute all tools in parallel
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

    # Add SHAP explanations if forecast data available
    if "forecast" in data and df is not None and "error" not in data["forecast"]:
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

    async def run(self, query: str, symbol: str = "BTC", days: int = 30) -> Dict[str, Any]:
        """Main agent execution with dynamic LLM routing."""
        cache_key = f"v2_agent:{symbol}:{hashlib.sha256(query.encode()).hexdigest()}:{days}"
        
        if cached := cache.get_json(cache_key):
            logger.info("Returning cached response")
            return cached

        logger.info(f"Processing query: '{query}' for {symbol}")

        # 1. Classify query to determine optimal processing
        classification = await self.classifier.classify(query)
        qtype = classification["qtype"]
        categories = classification["categories"]
        
        logger.info(f"Query classified as: {qtype}, categories: {categories}")

        # 2. Load data for analysis
        preprocessor = CoinPreprocessor()
        df = await asyncio.to_thread(preprocessor.load_features_series, symbol)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        # 3. Route to appropriate tools
        data = await route_tools(classification, symbol, df, query)

        # 4. Execute backtest if requested
        backtest = None
        if "backtest" in categories or qtype == "long_context":
            logger.info(f"Running backtest for {days} days")
            backtest = await self.backtester.run_hybrid_backtest( 
                symbol, days, categories
            )
            data["backtest"] = backtest

        # 5. Generate trading signals for relevant queries
        strategy = None
        if qtype in ["combined", "reasoning", "patterns"]:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            strategy = hybrid_signal(
                df, 
                data.get("forecast", {}), 
                data.get("sentiment", {}), 
                data.get("onchain", {}), 
                symbol, query_hash, days
            )
            data["strategy"] = strategy

        # 6. Select optimal LLM based on query type
        provider, model, temperature = LLM_REGISTRY.get(qtype, LLM_REGISTRY["combined"])
        logger.info(f"Using {provider}/{model} for {qtype} query (temp: {temperature})")

        # 7. Construct prompt and synthesize response
        prompt_text = construct_prompt(query, data, qtype, self.current_date)
        response = await synthesize(provider, model, temperature, prompt_text)

        # 8. Compile final result
        result = {
            "symbol": symbol,
            "query_type": qtype,
            "categories": categories,
            "llm_used": f"{provider}/{model}",
            "data_insights": {
                "forecast_available": "forecast" in data,
                "sentiment_available": "sentiment" in data, 
                "onchain_available": "onchain" in data,
                "backtest_available": "backtest" in data,
                "strategy_available": "strategy" in data
            },
            "response": response,
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }

        # Cache the result
        cache.set_json(cache_key, result)
        logger.info(f"Successfully processed query and cached result")
        
        return result

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
        print(f"\nRESPONSE:\n{result['response']}")
        print("\n" + "="*80)
        
        if 'strategy' in result.get('data_insights', {}) and result['data_insights']['strategy_available']:
            print(f"\nTRADING STRATEGY: {result.get('data', {}).get('strategy', {})}")
            
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
