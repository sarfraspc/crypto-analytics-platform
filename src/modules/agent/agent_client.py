import asyncio
import hashlib
import json
import logging
from typing import Dict, Any

import httpx
import google.generativeai as genai

from utils.cache import RedisCache
from core.logging_config import setup_logging
from core.config import settings
from modules.agent.prompts import construct_full_prompt
from modules.agent.backtester import run_backtest
from modules.agent.strategy_utils import get_hybrid_signals
from modules.agent.query_classifier import QueryClassifier
from modules.forecasting.data.preprocess_coin import CoinPreprocessor

# MCP SDK v1.17.0 imports 
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters  

setup_logging()
logger = logging.getLogger(__name__)

cache = RedisCache(expire_seconds=1800)

# VALID MODELS (tested Nov 2025)
LLM_REGISTRY = {
    "real_time": ("groq", "llama-3.3-70b-versatile"),          # Fast & cheap
    "reasoning": ("openrouter", "deepseek/deepseek-chat"),     # Deep reasoning
    "long_context": ("google", "gemini-2.5-flash"),            # Long context
}

SERVER_COMMANDS: Dict[str, StdioServerParameters] = {
    "crypto-sarimax-server": StdioServerParameters(
        command="python",
        args=["-m", "src.modules.forecasting.sarimax_mcp"],
    ),
    "crypto-sentiment-server": StdioServerParameters(
        command="python",
        args=["-m", "src.modules.sentiment.sentiment_mcp"],
    ),
    "crypto-rag-server": StdioServerParameters(
        command="python",
        args=["-m", "src.modules.sentiment.rag_mcp"],
    ),
    "crypto-onchain-server": StdioServerParameters(
        command="python",
        args=["-m", "src.modules.onchain.onchain_mcp"],
    ),
}

async def _parse_mcp_result(raw_result) -> dict:
    text_parts = []
    for content in raw_result.content or []:
        if getattr(content, "type", None) == "text":
            text_parts.append(content.text)
    full_text = "\n".join(text_parts)
    try:
        if full_text.strip().startswith(("{", "[")):
            return json.loads(full_text)
        return {"raw_text": full_text}
    except Exception as e:
        logger.warning(f"MCP parse error: {e}")
        return {"raw_text": full_text, "parse_error": True}

async def call_mcp_tool(server_name: str, tool: str, args: dict) -> dict:
    cache_key = f"mcp:{server_name}:{tool}:{hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()}"
    if cached := cache.get_json(cache_key):
        logger.info(f"[Cache HIT] {cache_key}")
        return cached

    params = SERVER_COMMANDS[server_name]
    logger.info(f"[Cache MISS] Spawning {server_name} → {tool}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            raw_result = await session.call_tool(tool, arguments=args)
            parsed = await _parse_mcp_result(raw_result)
            cache.set_json(cache_key, parsed)
            return parsed

async def get_specialist_data(symbol: str, question: str, classification) -> Dict[str, Any]:
    tasks = {}

    if classification.requires_forecast:
        tasks["forecast"] = call_mcp_tool("crypto-sarimax-server", "forecast_sarimax", {"symbol": symbol, "horizon": 7})

    if classification.requires_rag:
        rag_query = f"Recent events and news impacting {symbol}: {question}"
        tasks["rag"] = call_mcp_tool("crypto-rag-server", "query_rag", {"query": rag_query, "k": 6})

    if classification.requires_onchain:
        tasks["onchain"] = call_mcp_tool("crypto-onchain-server", "run_metrics_only", {"symbol": symbol, "window": "24h"})

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    data = {}
    for key, res in zip(tasks.keys(), results):
        data[key] = None if isinstance(res, Exception) else res

    sentiment = {"overall": "neutral", "bullish_score": 0.5, "source_count": 0}
    if data.get("rag"):
        contexts = [c.get("content", "") for c in data["rag"].get("contexts", [])][:10]
        if contexts:
            sentiment_raw = await call_mcp_tool("crypto-sentiment-server", "analyze_sentiment_batch", {"texts": contexts})
            scores = [r.get("bullish_score", 0) or r.get("BULLISH", 0) for r in sentiment_raw.get("results", [])]
            avg = sum(scores) / len(scores) if scores else 0.5
            overall = "bullish" if avg > 0.6 else "bearish" if avg < 0.4 else "neutral"
            sentiment = {"overall": overall, "bullish_score": avg, "source_count": len(contexts)}
    data["sentiment"] = sentiment

    return data

async def get_llm_response(provider: str, model: str, prompt: str) -> str:
    provider = provider.lower()
    temperature = 0.3 if "real_time" in prompt.lower() else 0.7

    if provider == "google":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY missing")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=4096)
        )
        return response.text

    if provider == "groq":
        api_key = settings.GROQ_API_KEY
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY
        base_url = "https://openrouter.ai/api/v1"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY missing")

    payload = {
        "model": model,  
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:  
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def orchestrate_query(symbol: str, question: str) -> Dict[str, Any]:
    cache_key = f"agent:{symbol.lower()}:{hashlib.sha256(question.encode()).hexdigest()}"
    if cached := cache.get_json(cache_key):
        logger.info(f"[Cache HIT] Full query {cache_key}")
        return cached

    logger.info(f"[Agent] Processing: {symbol} - {question}")
    classification = QueryClassifier().classify_query(question)
    logger.info(f"[Classification] {classification.query_type} | Sources: {vars(classification)}")

    data = await get_specialist_data(symbol, question, classification)

    backtest = None
    if classification.query_type == "long_context" or any(w in question.lower() for w in ["backtest", "strategy", "performance"]):
        pre = CoinPreprocessor()
        df = await asyncio.to_thread(pre.load_features_series, symbol)
        signals = get_hybrid_signals(df, data.get("forecast") or {}, data.get("sentiment") or {}, data.get("onchain") or {})
        backtest_raw = await asyncio.to_thread(run_backtest, symbol, signals, df)
        backtest = {"strategy_name": "hybrid", **backtest_raw}
        data["backtest"] = backtest

    provider, model = LLM_REGISTRY.get(classification.query_type, LLM_REGISTRY["reasoning"])
    full_prompt = construct_full_prompt(question, data, classification.query_type)
    final_answer = await get_llm_response(provider, model, full_prompt)

    result = {
        "symbol": symbol.upper(),
        "query_type": classification.query_type,
        "llm_used": f"{provider}/{model}",
        "forecast": data.get("forecast"),
        "sentiment": data.get("sentiment"),
        "rag": data.get("rag"),
        "onchain": data.get("onchain"),
        "backtest": backtest,
        "final_answer": final_answer.strip(),
    }

    cache.set_json(cache_key, result)
    logger.info(f"[Agent] Completed & cached {cache_key}")
    return result

async def main():
    queries = [
        ("BTC", "What is BTC price now?"),                          # Groq – instant fact
        ("BTC", "Explain why BTC dropped yesterday"),               # OpenRouter – deep reasoning
        ("ETH", "Full 30-day report on ETH with hybrid backtest"),  # Gemini – long report
    ]
    for symbol, q in queries:
        print(f"\n{'='*80}\n{ q.upper() }\n{'='*80}")
        result = await orchestrate_query(symbol, q)
        print("LLM:", result["llm_used"])
        print("ANSWER:", result["final_answer"][:1000] + "...\n")

if __name__ == "__main__":
    asyncio.run(main())