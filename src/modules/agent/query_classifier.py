# src/modules/agent/query_classifier.py
"""
Smart query classifier with dynamic LLM routing.

Provides hybrid classification using rule-based patterns and LLM
inference to route queries to appropriate processing pipelines.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from core.config import settings
from modules.agent.constants import LLM_REGISTRY, QUERY_CATEGORIES

logger = logging.getLogger(__name__)

# Enhanced query pattern recognition
QUERY_PATTERNS = {
    "real_time": {
        "keywords": ['price', 'current', 'now', 'latest', 'today', 'live', 'what is', 'how much', 'score'],
        "examples": ["BTC price now?", "Current ETH value"]
    },
    "reasoning": {
        "keywords": ['why', 'explain', 'reason', 'cause', 'because', 'analysis', 'drop', 'rise'],
        "examples": ["Why did BTC drop?", "Explain market movement"]
    },
    "long_context": {
        "keywords": ['report', 'summary', 'overview', 'history', 'backtest', 'strategy', 'performance', 'detailed'],
        "examples": ["30-day BTC report", "Backtest trading strategy"]
    },
    "patterns": {
        "keywords": ['ta', 'rsi', 'macd', 'sma', 'candlestick', 'bollinger', 'pattern', 'technical'],
        "examples": ["BTC RSI analysis", "Show me TA patterns"]
    },
    "sentiment": {
        "keywords": ['sentiment', 'mood', 'emotion', 'news', 'reddit', 'social', 'media'],
        "examples": ["BTC sentiment analysis", "Market mood today"]
    }
}

CLASSIFICATION_PROMPT = """
Analyze this cryptocurrency query and classify it:

QUERY: {query}

Available Categories: {categories}

Query Patterns:
- Real-time: Quick facts, current prices, immediate data
- Reasoning: Explanations, cause/effect, market analysis  
- Long Context: Reports, summaries, historical analysis, backtests
- Patterns: Technical analysis, indicators, chart patterns
- Sentiment: Market mood, news analysis, social sentiment

Respond with JSON only:
{{
  "query_type": "most_suitable_type",
  "categories": ["relevant", "categories"],
  "reasoning": "brief explanation of classification"
}}
"""

class HybridClassifier:
    """
    Hybrid query classifier combining rules and LLM inference.

    Uses fast rule-based classification as primary method with
    LLM fallback for nuanced query understanding.
    """

    def __init__(self):
        """Initialize classifier with prompt template."""
        self.prompt = PromptTemplate.from_template(CLASSIFICATION_PROMPT)
        self.rule_cache = {}

    def _rule_based_classify(self, query: str) -> Dict[str, List[str]]:
        """Classify query using keyword pattern matching."""
        query_lower = query.lower()
        
        # Explicit match for real-time queries
        if any(word in query_lower for word in ['price', 'now', 'current']):
            qtype = "real_time"
            categories = ["forecast"]
            logger.info(f"Rule-based classification confidence: qtype={qtype}, categories={categories}")
            return {"qtype": qtype, "categories": categories}

        # Determine query type based on keyword matches
        qtype_matches = {}
        for pattern_type, pattern_data in QUERY_PATTERNS.items():
            match_count = sum(1 for keyword in pattern_data["keywords"] if keyword in query_lower)
            if match_count > 0:
                qtype_matches[pattern_type] = match_count
        
        # Prioritize qtype based on highest match count, or default to "reasoning"
        qtype = max(qtype_matches, key=qtype_matches.get) if qtype_matches else "reasoning"

        # Determine categories based on keywords
        categories = []
        if any(word in query_lower for word in ['forecast', 'predict', 'future', 'next']):
            categories.append("forecast")
        if any(word in query_lower for word in ['onchain', 'whale', 'flow', 'transfer']):
            categories.append("onchain")
        if any(word in query_lower for word in ['sentiment', 'news', 'reddit', 'social']):
            categories.append("sentiment")
        if any(word in query_lower for word in QUERY_PATTERNS["patterns"]["keywords"]):
            categories.append("patterns")
        if any(word in query_lower for word in ['backtest', 'simulate', 'historical']):
            categories.append("backtest")

        # Specific prioritization for "real_time" queries if no other categories are found
        if qtype == "real_time" and not categories:
            categories.append("forecast") # Assume real-time price queries often need forecast data

        # If multiple categories, or no specific category, default to "combined"
        if len(categories) > 1 or not categories:
            categories = ["combined"]
        
        logger.info(f"Rule-based classification confidence: qtype_matches={qtype_matches}, categories={categories}")
        return {"qtype": qtype, "categories": categories}

    async def _llm_classify(self, query: str) -> Dict[str, List[str]]:
        """Classify query using LLM for nuanced understanding."""
        try:
            # Use reasoning LLM for classification
            provider, model, temp = LLM_REGISTRY.get("reasoning", ("groq", "mixtral-8x7b-32768", 0.1))
            
            prompt_text = self.prompt.format(
                query=query,
                categories=", ".join(QUERY_CATEGORIES)
            )

            if provider == "groq" and settings.GROQ_API_KEY:
                llm = ChatGroq(
                    groq_api_key=settings.GROQ_API_KEY,
                    model_name=model,
                    temperature=temp
                )
                response = await llm.ainvoke(prompt_text)
                result_text = response.content
            else:
                # Fallback to HTTP-based call
                from httpx import AsyncClient
                
                if provider == "openrouter" and settings.OPENROUTER_API_KEY:
                    api_key = settings.OPENROUTER_API_KEY
                    base_url = "https://openrouter.ai/api/v1"
                else:
                    # Final fallback to Groq
                    provider, model, temp = "groq", "llama-3.3-70b-versatile", 0.1
                    api_key = settings.GROQ_API_KEY
                    base_url = "https://api.groq.com/openai/v1"

                async with AsyncClient() as client:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "temperature": temp,
                        "max_tokens": 500
                    }
                    response = await client.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                    response.raise_for_status()
                    result_text = response.json()["choices"][0]["message"]["content"]

            # Robustly parse JSON response
            parsed = {}
            try:
                # Strip markdown code fences if present
                json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result_text.strip()

                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"LLM response JSON decoding failed: {e}. Raw: {result_text[:200]}...")
            except Exception as e:
                logger.warning(f"Unexpected error during LLM response parsing: {e}. Raw: {result_text[:200]}...")
            
            return {
                "qtype": parsed.get("query_type", "reasoning"),
                "categories": parsed.get("categories", ["combined"]),
                "reasoning": parsed.get("reasoning", "")
            }

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            raise

    async def classify(self, query: str) -> Dict[str, List[str]]:
        """Classify query using LLM with rule-based fallback."""
        # Try LLM classification first for accuracy
        try:
            result = await self._llm_classify(query)
            logger.info(f"LLM classification: {result['qtype']} -> {result['categories']}")
            return result
        except Exception as e:
            # Fallback to rule-based classification
            logger.warning(f"LLM classification failed, using rules: {e}")
            result = self._rule_based_classify(query)
            logger.info(f"Rule-based classification: {result['qtype']} -> {result['categories']}")
            return result