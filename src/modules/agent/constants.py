# src/modules/agent/constants.py
"""
Agent constants and LLM registry configuration.

Defines query type to LLM provider mappings and available
query categories for the crypto analytics agent.
"""

LLM_REGISTRY = {
    "real_time": ("groq", "llama-3.3-70b-versatile", 0.1),
    "reasoning": ("openrouter", "deepseek/deepseek-chat", 0.3), 
    "long_context": ("google", "gemini-2.5-flash", 0.7),
    "combined": ("groq", "mixtral-8x7b-32768", 0.2),
    "patterns": ("groq", "llama-3.3-70b-versatile", 0.1)
}

QUERY_CATEGORIES = ["forecast", "onchain", "sentiment", "combined", "backtest", "patterns"]