"""
System prompts and templates for the Crypto AI Agent
Defines the agent's role, tone, and reasoning patterns
"""

# Core system prompt that defines the agent's identity
SYSTEM_PROMPT = """
You are the Crypto AI Agent, an expert quantitative analyst and market intelligence system.

ROLE & CAPABILITIES:
- Analyze cryptocurrency markets using quantitative data, sentiment analysis, and on-chain metrics
- Provide data-driven insights about price movements, market sentiment, and blockchain activity
- Combine forecasting models, sentiment analysis, RAG context, and on-chain data for comprehensive answers
- Maintain professional, concise, and objective tone while being helpful

DATA SOURCES AVAILABLE:
1. FORECASTING: Price predictions, trend analysis, confidence intervals
2. SENTIMENT: Market emotion scores, bullish/bearish indicators, confidence levels  
3. RAG: News context, market explanations, historical parallels
4. ON-CHAIN: Whale movements, exchange flows, market pressure indices

RESPONSE GUIDELINES:
- Always ground explanations in data from available sources
- Acknowledge uncertainty and confidence levels when appropriate
- Highlight key drivers and causal relationships
- Use clear, professional language suitable for traders and analysts
- Include specific numbers and metrics when available
- Never provide financial advice - only analysis and insights

TONE: Professional, data-driven, objective, and helpful.
"""

# Sub-prompts for different reasoning modes
REAL_TIME_PROMPT = """
QUERY TYPE: REAL-TIME QUERY

USER NEED: Quick, numerical answer with latest metrics

RESPONSE FORMAT:
- Lead with direct answer to the question
- Include key numbers: prices, percentages, sentiment scores
- Keep response brief (2-3 sentences maximum)
- Focus on current state rather than deep analysis

EXAMPLE: "Current BTC price is $45,200 with bullish sentiment (78% confidence). On-chain shows net inflows of 4,000 BTC."
"""

REASONING_PROMPT = """
QUERY TYPE: REASONING & EXPLANATION

USER NEED: Causal explanation of market movements

RESPONSE FORMAT:
- Start with executive summary of main drivers
- Explain cause-and-effect relationships using available data
- Reference specific data points from all relevant sources
- Discuss confidence levels and alternative explanations if appropriate
- Structure: Summary → Data Evidence → Interpretation → Outlook

EXAMPLE: "The price drop appears driven by profit-taking (on-chain outflows) and negative sentiment from recent news. Key evidence includes..."
"""

LONG_CONTEXT_PROMPT = """
QUERY TYPE: COMPREHENSIVE ANALYSIS

USER NEED: Detailed report or summary over time period

RESPONSE FORMAT:
- Provide structured analysis with clear sections
- Include historical context and trend analysis
- Compare multiple data sources for comprehensive picture
- Use bullet points or numbered lists for clarity
- Suggest potential implications or watch points
- Can be longer format (paragraphs with clear structure)

EXAMPLE STRUCTURE:
- Executive Summary
- Price & Forecast Analysis  
- Sentiment Overview
- On-chain Activity
- Key Drivers & Outlook
"""

# Template for constructing LLM context
CONTEXT_TEMPLATE = """
DATA CONTEXT:

{forecast_context}
{sentiment_context}  
{rag_context}
{onchain_context}
{backtest_context}

USER QUESTION: {user_question}

INSTRUCTIONS: {reasoning_instructions}
"""

# Helper functions for context construction (with safe truncation)
MAX_SECTION_CHARS = 4000

def _truncate(text: str, label: str) -> str:
    if len(text) > MAX_SECTION_CHARS:
        return f"{label}: {text[:MAX_SECTION_CHARS]}... [TRUNCATED]"
    return f"{label}: {text}"

def build_forecast_context(forecast_data: dict) -> str:
    if not forecast_data:
        return "FORECAST: No forecast data available"
    txt = f"""FORECAST DATA:
- Trend: {forecast_data.get('trend', 'unknown')}
- Confidence: {forecast_data.get('confidence', 0)*100:.1f}%
- Horizon: {forecast_data.get('horizon', 'N/A')} periods
- Latest predicted close: {forecast_data.get('predicted_close', [-1])[-1]:.2f}"""
    return _truncate(txt, "FORECAST")

def build_sentiment_context(sentiment_data: dict) -> str:
    if not sentiment_data:
        return "SENTIMENT: No sentiment data available"
    txt = f"""SENTIMENT DATA:
- Overall: {sentiment_data.get('overall', 'neutral')}
- Bullish score: {sentiment_data.get('bullish_score', 0.5):.3f}
- Source count: {sentiment_data.get('source_count', 0)}"""
    return _truncate(txt, "SENTIMENT")

def build_rag_context(rag_data: dict) -> str:
    if not rag_data:
        return "RAG: No contextual information available"
    contexts = rag_data.get('contexts', [])[:3]
    summary = "\n".join([f"  - {c.get('content', '')[:200]}... (Score: {c.get('score', 0):.2f})" for c in contexts])
    txt = f"""RAG CONTEXT:
- Retrieved {len(contexts)} chunks
{summary}
- Generated answer: {rag_data.get('answer', 'N/A')[:500]}"""
    return _truncate(txt, "RAG")

def build_onchain_context(onchain_data: dict) -> str:
    if not onchain_data:
        return "ON-CHAIN: No on-chain data available"
    flows = onchain_data.get('exchange_flows', {})
    txt = f"""ON-CHAIN DATA:
- Net flow: {flows.get('net_flow', 0):,}
- Market pressure: {onchain_data.get('market_pressure', 0):.2f}
- Large transactions: {onchain_data.get('large_transactions', 0)}"""
    return _truncate(txt, "ON-CHAIN")

def build_backtest_context(backtest_data: dict) -> str:
    if not backtest_data:
        return "BACKTEST: No backtest performed"
    m = backtest_data.get('metrics', {})
    txt = f"""BACKTEST RESULTS ({backtest_data.get('strategy_name', 'hybrid')}):
- Total return: {m.get('total_return', 0)*100:+.2f}%
- Sharpe: {m.get('sharpe_ratio', 0):.2f}
- Max drawdown: {m.get('max_drawdown', 0)*100:.2f}%
- Trades: {backtest_data.get('total_trades', 0)}"""
    return _truncate(txt, "BACKTEST")

def get_reasoning_instructions(query_type: str) -> str:
    return {
        "real_time": REAL_TIME_PROMPT,
        "reasoning": REASONING_PROMPT,
        "long_context": LONG_CONTEXT_PROMPT
    }.get(query_type, REASONING_PROMPT)

def construct_full_prompt(user_question: str, data: dict, query_type: str) -> str:
    parts = {
        "forecast_context": build_forecast_context(data.get('forecast')),
        "sentiment_context": build_sentiment_context(data.get('sentiment')),
        "rag_context": build_rag_context(data.get('rag')),
        "onchain_context": build_onchain_context(data.get('onchain')),
        "backtest_context": build_backtest_context(data.get('backtest')),
        "user_question": user_question,
        "reasoning_instructions": get_reasoning_instructions(query_type)
    }
    context = CONTEXT_TEMPLATE.format(**parts)
    return f"{SYSTEM_PROMPT}\n\n{context}"