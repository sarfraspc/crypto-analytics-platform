import asyncio
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Dict, Any

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from core.logging_config import setup_logging
from utils.cache import RedisCache
from utils.mcp_utils import AsyncStdioWrapper
from core.exceptions import CryptoAnalyticsError, APIError
from modules.agent.agent_client import orchestrate_query  

setup_logging()
logger = logging.getLogger(__name__)

cache = RedisCache(expire_seconds=1800)  # Shared with agent for consistency

class AgentMCP:
    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        logger.info("Agent MCP initialized successfully")
        self.is_initialized = True

    def _validate_symbol(self, symbol: str) -> str:
        if not symbol or not isinstance(symbol, str) or len(symbol) > 10 or not symbol.isalnum():
            raise ValueError(f"Invalid symbol: {symbol}. Must be a valid ticker like 'BTC', 'ETH'.")
        return symbol.upper()

    async def get_agent_insight(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")

        args = request.arguments or {}
        symbol = self._validate_symbol(args.get("symbol", ""))
        question = args.get("question", "") or ""  # Sanitize: strip and limit length
        if len(question) > 1000:
            question = question[:1000] + "... [truncated]"

        options_raw = args.get("options", {})
        options = {
            "k_docs": options_raw.get("k_docs", 5),
            "window": options_raw.get("window", "24h"),
            "horizon": options_raw.get("horizon", 7),
            "force_llm": options_raw.get("force_llm"),  # None or 'real_time|reasoning|long_context'
        }
        no_cache = options_raw.get("no_cache", False)

        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] get_agent_insight called: symbol={symbol}, question={question[:100]}..., options={options}, no_cache={no_cache}")

        start_time = datetime.now()
        cache_hit = False
        warnings = []

        try:
            # Compute cache key including options
            opt_str = json.dumps(options, sort_keys=True)
            q_hash = hashlib.sha256(question.encode()).hexdigest()
            opt_hash = hashlib.sha256(opt_str.encode()).hexdigest()
            cache_key = f"agent:{symbol.lower()}:{opt_hash}:{q_hash}"

            result = None
            if not no_cache:
                if cached := cache.get_json(cache_key):
                    result = cached
                    cache_hit = True
                    logger.info(f"[{request_id}] Cache hit for {cache_key}")
                else:
                    logger.info(f"[{request_id}] Cache miss, invoking agent")

            if result is None:
                # Invoke agent (assumes orchestrate_query accepts these params)
                agent_result = await orchestrate_query(
                    symbol=symbol,
                    question=question,
                    options=options,
                    no_cache=no_cache,
                    force_query_type=options["force_llm"]
                )
                # Agent returns base result without timestamp/cache_hit/warnings/schema_version
                result = agent_result

                # Post-process: add warnings for None fields (partial results)
                if result.get("forecast") is None:
                    warnings.append("Forecast data unavailable (source timeout or error)")
                if result.get("sentiment") is None:
                    warnings.append("Sentiment data unavailable (source timeout or error)")
                if result.get("rag") is None:
                    warnings.append("RAG context unavailable (source timeout or error)")
                if result.get("onchain") is None:
                    warnings.append("On-chain data unavailable (source timeout or error)")
                if result.get("backtest") is None and options["horizon"] > 7:  # e.g., when expected
                    warnings.append("Backtest not performed (query not suitable or data issue)")

                result["warnings"] = warnings
                # Cap arrays/sizes for UI safety
                if "rag" in result and result["rag"].get("contexts"):
                    result["rag"]["contexts"] = result["rag"]["contexts"][:10]  # max 10
                if len(result["final_answer"]) > 5000:
                    result["final_answer"] = result["final_answer"][:5000] + "... [truncated]"

                if not no_cache:
                    cache.set_json(cache_key, result)
                    logger.info(f"[{request_id}] Agent result cached under {cache_key}")

            # Standardize output
            output = {
                "schema_version": "1.0",
                "symbol": result["symbol"],
                "query_type": result["query_type"],
                "llm_used": result["llm_used"],
                "timestamp": datetime.now().isoformat(),
                "cache_hit": cache_hit,
                **{k: v for k, v in result.items() if k not in ["symbol", "query_type", "llm_used", "warnings"]},
                "warnings": warnings if warnings else None,  # None if empty
            }

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(f"[{request_id}] Completed: query_type={output['query_type']}, llm={output['llm_used']}, duration_ms={duration_ms}, cache_hit={cache_hit}, warnings={len(warnings)}")

            output_str = json.dumps(output, default=str, indent=2)
            return CallToolResult(
                content=[TextContent(text=f"Agent Insight Result:\n\n{output_str}")]
            )

        except ValueError as e:
            logger.warning(f"[{request_id}] Input validation error: {e}")
            raise CryptoAnalyticsError(str(e))  # Maps to 400
        except asyncio.TimeoutError:
            err_msg = "Agent orchestration timed out (15-30s limit); please retry."
            logger.error(f"[{request_id}] Timeout error")
            raise APIError(err_msg, 504)
        except Exception as e:
            err_msg = f"Internal error: {str(e)}"
            logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
            raise CryptoAnalyticsError(err_msg)  # Maps to 500 or 204 if no data

async def main():
    server = Server("crypto-agent-server")
    mcp = AgentMCP()
    await mcp.initialize()

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="get_agent_insight",
                description="Get comprehensive crypto insight: forecast, sentiment, on-chain, RAG context, backtest, and synthesized answer",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Asset ticker (e.g., 'BTC', 'ETH', 'SOL')",
                            "examples": ["BTC", "ETH"]
                        },
                        "question": {
                            "type": "string",
                            "description": "Natural language query (e.g., 'why did price drop this week?')",
                            "default": ""
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "k_docs": {"type": "integer", "default": 5, "description": "RAG top-K contexts"},
                                "window": {"type": "string", "default": "24h", "description": "On-chain lookback window", "enum": ["1h", "24h"]},
                                "horizon": {"type": "integer", "default": 7, "description": "Forecast horizon (days)"},
                                "force_llm": {"type": "string", "enum": ["real_time", "reasoning", "long_context"], "description": "Override query classifier"},
                                "no_cache": {"type": "boolean", "default": False, "description": "Bypass cache for debugging"}
                            },
                            "default": {}
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            # Optional future tools
            # Tool(name="get_backtest_report", ...),
            # Tool(name="explain_decision", ...),
            # Tool(name="agent_health", ...),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        if name == "get_agent_insight":
            return await mcp.get_agent_insight(CallToolRequest(name=name, arguments=arguments))
        else:
            raise Exception(f"Unknown tool: {name}")

    @server.list_resources()
    async def list_resources():
        return []

    @server.read_resource()
    async def read_resource(name: str):
        raise Exception(f"Unknown resource: {name}")

    read_stream = AsyncStdioWrapper(sys.stdin.buffer, mode='r')
    write_stream = AsyncStdioWrapper(sys.stdout.buffer, mode='w')
    init_options = {"name": "crypto-agent-server"}

    try:
        await server.run(read_stream, write_stream, init_options)
    except Exception:
        logger.exception("Server.run failed")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent server stopped by user")