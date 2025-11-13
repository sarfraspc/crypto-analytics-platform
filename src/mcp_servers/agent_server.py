import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from core.exceptions import CryptoAnalyticsError, APIError
from core.logging_config import setup_logging
from modules.agent.agent_client import orchestrate_query
from utils.cache import RedisCache

setup_logging()
logger = logging.getLogger(__name__)

cache = RedisCache(expire_seconds=1800)

MAX_QUESTION_LEN = 1000
SUPPORTED_WINDOWS = {"1h", "24h", "7d"}
DEFAULT_OPTIONS = {
    "k_docs": 5,
    "window": "24h",
    "horizon": 3,  # days
    "ingest_days": 7,
    "backtest_days": 30,
    "force_llm": None,
    "refresh_sentiment": False,
    "explain_forecast": False,
    "run_backtest": False,
}


class AgentMCP:
    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        logger.info("Agent MCP initialized successfully")
        self.is_initialized = True

    def _validate_symbol(self, symbol: str) -> str:
        if not symbol or not isinstance(symbol, str) or len(symbol) > 10 or not symbol.isalnum():
            raise ValueError("Invalid symbol. Provide an alphanumeric ticker like 'BTC' or 'ETH'.")
        return symbol.upper()

    def _normalize_question(self, question: Optional[str], symbol: str) -> str:
        sanitized = (question or "").strip() or f"{symbol} market overview"
        if len(sanitized) > MAX_QUESTION_LEN:
            sanitized = sanitized[:MAX_QUESTION_LEN] + "... [truncated]"
        return sanitized

    def _extract_options(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        raw_options = args.get("options", {})
        if isinstance(raw_options, str):
            try:
                raw_options = json.loads(raw_options)
            except json.JSONDecodeError:
                logger.warning("options payload is not valid JSON; falling back to defaults")
                raw_options = {}
        raw_options = raw_options or {}

        no_cache = bool(args.get("no_cache") or raw_options.pop("no_cache", False))

        merged = {**DEFAULT_OPTIONS, **raw_options}
        merged["k_docs"] = max(1, int(merged.get("k_docs", DEFAULT_OPTIONS["k_docs"])))
        merged["horizon"] = max(1, int(merged.get("horizon", DEFAULT_OPTIONS["horizon"])))
        merged["ingest_days"] = max(1, int(merged.get("ingest_days", DEFAULT_OPTIONS["ingest_days"])))
        merged["backtest_days"] = max(
            merged["horizon"],
            int(merged.get("backtest_days", DEFAULT_OPTIONS["backtest_days"])),
        )
        merged["refresh_sentiment"] = bool(merged.get("refresh_sentiment", False))
        merged["explain_forecast"] = bool(merged.get("explain_forecast", False))
        merged["run_backtest"] = bool(merged.get("run_backtest", False))

        window = str(merged.get("window", DEFAULT_OPTIONS["window"]))
        if window not in SUPPORTED_WINDOWS:
            logger.debug("Window '%s' not supported; defaulting to %s", window, DEFAULT_OPTIONS["window"])
            window = DEFAULT_OPTIONS["window"]
        merged["window"] = window

        force_llm = merged.get("force_llm")
        if force_llm not in {None, "real_time", "reasoning", "long_context", "combined", "patterns"}:
            logger.debug("Invalid force_llm override '%s' ignored", force_llm)
            merged["force_llm"] = None

        return merged, no_cache

    def _post_process(self, result: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        processed = dict(result)
        warnings = []

        if not processed.get("forecast"):
            warnings.append("Forecast data unavailable (source timeout or error)")
        if not processed.get("sentiment"):
            warnings.append("Sentiment data unavailable (source timeout or error)")
        if not processed.get("rag"):
            warnings.append("RAG context unavailable (source timeout or error)")
        if not processed.get("onchain"):
            warnings.append("On-chain data unavailable (source timeout or error)")
        if options.get("horizon", 7) >= 14 and not processed.get("backtest"):
            warnings.append("Backtest not performed (long-horizon query)")

        rag_block = processed.get("rag") or {}
        contexts = rag_block.get("contexts") or rag_block.get("sources")
        if isinstance(contexts, list) and len(contexts) > 10:
            rag_block = dict(rag_block)
            rag_block["contexts" if rag_block.get("contexts") else "sources"] = contexts[:10]
            processed["rag"] = rag_block

        final_answer = processed.get("final_answer") or processed.get("response") or ""
        if isinstance(final_answer, str) and len(final_answer) > 5000:
            processed["final_answer"] = final_answer[:5000] + "... [truncated]"

        processed["warnings"] = warnings or None
        processed.setdefault("timestamp", datetime.utcnow().isoformat())
        return processed

    async def get_agent_insight(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")

        args = request.arguments or {}
        symbol = self._validate_symbol(args.get("symbol", ""))
        question = self._normalize_question(args.get("question"), symbol)
        options, no_cache = self._extract_options(args)

        request_id = str(uuid.uuid4())
        logger.info(
            "[%s] get_agent_insight: symbol=%s horizon=%s window=%s k_docs=%s refresh_sentiment=%s explain_forecast=%s run_backtest=%s no_cache=%s",
            request_id,
            symbol,
            options.get("horizon"),
            options.get("window"),
            options.get("k_docs"),
            options.get("refresh_sentiment"),
            options.get("explain_forecast"),
            options.get("run_backtest"),
            no_cache,
        )

        start_time = datetime.now()
        cache_hit = False

        try:
            opt_str = json.dumps(options, sort_keys=True)
            q_hash = hashlib.sha256(question.encode()).hexdigest()
            opt_hash = hashlib.sha256(opt_str.encode()).hexdigest()
            cache_key = f"agent:{symbol.lower()}:{opt_hash}:{q_hash}"

            payload = None
            if not no_cache:
                payload = cache.get_json(cache_key)
                cache_hit = payload is not None
                if cache_hit:
                    logger.info("[%s] Cache hit for %s", request_id, cache_key)

            if payload is None:
                agent_result = await orchestrate_query(
                    symbol=symbol,
                    question=question,
                    options=options,
                    no_cache=no_cache,
                    force_query_type=options.get("force_llm"),
                )
                payload = self._post_process(agent_result, options)
                if not no_cache:
                    cache.set_json(cache_key, payload)
                    logger.info("[%s] Cached agent response under %s", request_id, cache_key)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            output = {
                "schema_version": "1.1",
                "request_id": request_id,
                "symbol": payload.get("symbol", symbol),
                "query_type": payload.get("query_type"),
                "llm_used": payload.get("llm_used"),
                "timestamp": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "cache_hit": cache_hit,
                **{k: v for k, v in payload.items() if k not in {"symbol", "query_type", "llm_used"}},
            }

            logger.info(
                "[%s] Completed: query_type=%s llm=%s duration_ms=%s cache_hit=%s",
                request_id,
                output.get("query_type"),
                output.get("llm_used"),
                duration_ms,
                cache_hit,
            )

            output_str = json.dumps(output, default=str, indent=2)
            return CallToolResult(content=[TextContent(text=f"Agent Insight Result:\n\n{output_str}")])

        except ValueError as exc:
            logger.warning("[%s] Input validation error: %s", request_id, exc)
            raise CryptoAnalyticsError(str(exc))
        except asyncio.TimeoutError:
            err_msg = "Agent orchestration timed out (15-30s limit); please retry."
            logger.error("[%s] Timeout error", request_id)
            raise APIError(err_msg, 504)
        except Exception as exc:
            err_msg = f"Internal error: {exc}"
            logger.error("[%s] Unexpected error", request_id, exc_info=True)
            raise CryptoAnalyticsError(err_msg)


server = Server("crypto-agent-server")
mcp = AgentMCP()
INIT_OPTIONS = {"name": "crypto-agent-server"}


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_agent_insight",
            description="Get unified crypto insight (forecast, sentiment, on-chain, RAG, backtest, LLM synthesis)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Asset ticker (e.g., BTC, ETH, SOL)",
                        "examples": ["BTC", "ETH"],
                    },
                    "question": {
                        "type": "string",
                        "description": "Natural language question (e.g., 'Why did BTC drop this week?')",
                        "default": "",
                    },
                    "no_cache": {
                        "type": "boolean",
                        "default": False,
                        "description": "Bypass agent + MCP caches for debugging",
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "k_docs": {"type": "integer", "default": 5, "description": "RAG top-K contexts"},
                            "window": {
                                "type": "string",
                                "default": "24h",
                                "enum": sorted(list(SUPPORTED_WINDOWS)),
                                "description": "On-chain lookback window",
                            },
                            "horizon": {"type": "integer", "default": 3, "description": "Forecast horizon (days)"},
                            "ingest_days": {"type": "integer", "default": 7, "description": "Sentiment ingest lookback"},
                            "backtest_days": {"type": "integer", "default": 30, "description": "Backtest window (days)"},
                            "force_llm": {
                                "type": "string",
                                "enum": ["real_time", "reasoning", "long_context", "combined", "patterns"],
                                "description": "Override classifier-selected LLM profile",
                            },
                            "no_cache": {
                                "type": "boolean",
                                "default": False,
                                "description": "Bypass orchestrator cache only",
                            },
                            "explain_forecast": {
                                "type": "boolean",
                                "default": False,
                                "description": "Run SHAP explanations for the SARIMAX forecast",
                            },
                            "run_backtest": {
                                "type": "boolean",
                                "default": False,
                                "description": "Force backtesting even if the classifier does not request it",
                            },
                            "refresh_sentiment": {
                                "type": "boolean",
                                "default": False,
                                "description": "Force a new sentiment/RAG ingestion cycle before answering",
                            },
                        },
                        "default": {},
                    },
                },
                "required": ["symbol"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name == "get_agent_insight":
        return await mcp.get_agent_insight(CallToolRequest(name=name, arguments=arguments))
    raise Exception(f"Unknown tool: {name}")


@server.list_resources()
async def list_resources():
    return []


@server.read_resource()
async def read_resource(name: str):
    raise Exception(f"Unknown resource: {name}")


async def main():
    await mcp.initialize()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, INIT_OPTIONS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent server stopped by user")
