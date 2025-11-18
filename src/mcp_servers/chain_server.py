import asyncio
import logging
from typing import Dict, Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    Tool,
    TextContent,
)
from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.onchain_client import setup_mlflow
from data.market_client import get_top_symbols, run_ta_patterns
from modules.onchain.metrics.pipeline import run_onchain_metrics
import json

setup_logging()
logger = logging.getLogger(__name__)

class OnchainMCP:
    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        try:
            await asyncio.to_thread(setup_mlflow)
            logger.info("Onchain MCP initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.exception("Onchain MCP initialization failed")
            raise e

    async def run_metrics_only(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        chain = input_data.get("chain", "ethereum")
        time_window = input_data.get("window", "24h")

        try:
            result = await asyncio.to_thread(run_onchain_metrics, chain, time_window)
            # Emit pure JSON so upstream callers receive a structured payload,
            # avoiding any \"Metrics Result:\" wrappers that break dict parsing.
            result_str = json.dumps(result, default=str)
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=result_str,
                )]
            )
        except Exception as e:
            err = f"Error: {type(e).__name__} - {str(e)}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=err)]
            )

    async def run_patterns_only(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        exchange = input_data.get('exchange', 'binance')
        interval = input_data.get('interval', '1d')
        limit = input_data.get('limit', 50)

        try:
            def execute_patterns():
                # Use the timescale DB so we select symbols
                # directly from the OHLCV table for TA patterns.
                with get_timescale_db() as ts_db:
                    symbols = get_top_symbols(ts_db, limit=limit)
                return run_ta_patterns(symbols, exchange=exchange, interval=interval)

            result = await asyncio.to_thread(execute_patterns)
            # Emit pure JSON so downstream parsers see a clean dict.
            result_str = json.dumps(result, default=str)
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=result_str,
                )]
            )
        except Exception as e:
            err = f"Error: {type(e).__name__} - {str(e)}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=err)]
            )

# === START OF STRUCTURAL FIX ===
# Definitions must stay at module scope for decorator registration.
server = Server("crypto-onchain-server")
mcp = OnchainMCP()


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="run_metrics_only",
            description="Run only metrics computation (exchange flows, whale summaries, market pressure index)",
            inputSchema={
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "default": "ethereum"},
                    "window": {"type": "string", "default": "24h", "enum": ["1h", "24h"]},
                },
                "required": [],
            },
        ),
        Tool(
            name="run_patterns_only",
            description="Run only TA patterns generation (RSI, MACD, candlesticks) for top symbols",
            inputSchema={
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "default": "binance"},
                    "interval": {"type": "string", "default": "1d"},
                    "limit": {"type": "integer", "default": 50}
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name == "run_metrics_only":
        return await mcp.run_metrics_only(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    if name == "run_patterns_only":
        return await mcp.run_patterns_only(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    else:
        raise Exception(f"Unknown tool: {name}")


@server.list_resources()
async def list_resources():
    return []


@server.read_resource()
async def read_resource(name: str):
    raise Exception(f"Unknown resource: {name}")


async def main():
    """This is the main coroutine that runs the server."""
    await mcp.initialize()
    logger.info(f"Starting {server.name}...")
    init_options = InitializationOptions(
        server_name=server.name,
        server_version="1.0.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
    logger.info(f"Init options: {init_options}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)
# === END OF STRUCTURAL FIX ===


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Onchain server stopped by user")
