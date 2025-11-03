import asyncio
import logging
import sys
from typing import Dict, Any
from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent
from core.database import get_timescale_db, get_metadata_db
from core.logging_config import setup_logging
from modules.onchain.pipeline import run_onchain_pipeline, setup_mlflow, get_top_symbols, run_ta_patterns
from modules.onchain.metrics.pipeline import run_onchain_metrics
from utils.mcp_utils import AsyncStdioWrapper
import json

setup_logging()
logger = logging.getLogger(__name__)

class OnchainMCP:
    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        try:
            setup_mlflow()
            logger.info("Onchain MCP initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.exception("Onchain MCP initialization failed")
            raise e

    async def run_pipeline(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")
        input_data = request.arguments or {}
        chain = input_data.get('chain', 'ethereum')
        batch_size = input_data.get('batch_size', 100)
        threshold_usd = input_data.get('threshold_usd', 500000.0)
        time_window = input_data.get('window', '24h')
        symbol = input_data.get('symbol', 'BTC')
        run_steps_str = input_data.get('run_steps', 'all')
        run_steps = run_steps_str.split(',') if run_steps_str != 'all' else None

        try:
            with get_timescale_db() as db:
                result = run_onchain_pipeline(
                    db=db,
                    chain=chain,
                    batch_size=batch_size,
                    threshold_usd=threshold_usd,
                    time_window=time_window,
                    symbol=symbol,
                    run_steps=run_steps
                )
            result_str = json.dumps(result, default=str, indent=2) 
            return CallToolResult(
                content=[TextContent(
                    text=f"Onchain Pipeline Result:\n\n{result_str}"
                )]
            )
        except Exception as e:
            err = f"Error: {type(e).__name__} - {str(e)}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                content=[TextContent(text=err)]
            )

    async def run_ingestion_only(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")
        input_data = request.arguments or {}
        chain = input_data.get('chain', 'ethereum')
        batch_size = input_data.get('batch_size', 100)
        threshold_usd = input_data.get('threshold_usd', 500000.0)

        try:
            with get_timescale_db() as db:
                result = run_onchain_pipeline(
                    db=db,
                    chain=chain,
                    batch_size=batch_size,
                    threshold_usd=threshold_usd,
                    run_steps=['ingestion']
                )
            result_str = json.dumps(result, default=str, indent=2)
            return CallToolResult(
                content=[TextContent(
                    text=f"Ingestion Result:\n\n{result_str}"
                )]
            )
        except Exception as e:
            err = f"Error: {type(e).__name__} - {str(e)}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                content=[TextContent(text=err)]
            )

    async def run_metrics_only(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")
        input_data = request.arguments or {}
        chain = input_data.get('chain', 'ethereum')
        time_window = input_data.get('window', '24h')
        symbol = input_data.get('symbol', 'BTC')

        try:
            result = run_onchain_metrics(chain, time_window, symbol)
            result_str = json.dumps(result, default=str, indent=2)
            return CallToolResult(
                content=[TextContent(
                    text=f"Metrics Result:\n\n{result_str}"
                )]
            )
        except Exception as e:
            err = f"Error: {type(e).__name__} - {str(e)}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                content=[TextContent(text=err)]
            )

    async def run_patterns_only(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")
        input_data = request.arguments or {}
        exchange = input_data.get('exchange', 'binance')
        interval = input_data.get('interval', '1d')
        limit = input_data.get('limit', 50)

        try:
            with get_metadata_db() as meta_db:
                symbols = get_top_symbols(meta_db, limit=limit)
            result = run_ta_patterns(symbols, exchange=exchange, interval=interval)
            result_str = json.dumps(result, default=str, indent=2)
            return CallToolResult(
                content=[TextContent(
                    text=f"Patterns Result:\n\n{result_str}"
                )]
            )
        except Exception as e:
            err = f"Error: {type(e).__name__} - {str(e)}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                content=[TextContent(text=err)]
            )

async def main():
    server = Server("crypto-onchain-server")
    mcp = OnchainMCP()
    await mcp.initialize()

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="run_onchain_pipeline",
                description="Run the full on-chain analytics pipeline: whale alerts ingestion + exchange flows + TA patterns (Ethereum default)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chain": {"type": "string", "default": "ethereum"},
                        "batch_size": {"type": "integer", "default": 100},
                        "threshold_usd": {"type": "number", "default": 500000.0},
                        "window": {"type": "string", "default": "24h", "enum": ["1h", "24h"]},
                        "symbol": {"type": "string", "default": "BTC"},
                        "run_steps": {"type": "string", "description": "Comma-separated steps (e.g., 'ingestion,metrics') or 'all'"}
                    },
                    "required": []
                }
            ),
            Tool(
                name="run_ingestion_only",
                description="Run only whale alert ingestion (fetches and processes recent transfers > threshold USD)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chain": {"type": "string", "default": "ethereum"},
                        "batch_size": {"type": "integer", "default": 100},
                        "threshold_usd": {"type": "number", "default": 500000.0}
                    },
                    "required": []
                }
            ),
            Tool(
                name="run_metrics_only",
                description="Run only metrics computation (exchange flows, whale summaries, market pressure index)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chain": {"type": "string", "default": "ethereum"},
                        "window": {"type": "string", "default": "24h", "enum": ["1h", "24h"]},
                        "symbol": {"type": "string", "default": "BTC"}
                    },
                    "required": []
                }
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
        if name == "run_onchain_pipeline":
            return await mcp.run_pipeline(CallToolRequest(name=name, arguments=arguments))
        elif name == "run_ingestion_only":
            return await mcp.run_ingestion_only(CallToolRequest(name=name, arguments=arguments))
        elif name == "run_metrics_only":
            return await mcp.run_metrics_only(CallToolRequest(name=name, arguments=arguments))
        elif name == "run_patterns_only":
            return await mcp.run_patterns_only(CallToolRequest(name=name, arguments=arguments))
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
    init_options = {"name": "crypto-onchain-server"}

    try:
        await server.run(read_stream, write_stream, init_options)
    except Exception:
        logger.exception("Server.run failed")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Onchain server stopped by user")