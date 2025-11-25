import asyncio
import logging
from typing import Dict, Any
from pathlib import Path
import os
import sys

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

# Ensure project root is on sys.path for imports like `core` and `data`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.onchain_client import setup_mlflow
from data.storage.models import TASignal as TASignalModel, OnchainMetric as OnchainMetricModel
from sqlalchemy import select
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
            def execute_metrics():
                with get_timescale_db() as ts_db:
                    def latest(metric_name: str):
                        stmt = (
                            select(OnchainMetricModel.value)
                            .where(
                                OnchainMetricModel.chain == chain,
                                OnchainMetricModel.metric == metric_name,
                                OnchainMetricModel.raw.op("->>")("window") == time_window,
                            )
                            .order_by(OnchainMetricModel.time.desc())
                            .limit(1)
                        )
                        value = ts_db.execute(stmt).scalar_one_or_none()
                        return float(value) if value is not None else None

                    flows = {
                        "exchange_inflow_usd": latest("exchange_inflow_usd"),
                        "exchange_outflow_usd": latest("exchange_outflow_usd"),
                        "net_flow_usd": latest("net_flow_usd"),
                        "exchange_flow_ratio": latest("exchange_flow_ratio"),
                        "flow_trend_24h": latest("flow_trend_24h"),
                    }

                    whales = {
                        "whale_count": latest("whale_count"),
                        "total_whale_volume_usd": latest("total_whale_volume_usd"),
                        "avg_whale_tx_size_usd": latest("avg_whale_tx_size_usd"),
                        "whale_exchange_inflow": latest("whale_exchange_inflow"),
                        "whale_exchange_outflow": latest("whale_exchange_outflow"),
                        "whale_exchange_ratio": latest("whale_exchange_ratio"),
                        "unique_whale_addresses": latest("unique_whale_addresses"),
                    }

                    aggregated_metrics = {
                        "market_pressure_index": latest("market_pressure_index"),
                        "whale_to_exchange_ratio": latest("whale_to_exchange_ratio"),
                        "price_whale_corr_7d": latest("price_whale_corr_7d"),
                        "flow_trend_7d": latest("flow_trend_7d"),
                        "price_change_pct": latest("price_change_pct"),
                    }

                # Derive market_bias on the fly from stored metrics.
                net_flow = flows.get("net_flow_usd") or 0.0
                price_change = aggregated_metrics.get("price_change_pct") or 0.0
                whale_ratio = aggregated_metrics.get("whale_to_exchange_ratio") or 0.0

                flow_bias = 1 if net_flow > 0 else -1 if net_flow < 0 else 0
                price_bias = 1 if price_change > 0 else -1 if price_change < 0 else 0
                ratio_bias = 1 - whale_ratio

                bias_score = (flow_bias * 0.4) + (price_bias * 0.3) + (ratio_bias * 0.3)
                if bias_score > 0.3:
                    market_bias = "bullish"
                elif bias_score < -0.3:
                    market_bias = "bearish"
                else:
                    market_bias = "neutral"

                aggregated = {
                    **aggregated_metrics,
                    "market_bias": market_bias,
                }

                return {
                    "flows": flows,
                    "whales": whales,
                    "aggregated": aggregated,
                    "errors": [],
                }

            result = await asyncio.to_thread(execute_metrics)
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
        exchange = input_data.get("exchange", "binance")
        interval = input_data.get("interval", "1d")
        limit = input_data.get("limit", 50)

        try:
            def execute_patterns():
                with get_timescale_db() as ts_db:
                    stmt = (
                        select(TASignalModel)
                        .where(
                            TASignalModel.exchange == exchange,
                            TASignalModel.interval == interval,
                        )
                        .order_by(TASignalModel.symbol)
                        .limit(limit)
                    )
                    rows = ts_db.execute(stmt).scalars().all()

                patterns = {
                    row.symbol: {
                        "exchange": row.exchange,
                        "interval": row.interval,
                        "timestamp": row.time.isoformat() if row.time else None,
                        "signal": row.signal,
                        "rsi": float(row.rsi) if row.rsi is not None else None,
                        "macd_hist": float(row.macd_hist) if row.macd_hist is not None else None,
                        "pattern": row.pattern,
                    }
                    for row in rows
                }

                if patterns:
                    return {"patterns": patterns, "status": "success"}
                else:
                    return {"patterns": {}, "status": "no_data"}

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
