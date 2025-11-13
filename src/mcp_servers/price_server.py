import asyncio
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

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

from modules.forecasting.models.sarimax import SarimaxModel
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.registry.mlflow_utils import log_model_params_and_metrics
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


MODEL_DIR = Path("src/modules/forecasting/models/saved")
SARIMAX_BASE_DIR = MODEL_DIR / "sarimax"

class SarimaxMCP:
    def __init__(self):
        self.coin_pre = CoinPreprocessor()
        self.default_model = None
        self.is_initialized = False

    async def initialize(self):
        try:
            self.default_model = SarimaxModel('BTC')
            path = SARIMAX_BASE_DIR / "sarimax_BTC.pkl"
            if path.exists():
                await asyncio.to_thread(self.default_model.load)
                logger.info("Loaded default SARIMAX for BTC")
            else:
                logger.warning("No default model found; will train on demand")
            self.is_initialized = True
        except Exception as e:
            logger.exception("Initialization failed")
            raise e

    async def get_model_for_symbol(self, symbol: str):
        path = SARIMAX_BASE_DIR / f"sarimax_{symbol}.pkl"
        model = SarimaxModel(symbol)
        if path.exists():
            await asyncio.to_thread(model.load)
            logger.info(f"Loaded SARIMAX for {symbol}")
        else:
            logger.info(f"Training new SARIMAX model for {symbol}")
            df = await asyncio.to_thread(self.coin_pre.load_features_series, symbol)
            await asyncio.to_thread(model.train, df, target_col='close')
            await asyncio.to_thread(model.save)
            logger.info(f"SARIMAX model saved for {symbol}")
        return model

    async def run(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        try:
            params = request.params if hasattr(request, "params") else None
            input_data = (params.arguments if params and params.arguments is not None else {})  # type: ignore[attr-defined]
            symbol = input_data.get('symbol', 'BTC')
            horizon = input_data.get('horizon', 7)
            start_date = input_data.get('start_date')

            model = await self.get_model_for_symbol(symbol)

            df = await asyncio.to_thread(self.coin_pre.load_features_series, symbol)
            if start_date:
                try:
                    df = df[df.index >= pd.to_datetime(start_date)]
                except ValueError:
                    logger.warning(f"Invalid start_date format: {start_date}. Ignoring date filter.")

            last_date = df.index[-1]
            forecast = await asyncio.to_thread(model.forecast, steps=horizon, last_date=last_date, freq='h')
            forecast_df = pd.DataFrame({
                'timestamp': pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='h'),
                'predicted_close': forecast.values
            })

            metrics = {'mae_forecast': np.mean(np.abs(forecast.values))}
            await asyncio.to_thread(
                log_model_params_and_metrics,
                'SARIMAX-Forecast',
                symbol,
                {'horizon': horizon},
                metrics
            )

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"SARIMAX Forecast for {symbol}\nNext {horizon} hours:\n{forecast_df.to_string(index=False)}"
                )]
            )
        except Exception as exc:
            err = f"Error: {type(exc).__name__} - {exc}"
            logger.error(err, exc_info=True)
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=err)]
            )

# === START OF STRUCTURAL FIX ===
# Definitions must be at module scope so decorators run on import.
server = Server("crypto-sarimax-server")
mcp = SarimaxMCP()


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="forecast_sarimax",
            description="Generate SARIMAX forecast for any crypto symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "horizon": {"type": "integer"},
                    "start_date": {"type": "string"},
                },
                "required": ["symbol"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name == "forecast_sarimax":
        return await mcp.run(
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
        logger.info("Server stopped by user")
