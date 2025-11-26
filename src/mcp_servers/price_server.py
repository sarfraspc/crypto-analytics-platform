import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is on sys.path for imports like `core` and `modules`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# MCP Imports
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

# Core Modules
from core.config import settings
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.registry.mlflow_utils import log_model_params_and_metrics
from core.logging_config import setup_logging

from modules.forecasting.models.prophet import ProphetModel

setup_logging()
logger = logging.getLogger(__name__)

class ProphetMCP:
    def __init__(self):
        self.coin_pre = CoinPreprocessor()
        self.model_cache: Dict[str, Any] = {}
        self.is_initialized = False

    async def initialize(self):
        self.is_initialized = True
        logger.info("ProphetMCP Initialized.")

    def _get_real_price_df(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """
        Load raw OHLCV prices for Prophet (unscaled close in USD).
        """
        exchange = getattr(settings, "MARKET_EXCHANGE_ID", "binance")
        df = self.coin_pre.load_data(
            symbol=symbol,
            exchange=exchange,
            interval="1h",
            lookback_days=lookback_days,
        )
        if df.empty or "close" not in df.columns:
            raise ValueError(f"Could not load OHLCV data for {symbol} on {exchange}.")
        return df[["close"]].copy()

    async def get_or_train_model(self, symbol: str) -> ProphetModel:
        symbol = symbol.upper()

        if symbol in self.model_cache:
            return self.model_cache[symbol]

        model = ProphetModel(symbol)
        df_real: Optional[pd.DataFrame] = None

        # Try loading an existing model; if it appears to be trained on 0–1
        # scaled prices, retrain it on real USD closes.
        if model.load():
            try:
                df_real = await asyncio.to_thread(self._get_real_price_df, symbol, lookback_days=180)
                last_close = float(df_real["close"].iloc[-1])
                probe_forecast = await asyncio.to_thread(model.forecast, steps=1, freq="h")
                probe_val = float(probe_forecast["yhat"].iloc[-1])

                if probe_val < 1.0 and last_close > 10.0:
                    logger.info(
                        "Prophet model for %s appears to be scaled (probe=%s, last_close=%s); retraining.",
                        symbol,
                        probe_val,
                        last_close,
                    )
                    await asyncio.to_thread(model.train, df_real, target_col="close")
                    await asyncio.to_thread(model.save)

                self.model_cache[symbol] = model
                return model
            except Exception as exc:
                logger.warning(
                    "Validation of loaded Prophet model for %s failed; retraining from scratch: %s",
                    symbol,
                    exc,
                )

        if df_real is None:
            df_real = await asyncio.to_thread(self._get_real_price_df, symbol, lookback_days=180)

        await asyncio.to_thread(model.train, df_real, target_col="close")
        await asyncio.to_thread(model.save)

        self.model_cache[symbol] = model
        return model

    async def run(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            await self.initialize()

        try:
            params = request.params if hasattr(request, "params") else None
            input_data = (params.arguments if params and params.arguments is not None else {})
            symbol = input_data.get('symbol', 'BTC').upper()
            horizon = int(input_data.get('horizon', 24))
            
            logger.info(f"Processing Prophet forecast for {symbol} (horizon={horizon})")

            # 1. Get Model (Load or Train)
            model = await self.get_or_train_model(symbol)

            # 2. Generate Base Forecast
            forecast_df = await asyncio.to_thread(model.forecast, steps=horizon, freq='h')
            
            # 3. Calculate Volatility
            df_recent = await asyncio.to_thread(self._get_real_price_df, symbol, lookback_days=7)
            
            if 'close' in df_recent.columns and len(df_recent) > 1:
                recent_returns = np.log(df_recent['close'] / df_recent['close'].shift(1))
                volatility = recent_returns.std()
            else:
                volatility = 0.002 

            if np.isnan(volatility) or volatility == 0:
                volatility = 0.002

            # 4. Apply Stochastic Noise
            predicted_prices = []
            timestamps = []
            
            base_trend = forecast_df['yhat'].values
            base_dates = forecast_df['ds'].tolist()

            for i, price in enumerate(base_trend):
                noise_pct = np.random.normal(0, volatility)
                noisy_price = price * (1 + noise_pct)
                predicted_prices.append(noisy_price)
                timestamps.append(str(base_dates[i]))

            # 5. Format Response
            df_view = pd.DataFrame({'timestamp': timestamps, 'predicted_close': predicted_prices})
            raw_text = f"Prophet Forecast for {symbol} (Stochastic)\n{df_view.head(24).to_string(index=False)}"

            response_data = {
                "symbol": symbol,
                "model_used": "prophet_v1_stochastic",
                "timestamps": timestamps,
                "predicted_close": predicted_prices,
                "raw_text": raw_text
            }

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response_data)
                )]
            )

        except Exception as exc:
            # Capture full traceback in logs
            logger.error(f"Forecast Error for {symbol}: {exc}", exc_info=True)
            # Return simple error to client
            return CallToolResult(isError=True, content=[TextContent(type="text", text=f"Error: {str(exc)}")])

server = Server("crypto-prophet-server")
mcp = ProphetMCP()

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="forecast_prophet",
            description="Generate Prophet forecast for any crypto symbol (Trend + Seasonality + Noise)",
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
    if name == "forecast_prophet":
        request = CallToolRequest(
            params=CallToolRequestParams(name=name, arguments=arguments),
            method="tools/call" 
        )
        return await mcp.run(request)
    else:
        raise Exception(f"Unknown tool: {name}")

@server.list_resources()
async def list_resources():
    return []

@server.read_resource()
async def read_resource(name: str):
    raise Exception(f"Unknown resource: {name}")

async def main():
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
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
