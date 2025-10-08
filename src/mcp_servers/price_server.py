import asyncio
import logging
import sys
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from modules.forecasting.models.sarimax import SarimaxModel
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.registry.mlflow_utils import log_model_params_and_metrics

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODEL_DIR = Path(r"D:\python_projects\crypto-analytics-platform\src\modules\forecasting\models\saved")
SARIMAX_BASE_DIR = MODEL_DIR / "sarimax"

class AsyncStdioWrapper:
    def __init__(self, fileobj, mode: str):
        self._file = fileobj
        self._mode = mode
        self._loop = None
        self._closed = False

    async def __aenter__(self):
        self._loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._closed = True
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        data = await self.readline()
        if not data:
            raise StopAsyncIteration
        return data

    async def read(self, n: int = -1):
        if self._closed:
            return b""
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.read, n)

    async def readline(self):
        if self._closed:
            return b""
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.readline)

    async def readexactly(self, n: int):
        if self._closed:
            return b""
        loop = self._loop or asyncio.get_running_loop()

        def _readexactly(cnt):
            data = b''
            while len(data) < cnt:
                chunk = self._file.read(cnt - len(data))
                if not chunk:
                    break
                data += chunk
            return data

        return await loop.run_in_executor(None, _readexactly, n)

    async def write(self, data: bytes):
        if self._closed:
            return 0
        loop = self._loop or asyncio.get_running_loop()

        def _write(d):
            written = self._file.write(d)
            try:
                self._file.flush()
            except Exception:
                pass
            return written

        return await loop.run_in_executor(None, _write, data)

    async def drain(self):
        await asyncio.sleep(0)

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
                self.default_model.load()
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
            model.load()
            logger.info(f"Loaded SARIMAX for {symbol}")
        else:
            logger.info(f"Training new SARIMAX model for {symbol}")
            df = self.coin_pre.load_features_series(symbol)
            model.train(df, target_col='close')
            model.save()
            logger.info(f"SARIMAX model saved for {symbol}")
        return model

    async def run(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")

        input_data = request.arguments or {}
        symbol = input_data.get('symbol', 'BTC')
        horizon = input_data.get('horizon', 7)
        start_date = input_data.get('start_date')

        model = await self.get_model_for_symbol(symbol)

        df = self.coin_pre.load_features_series(symbol)
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]

        last_date = df.index[-1]
        forecast = model.forecast(steps=horizon, last_date=last_date, freq='H')
        forecast_df = pd.DataFrame({
            'timestamp': pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='H'),
            'predicted_close': forecast.values
        })

        metrics = {'mae_forecast': np.mean(np.abs(forecast.values))}
        log_model_params_and_metrics('SARIMAX-Forecast', symbol, {'horizon': horizon}, metrics)

        return CallToolResult(
            content=[TextContent(
                text=f"SARIMAX Forecast for {symbol}\nNext {horizon} hours:\n{forecast_df.to_string(index=False)}"
            )]
        )

async def main():
    server = Server("crypto-sarimax-server")
    mcp = SarimaxMCP()
    await mcp.initialize()

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
            return await mcp.run(CallToolRequest(name=name, arguments=arguments))
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
    init_options = {"name": "crypto-sarimax-server"}

    try:
        await server.run(read_stream, write_stream, init_options)
    except Exception:
        logger.exception("Server.run failed")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
