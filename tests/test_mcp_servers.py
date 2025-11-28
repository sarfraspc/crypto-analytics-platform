"""
Tests for MCP server implementations.

Covers price forecasting, on-chain metrics, and sentiment
analysis MCP tool handlers and request processing.
"""

import json

import pandas as pd
import pytest
from mcp.types import CallToolRequest, CallToolRequestParams
from unittest.mock import MagicMock, patch

from src.mcp_servers.chain_server import OnchainMCP
from src.mcp_servers.price_server import ProphetMCP
from src.mcp_servers.sentiment_server import PipelineMCP


# Price Server Tests

@pytest.mark.asyncio
async def test_prophet_mcp_forecast():
    """Test the forecast_prophet tool logic."""
    
    # Mock the dependencies inside ProphetMCP
    with patch("src.mcp_servers.price_server.CoinPreprocessor") as MockPrep, \
         patch("src.mcp_servers.price_server.ProphetModel") as MockModel:
        
        # Setup Mock Preprocessor
        mock_prep = MockPrep.return_value
        mock_prep.scaler_dir = "fake_dir"
        mock_prep.load_features_series.return_value = pd.DataFrame() 
        
        # Setup Mock Model
        mock_model_instance = MockModel.return_value
        # Forecast returns DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        mock_model_instance.forecast.return_value = pd.DataFrame({
            'ds': [pd.Timestamp("2024-01-01 10:00")],
            'yhat': [50000.0],
            'yhat_lower': [49000.0],
            'yhat_upper': [51000.0]
        })
        # Mock load() to return True so it doesn't try to retrain
        mock_model_instance.load.return_value = True
        
        # Initialize MCP
        mcp = ProphetMCP()
        await mcp.initialize()
        
        # We also need to mock _get_real_price_df to return a dataframe for volatility calc
        # Since it's an instance method, we patch it on the instance or via context
        # Easier way: patch asyncio.to_thread to handle the specific calls
        
        with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs) if not asyncio.iscoroutinefunction(func) else func(*args, **kwargs)):
            # Mock the _get_real_price_df explicitly since it's complex
            mcp._get_real_price_df = MagicMock(return_value=pd.DataFrame({'close': [100, 101]}))
            
            # Construct Request
            request = CallToolRequest(
                params=CallToolRequestParams(
                    name="forecast_prophet",
                    arguments={"symbol": "BTC", "horizon": 24}
                ),
                method="tools/call"
            )
            
            # Run Tool
            result = await mcp.run(request)
            
            # Assertions
            assert not result.isError
            content = json.loads(result.content[0].text)
            assert content['symbol'] == "BTC"
            assert content['model_used'] == "prophet_v1_stochastic"
            assert len(content['predicted_close']) > 0

# Onchain Server Tests

@pytest.mark.asyncio
async def test_onchain_mcp_metrics():
    """Test run_metrics_only tool."""
    
    mcp = OnchainMCP()
    # Manually set init flag to skip setup_mlflow
    mcp.is_initialized = True
    
    with patch("src.mcp_servers.chain_server.get_timescale_db") as mock_db_ctx:
        mock_session = MagicMock()
        mock_db_ctx.return_value.__enter__.return_value = mock_session
        
        # The MCP queries multiple metrics. It executes SQL.
        # We'll mock scalar_one_or_none to return a dummy float value
        mock_session.execute.return_value.scalar_one_or_none.return_value = 100.0
        
        request = CallToolRequest(
            params=CallToolRequestParams(
                name="run_metrics_only",
                arguments={"chain": "ethereum", "window": "24h"}
            ),
            method="tools/call"
        )
        
        result = await mcp.run_metrics_only(request)
        
        assert not result.isError
        content = json.loads(result.content[0].text)
        assert "flows" in content
        assert content['flows']['net_flow_usd'] == 100.0
        assert "aggregated" in content

@pytest.mark.asyncio
async def test_onchain_mcp_patterns():
    """Test run_patterns_only tool."""
    mcp = OnchainMCP()
    mcp.is_initialized = True
    
    with patch("src.mcp_servers.chain_server.get_timescale_db") as mock_db_ctx:
        mock_session = MagicMock()
        mock_db_ctx.return_value.__enter__.return_value = mock_session
        
        # Mock returning a list of TASignal objects
        mock_signal = MagicMock()
        mock_signal.symbol = "BTC"
        mock_signal.signal = "bullish"
        mock_signal.time = pd.Timestamp("2024-01-01")
        
        mock_session.execute.return_value.scalars.return_value.all.return_value = [mock_signal]
        
        request = CallToolRequest(
            params=CallToolRequestParams(
                name="run_patterns_only",
                arguments={"exchange": "binance"}
            ),
            method="tools/call"
        )
        
        result = await mcp.run_patterns_only(request)
        
        content = json.loads(result.content[0].text)
        assert content['status'] == "success"
        assert "BTC" in content['patterns']
        assert content['patterns']['BTC']['signal'] == "bullish"

# Sentiment Server Tests

@pytest.mark.asyncio
async def test_pipeline_mcp_sentiment():
    """Test analyze_sentiment tool."""
    
    # must mock the initialization where it loads the heavy classifier
    with patch("src.mcp_servers.sentiment_server.get_sentiment_classifier") as mock_loader, \
         patch("src.mcp_servers.sentiment_server.analyze_sentiment") as mock_analyze:
        
        mcp = PipelineMCP()
        await mcp.initialize()
        
        # Mock Inference Result
        mock_analyze.return_value = {
            "sentiment": "BULLISH",
            "confidence": 0.9,
            "bullish_score": 0.9,
            "bearish_score": 0.1,
            "neutral_score": 0.0
        }
        
        request = CallToolRequest(
            params=CallToolRequestParams(
                name="analyze_sentiment",
                arguments={"text": "Bitcoin to the moon"}
            ),
            method="tools/call"
        )
        
        result = await mcp.analyze_sentiment(request)
        
        assert not result.isError
        content = json.loads(result.content[0].text)
        assert content['sentiment'] == "BULLISH"
        assert content['confidence'] == 0.9

@pytest.mark.asyncio
async def test_pipeline_mcp_rag_query():
    """Test query_rag tool (with mocked retrieval)."""
    
    # We need to patch the components created in __init__
    with patch("src.mcp_servers.sentiment_server.Embedder"), \
         patch("src.mcp_servers.sentiment_server.QdrantVectorStore") as MockVS, \
         patch("src.mcp_servers.sentiment_server.Retriever"), \
         patch("src.mcp_servers.sentiment_server.Generator"), \
         patch("src.mcp_servers.sentiment_server.RedisCache"), \
         patch("src.mcp_servers.sentiment_server.get_sentiment_classifier"):
        
        mcp = PipelineMCP()
        await mcp.initialize()
        
        # Mock Document Count check (must be > 0)
        mcp._get_vector_count = MagicMock(return_value=100)
        
        # Mock Cache (return None to force generation)
        mcp.cache.get_json.return_value = None
        
        # Mock RAG Components
        mcp.retriever.retrieve.return_value = [{'content': 'context'}]
        mcp.generator.generate.return_value = "AI Answer"
        
        request = CallToolRequest(
            params=CallToolRequestParams(
                name="query_rag",
                arguments={"query": "Why is BTC up?"}
            ),
            method="tools/call"
        )
        
        result = await mcp.query_rag(request)
        
        assert not result.isError
        content = json.loads(result.content[0].text)
        assert content['response'] == "AI Answer"
        assert len(content['contexts']) == 1