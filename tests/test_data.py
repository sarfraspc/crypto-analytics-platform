"""
Tests for data ingestion and client modules.

Covers market data, news aggregation, and on-chain data
ingestion pipelines and orchestration logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data import market_client, news_client, onchain_client


# Fixtures

@pytest.fixture
def mock_db_session():
    """Creates a mock SQLAlchemy session."""
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = []
    return session

# Market Client Tests (Orchestrator)

def test_get_symbols_from_tokens(mock_db_session):
    """Test extracting and formatting symbols from the Token table."""
    # Setup mock data
    mock_token = MagicMock()
    mock_token.symbol = "BTC"
    mock_token.token_metadata = {"market_cap_rank": 1}
    
    # Mock the chain: db.execute(...).scalars().all()
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = [mock_token]

    symbols = market_client.get_symbols_from_tokens(mock_db_session, limit=10)

    assert len(symbols) == 1
    assert symbols[0]['label'] == "BTC"
    assert symbols[0]['use_ccxt_symbol'] == "BTC/USDT"

def test_run_ta_patterns_success():
    """Test TA pattern generation logic when signals are found."""
    # Patch where generate_ta_signal is IMPORTED in market_client.py
    with patch("src.data.market_client.generate_ta_signal") as mock_gen:
        # Mock return value for generate_ta_signal
        mock_gen.return_value = {"pattern": "DOJI", "signal": "buy"}
        
        result = market_client.run_ta_patterns(["BTC"], exchange="kraken")
        
        assert result["status"] == "success"
        assert "BTC" in result["patterns"]

def test_run_ta_patterns_no_data():
    """Test TA pattern logic when no signals are generated."""
    with patch("src.data.market_client.generate_ta_signal") as mock_gen:
        mock_gen.return_value = None  # No signal found
        
        result = market_client.run_ta_patterns(["BTC"])
        
        assert result["status"] == "no_data"
        assert result["patterns"] == {}

@pytest.mark.asyncio
async def test_run_ingestion_cycle(mock_db_session):
    """
    Test the full ingestion orchestration:
    1. Backfill (Phase 1)
    2. TA Generation (Phase 2)
    3. DB Updates
    """
    # Mocks for external dependencies
    with patch("src.data.market_client.backfill_and_ta") as mock_backfill, \
         patch("src.data.market_client.generate_ta_signal") as mock_ta, \
         patch("src.data.market_client.update_ingestion_job") as mock_update_job, \
         patch("src.data.market_client.get_top_symbols") as mock_get_top, \
         patch("src.data.market_client.log_ingestion_to_mlflow") as mock_mlflow:

        # Setup
        mock_backfill.return_value = 100  # 100 records inserted
        mock_ta.return_value = {"signal": "buy"}
        mock_get_top.return_value = ["BTC"] # TA candidates
        
        # Input symbols
        symbols = [{'label': 'BTC', 'use_ccxt_symbol': 'BTC/USDT', 'exchange': 'kraken'}]

        # Run
        await market_client.run_ingestion_cycle(
            mock_db_session, 
            mock_db_session, 
            pipeline="test_cycle",
            symbols=symbols,
            delta_only=True
        )

        # Assertions
        assert mock_backfill.call_count == 1
        mock_update_job.assert_called_once()

# News Client Tests (Orchestrator)

@pytest.mark.asyncio
async def test_run_news_cycle(mock_db_session):
    """Test news aggregation from multiple sources."""
    
    # Patch the functions imported into src.data.news_client
    with patch("src.data.news_client.ingest_cryptopanic") as mock_cp, \
         patch("src.data.news_client.ingest_reddit_praw") as mock_reddit, \
         patch("src.data.news_client.ingest_fng") as mock_fng, \
         patch("src.data.news_client.log_ingestion_to_mlflow") as mock_mlflow:

        # Setup return values (inserted_count, skipped_count)
        mock_cp.return_value = (10, 2)
        mock_reddit.return_value = (5, 0)
        mock_fng.return_value = (1, 0)

        # Run
        result = await news_client.run_news_cycle(mock_db_session)

        # Assertions
        assert result["cryptopanic_inserted"] == 10
        assert result["total_inserted"] == 16
        mock_mlflow.assert_called_once()

# 3. Chain Client Tests (Orchestrator)

def test_run_whale_ingestion_found_whales():
    """Test logic when whales are detected."""
    
    with patch("src.data.onchain_client.chain_client.scan_eth_transfers") as mock_scan:
        # Setup: Scan returns whale alerts
        mock_scan.return_value = {"whale_alerts": 5}

        result = onchain_client.run_whale_ingestion(
            chain="ethereum", time_window="24h"
        )

        assert result["status"] == "success"
        assert result["ingestion"]["whale_alerts"] == 5

def test_run_whale_ingestion_no_whales():
    """Test logic when NO whales are detected."""
    
    with patch("src.data.onchain_client.chain_client.scan_eth_transfers") as mock_scan:
        # Setup: Scan returns 0 alerts
        mock_scan.return_value = {"whale_alerts": 0}

        result = onchain_client.run_whale_ingestion()

        assert result["status"] == "no_data"

def test_run_metrics_update_success():
    """Test the metrics update orchestration."""
    
    with patch("src.data.onchain_client.run_onchain_metrics") as mock_calc:
        mock_calc.return_value = {"status": "success", "errors": []}

        result = onchain_client.run_metrics_update(chain="ethereum")

        assert result["status"] == "success"

def test_run_onchain_pipeline_full():
    """Test the high-level pipeline running both ingestion and metrics."""
    
    with patch("src.data.onchain_client.run_whale_ingestion") as mock_ingest, \
         patch("src.data.onchain_client.run_metrics_update") as mock_metrics, \
         patch("src.data.onchain_client.log_pipeline_to_mlflow") as mock_log:

        mock_ingest.return_value = {"status": "success"}
        mock_metrics.return_value = {"status": "success"}

        result = onchain_client.run_onchain_pipeline(run_steps=["ingestion", "metrics"])

        assert mock_ingest.called
        assert mock_metrics.called
        mock_log.assert_called_once()