"""
Tests for on-chain analytics modules.

Covers whale alert detection, exchange flow calculations,
metric aggregation, and technical analysis pattern generation.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd

from src.modules.onchain.metrics.aggregator import combine_metrics
from src.modules.onchain.metrics.exchange_flows import compute_exchange_flows
from src.modules.onchain.metrics.whale_alerts import summarize_whale_alerts
from src.modules.onchain.patterns.ta_patterns import _generate_signal, generate_ta_signal


# Whale Alerts Tests

def test_summarize_whale_alerts_logic():
    """Test that whale alerts are summed correctly."""
    
    # Mock the DB context manager and query result
    with patch("src.modules.onchain.metrics.whale_alerts.get_timescale_db") as mock_db_ctx:
        mock_session = MagicMock()
        mock_db_ctx.return_value.__enter__.return_value = mock_session
        
        alert1 = MagicMock(usd_value=Decimal(1000000), from_address="addr1", to_address="exchange1")
        alert2 = MagicMock(usd_value=Decimal(2000000), from_address="addr2", to_address="addr3")
        
        mock_session.execute.return_value.scalars.return_value.all.return_value = [alert1, alert2]
        
        result = summarize_whale_alerts(chain="ethereum", time_window="24h")
        
        assert result['whale_count'] == 2
        assert result['total_whale_volume_usd'] == 3000000.0
        assert result['unique_whale_addresses'] == 4

# 2. Exchange Flows Tests

def test_compute_exchange_flows_inflow_outflow():
    """Test classification of inflow vs outflow based on address lists."""
    
    with patch("src.modules.onchain.metrics.exchange_flows.get_timescale_db") as mock_db_ctx, \
         patch("src.modules.onchain.metrics.exchange_flows.EXCHANGE_ADDRS", {"exc_in": "kraken", "exc_out": "kraken"}):
        mock_session = MagicMock()
        mock_db_ctx.return_value.__enter__.return_value = mock_session
        
        # Scenario: Wallet -> Exchange ($100 Inflow), Exchange -> Wallet ($50 Outflow)
        alert1 = MagicMock(usd_value=Decimal(100), from_address="wallet1", to_address="exc_in")
        alert2 = MagicMock(usd_value=Decimal(50), from_address="exc_out", to_address="wallet2")
        
        # Mock current window alerts (first call) then previous window (second call)
        mock_session.execute.return_value.scalars.return_value.all.side_effect = [
            [alert1, alert2], 
            [] 
        ]
        
        result = compute_exchange_flows(chain="ethereum")
        
        assert result['exchange_inflow_usd'] == 100.0
        assert result['exchange_outflow_usd'] == 50.0
        assert result['net_flow_usd'] == -50.0

# Aggregator Tests

def test_combine_metrics_math():
    """Test the 'Market Pressure Index' calculation logic."""
    
    with patch("src.modules.onchain.metrics.aggregator.get_timescale_db") as mock_db_ctx:
        mock_session = MagicMock()
        mock_db_ctx.return_value.__enter__.return_value = mock_session
        
        # Mock Metric Queries (scalar_one_or_none)
        # 1. Net Flow (-100)
        # 2. Whale Inflow (50)
        mock_session.execute.return_value.scalar_one_or_none.side_effect = [-100.0, 50.0]
        
        # Mock OHLCV Query (list of objects)
        ohlc_start = MagicMock(close=100)
        ohlc_end = MagicMock(close=110)
        
        # mock the .all() return for list queries
        # 1. OHLCV (Current Window) -> [Start, End]
        # 2. 7D Flows -> [100, 100]
        # 3. 7D Prices -> [100, 110]
        mock_session.execute.return_value.scalars.return_value.all.return_value = [ohlc_start, ohlc_end]
        mock_session.execute.return_value.all.return_value = [(100,), (110,)] 
        
        result = combine_metrics(chain="ethereum")
        
        assert result is not None
        assert result['market_pressure_index'] >= 0
        assert result['symbol'] == "ETH"
        assert result['price_change_pct'] == 10.0

# TA Patterns Tests

def test_generate_signal_logic():
    """Test the pure logic function _generate_signal."""
    signal, explanation = _generate_signal(rsi=25, macd_hist=0.5, pattern="hammer", pattern_direction="bullish")
    assert signal == "bullish"
    assert "RSI below 30" in explanation

def test_generate_ta_signal_flow():
    """Test the full TA pipeline with mocked data loading."""
    
    with patch("src.modules.onchain.patterns.ta_patterns.load_recent_ohlcv") as mock_load, \
         patch("src.modules.onchain.patterns.ta_patterns.compute_ta_indicators") as mock_indic, \
         patch("src.modules.onchain.patterns.ta_patterns.detect_candlestick_patterns") as mock_patterns, \
         patch("src.modules.onchain.patterns.ta_patterns.get_timescale_db"):
        dates = pd.date_range(start="2024-01-01", periods=40, freq="h")
        df = pd.DataFrame({
            'close': [100]*40, 
            'open': [100]*40,
            'high': [100]*40,
            'low': [100]*40,
            'time': dates
        })
        mock_load.return_value = df
        
        # Mock Indicator Return
        df_indic = df.copy()
        df_indic['rsi'] = 20 # Oversold
        df_indic['macd_hist'] = 1.0
        mock_indic.return_value = df_indic
        
        # Mock Pattern Return
        df_pat = df_indic.copy()
        df_pat['cdl_engulfing'] = 0
        df_pat['cdl_hammer'] = 0
        # Set the LAST row to have a hammer pattern
        df_pat.iloc[-1, df_pat.columns.get_loc('cdl_hammer')] = 100
        
        # Initialize required columns to 0
        for col in ['cdl_harami', 'cdl_shootingstar', 'cdl_doji', 'cdl_invertedhammer', 'cdl_spinningtop', 'cdl_marubozu']:
            df_pat[col] = 0
            
        mock_patterns.return_value = df_pat
        
        result = generate_ta_signal(symbol="BTC", use_cache=False)
        
        assert result is not None
        assert result['signal'] == "bullish"
        assert result['pattern'] == "hammer"
        assert result['rsi'] == 20.0