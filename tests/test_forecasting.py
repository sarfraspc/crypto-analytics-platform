import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.modules.forecasting.models.sarimax import SarimaxModel
from src.modules.forecasting.models.prophet import ProphetModel
from src.modules.forecasting.evaluation.metrics import compute_forecast_metrics
from src.modules.forecasting.registry.mlflow_utils import log_model_params_and_metrics

# Fixtures

@pytest.fixture
def sample_data():
    """Creates a synthetic hourly price dataframe."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    # Random walk + trend
    data = np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)
    df = pd.DataFrame({"close": data}, index=dates)
    # Add log return for SARIMAX
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()

# SARIMAX Tests

def test_sarimax_initialization():
    """Test that directories are created (mocked) on init."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        model = SarimaxModel(symbol="BTC")
        assert model.symbol == "BTC"
        assert model.order == (1, 0, 1)
        mock_mkdir.assert_called()

def test_sarimax_train_flow(sample_data):
    """Test the training plumbing (without running actual statsmodels optimization)."""
    model = SarimaxModel(symbol="ETH")
    
    # Mock the internal statsmodels SARIMAX object
    with patch("src.modules.forecasting.models.sarimax.SARIMAX") as MockSARIMAX:
        mock_instance = MockSARIMAX.return_value
        mock_fit = mock_instance.fit.return_value
        
        model.train(sample_data, target_col="log_return")
        
        # Check if SARIMAX was initialized with correct data shape
        assert MockSARIMAX.called
        # Check if fit() was called
        assert mock_instance.fit.called
        assert model.model_fit == mock_fit

def test_sarimax_forecast_without_training_raises_error():
    """Test guardrail for forecasting before training."""
    model = SarimaxModel(symbol="BTC")
    with pytest.raises(RuntimeError):
        model.forecast(steps=5)

def test_sarimax_save_load():
    """Test joblib pickling (mocked)."""
    model = SarimaxModel(symbol="SOL")
    model.model_fit = MagicMock() # Pretend it's trained
    
    with patch("joblib.dump") as mock_dump, \
         patch("joblib.load") as mock_load, \
         patch("pathlib.Path.exists", return_value=True):
        
        # Test Save
        model.save()
        mock_dump.assert_called_once()
        
        # Test Load
        model.load()
        mock_load.assert_called_once()

# Prophet Tests

def test_prophet_train_flow(sample_data):
    """Test Prophet data formatting and training call."""
    model = ProphetModel(symbol="BTC")
    
    with patch("src.modules.forecasting.models.prophet.Prophet") as MockProphet:
        mock_instance = MockProphet.return_value
        
        model.train(sample_data, target_col="close")
        
        # Verify Prophet was initialized
        assert MockProphet.called
        
        # Verify fit was called
        assert mock_instance.fit.called
        
        # Verify data passed to fit has 'ds' and 'y' columns (Prophet requirement)
        call_args = mock_instance.fit.call_args
        df_passed = call_args[0][0]
        assert "ds" in df_passed.columns
        assert "y" in df_passed.columns

def test_prophet_forecast_logic(sample_data):
    """Test that forecast returns formatted dataframe and clamps negative values."""
    model = ProphetModel(symbol="BTC")
    model.model = MagicMock() # Mock trained model
    
    # Setup mock forecast return
    future_dates = pd.date_range(start="2024-01-02", periods=5, freq="h")
    # Include a negative value to test clamping logic
    mock_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': [100, 101, -5, 103, 104], 
        'yhat_lower': [90, 90, -10, 90, 90],
        'yhat_upper': [110, 110, 0, 110, 110],
        'trend': [100]*5 # Extra col to ensure it filters output
    })
    
    model.model.make_future_dataframe.return_value = pd.DataFrame({'ds': future_dates})
    model.model.predict.return_value = mock_forecast
    
    # Run forecast
    result = model.forecast(steps=5)
    
    # Assertions
    assert len(result) == 5
    # Check if negative value was clamped to 0
    assert result.iloc[2]['yhat'] == 0
    assert 'trend' not in result.columns # Should only return yhat columns

# 3. Metric Tests

def test_compute_metrics_simple():
    """Test math correctness on known inputs."""
    y_true = [100, 200, 300]
    y_pred = [110, 190, 300]
    # Errors: 10, 10, 0 -> MAE = 20/3 = 6.666...
    
    metrics = compute_forecast_metrics(y_true, y_pred)
    
    assert metrics['mae'] == pytest.approx(6.666, 0.01)
    assert metrics['rmse'] > 0
    assert 'mape' in metrics

def test_compute_metrics_multi_horizon():
    """Test handling of 2D arrays (sequence outputs)."""
    # Shape: (Samples, Horizon) -> (2 samples, 2 steps ahead)
    y_true = np.array([[10, 12], [20, 22]])
    y_pred = np.array([[11, 11], [21, 21]])
    
    metrics = compute_forecast_metrics(y_true, y_pred, multi_horizon=True)
    
    assert 'mae_per_horizon' in metrics
    assert len(metrics['mae_per_horizon']) == 2 
    assert isinstance(metrics['mae'], float) 

# MLflow Registry Tests

def test_mlflow_logging():
    """Test that the logger calls MLflow functions correctly."""
    
    with patch("mlflow.start_run") as mock_start, \
         patch("mlflow.log_param") as mock_param, \
         patch("mlflow.log_metric") as mock_metric, \
         patch("mlflow.log_artifacts") as mock_artifacts, \
         patch("os.path.exists", return_value=True): 
        
        params = {"order": (1,1,1)}
        metrics = {"mae": 0.5, "rmse": 0.8}
        
        log_model_params_and_metrics(
            model_type="SARIMAX",
            symbol="BTC",
            params=params,
            metrics=metrics,
            artifacts_path="/tmp/model"
        )
        
        # Assertions
        mock_start.assert_called_once()
        
        # Check Params
        # 1. model_type
        # 2. symbol
        # 3. order
        assert mock_param.call_count == 3
        
        # Check Metrics
        assert mock_metric.call_count == 2 # mae, rmse
        
        # Check Artifacts
        mock_artifacts.assert_called_once()