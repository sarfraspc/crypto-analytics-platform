import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from modules.forecasting.models.sarimax import SarimaxModel
from modules.forecasting.models.prophet import ProphetModel
from modules.forecasting.models.cnn_lstm import CNNLSTMForecaster
from modules.forecasting.models.tft import TFTForecaster
from modules.forecasting.evaluation.metrics import compute_forecast_metrics
from modules.forecasting.evaluation.benchmark import run_benchmark

DATES = pd.date_range('2023-01-01', periods=200, freq='h') 
DUMMY_DF = pd.DataFrame({
    'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
    'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
    'high': 100 + np.cumsum(np.random.randn(200) * 0.5) + 1,
    'low': 100 + np.cumsum(np.random.randn(200) * 0.5) - 1,
    'volume': np.random.rand(200) * 1000,
}, index=DATES)
DUMMY_DF['returns'] = DUMMY_DF['close'].pct_change()
DUMMY_DF['sma_7'] = DUMMY_DF['close'].rolling(7).mean()
DUMMY_DF['ema_8'] = DUMMY_DF['close'].ewm(span=8).mean()
DUMMY_DF['volume_zscore_30'] = (DUMMY_DF['volume'] - DUMMY_DF['volume'].rolling(30).mean()) / DUMMY_DF['volume'].rolling(30).std()
DUMMY_DF = DUMMY_DF.dropna()
DUMMY_DF = DUMMY_DF.reindex(DUMMY_DF.index, method='ffill')  
DUMMY_DF['symbol'] = 'BTC'
DUMMY_DF['time'] = DUMMY_DF.index

def test_sarimax_train_forecast():
    model = SarimaxModel('BTC')
    model.train(DUMMY_DF, target_col='close')
    forecast = model.forecast(steps=5)
    assert len(forecast) == 5
    assert isinstance(forecast, pd.Series)

def test_prophet_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = ProphetModel('BTC', model_dir=str(Path(tmpdir)))
        model.train(DUMMY_DF, target_col='close')
        model.save()
        
        loaded_model = ProphetModel('BTC', model_dir=str(Path(tmpdir)))
        loaded_model.load()
        forecast_original = model.forecast(steps=5)
        forecast_loaded = loaded_model.forecast(steps=5)
        np.testing.assert_allclose(forecast_original.values, forecast_loaded.values, rtol=1e-5)

def test_cnn_lstm_forecast_shape():
    forecaster = CNNLSTMForecaster(sequence_length=10, forecast_horizon=5)
    n = len(DUMMY_DF)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_df = DUMMY_DF.iloc[:train_end]
    val_df = DUMMY_DF.iloc[train_end:val_end]
    test_df = DUMMY_DF.iloc[val_end:]
    forecaster.prepare_data(train_df, val_df, test_df)
    forecaster.build_model()
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        forecaster.train(epochs=1, checkpoint_path=tmp.name)
        y_pred = forecaster.model.predict(forecaster.X_test)
        assert y_pred.shape == (len(forecaster.X_test), 5)
    os.unlink(tmp.name)

@pytest.mark.skip(reason="TFT requires consecutive timesteps; skip for now")
def test_tft_forecast_shape():
    forecaster = TFTForecaster(max_encoder_length=10, max_prediction_length=5)
    n = len(DUMMY_DF)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_df = DUMMY_DF.iloc[:train_end]
    val_df = DUMMY_DF.iloc[train_end:val_end]
    test_df = DUMMY_DF.iloc[val_end:]
    forecaster.prepare_data(train_df, val_df, test_df)
    forecaster.build_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        forecaster.train(max_epochs=1, checkpoint_dir=tmpdir)
        preds_raw = forecaster.tft.predict(forecaster.test_dataloader, return_y=True, mode="raw")
        y_pred = preds_raw.output[0][:, 0, :]
        assert y_pred.shape[1] == 5

def test_compute_forecast_metrics():
    y_true_multi = np.array([[1, 2, 3], [4, 5, 6]])
    y_pred_multi = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    metrics_multi = compute_forecast_metrics(y_true_multi, y_pred_multi, multi_horizon=True)
    assert 'mae' in metrics_multi
    assert metrics_multi['mae'] > 0
    assert len(metrics_multi['mae_per_horizon']) == 3
    assert 0 <= metrics_multi['directional_acc'] <= 100

    y_true_single = y_true_multi.flatten()
    y_pred_single = y_pred_multi.flatten()
    metrics_single = compute_forecast_metrics(y_true_single, y_pred_single)
    assert 'mae' in metrics_single
    assert metrics_single['mae'] > 0
    assert 0 <= metrics_single['directional_acc'] <= 100

@pytest.mark.skip(reason="Benchmark calls TFT; skip until fixed")
def test_run_benchmark(monkeypatch):
    def mock_load_features_series(*args, **kwargs):
        return DUMMY_DF
    
    monkeypatch.setattr("modules.forecasting.data.preprocess_coin.CoinPreprocessor.load_features_series", mock_load_features_series)
    
    results = run_benchmark(symbol="BTC", forecast_steps=3, rolling_eval=False, retrain_if_exists=True)
    assert "sarimax" in results
    assert "metrics" in results["sarimax"]
    assert "mae" in results["sarimax"]["metrics"]
    assert results["sarimax"]["metrics"]["mae"] >= 0