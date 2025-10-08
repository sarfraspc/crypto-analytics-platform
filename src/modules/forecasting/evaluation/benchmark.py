import pandas as pd
import numpy as np
from typing import Any
from pathlib import Path
import shutil

from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_panel import PanelPreprocessor
from modules.forecasting.explainers.xai import explain_model_predictions
from modules.forecasting.models.sarimax import SarimaxModel
from modules.forecasting.models.prophet import ProphetModel
from modules.forecasting.models.cnn_lstm import CNNLSTMForecaster
from modules.forecasting.models.tft import TFTForecaster
from modules.forecasting.evaluation.metrics import compute_forecast_metrics
from modules.forecasting.registry.mlflow_utils import init_mlflow_experiment, log_model_params_and_metrics
from pytorch_forecasting import TemporalFusionTransformer
import torch  


def split_data_for_evaluation(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1):
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))
    
    train_df = df.iloc[:val_start]
    val_df = df.iloc[val_start:test_start]
    test_df = df.iloc[test_start:]
    
    return train_df, val_df, test_df

def _rolling_forecast(
    model: Any,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_steps: int,
    target_col: str = 'close'
):
    y_pred_all = []
    y_true_all = []
    
    data_freq = pd.infer_freq(train_val_df.index) or 'H'
    
    step_size = 1
    for i in range(0, len(test_df) - forecast_steps + 1, step_size):
        history = pd.concat([train_val_df, test_df.iloc[:i]])
        model.train(history, target_col=target_col)
        forecast_result = model.forecast(steps=forecast_steps, last_date=history.index[-1], freq=data_freq)
        y_pred_all.extend(forecast_result.values)
        y_true_all.extend(test_df[target_col].iloc[i:i+forecast_steps].values)
        
    return np.array(y_pred_all), np.array(y_true_all)


def evaluate_sarimax(
    symbol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_steps: int = 7,
    retrain_if_exists: bool = False,
    **kwargs
):
    if len(test_df) == 0:
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}
    
    train_val_df = pd.concat([train_df, val_df])
    
    sarimax_model = SarimaxModel(symbol)
    if sarimax_model.model_path.exists() and not retrain_if_exists:
        sarimax_model.load()
    else:
        sarimax_model.train(train_val_df, target_col='close')
        sarimax_model.save()
    
    rolling_eval = kwargs.get('rolling_eval', True)
    
    if rolling_eval and len(test_df) > forecast_steps:
        y_pred, y_true = _rolling_forecast(sarimax_model, train_val_df, test_df, forecast_steps)
    else:
        data_freq = pd.infer_freq(train_val_df.index) or 'H'
        forecast = sarimax_model.forecast(steps=len(test_df), last_date=train_val_df.index[-1], freq=data_freq)
        y_pred = forecast.values[:len(test_df)]
        y_true = test_df['close'].values

    coin_pre = CoinPreprocessor()
    explanation = explain_model_predictions(
        model_type='SARIMAX',  
        model=sarimax_model,   
        preprocessor=coin_pre,
        symbol=symbol,
        test_df=test_df,      
        n_samples=50           
    )
    print(f"Top SHAP feature for {symbol} (SARIMAX): {explanation['features'][np.argmax(np.mean(np.abs(explanation['shap_values']), axis=0))]}")
    
    metrics = compute_forecast_metrics(y_true, y_pred)
    
    params = {
        'order': sarimax_model.order,
        'seasonal_order': sarimax_model.seasonal_order,
        'forecast_steps': forecast_steps
    }
    
    log_model_params_and_metrics('SARIMAX', symbol, params, metrics, str(sarimax_model.model_dir))
    
    return {'metrics': metrics, 'params': params}


def evaluate_prophet(
    symbol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_steps: int = 7,
    retrain_if_exists: bool = False,
    **kwargs
):
    if len(test_df) == 0:
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}

    train_val_df = pd.concat([train_df, val_df])
    
    prophet_model = ProphetModel(symbol)
    if prophet_model.model_path.exists() and not retrain_if_exists:
        prophet_model.load()
    else:
        prophet_model.train(train_val_df, target_col='close')
        prophet_model.save()
    
    rolling_eval = kwargs.get('rolling_eval', True)

    if rolling_eval and len(test_df) > forecast_steps:
        y_pred, y_true = _rolling_forecast(prophet_model, train_val_df, test_df, forecast_steps)
    else:
        data_freq = pd.infer_freq(train_val_df.index) or 'H'
        forecast = prophet_model.forecast(steps=len(test_df), last_date=train_val_df.index[-1], freq=data_freq)
        y_pred = forecast.values[:len(test_df)]
        y_true = test_df['close'].values

    coin_pre = CoinPreprocessor()
    explanation = explain_model_predictions(
        model_type='Prophet',
        model=prophet_model,
        preprocessor=coin_pre,
        symbol=symbol,
        test_df=test_df,
        n_samples=50
    )
    print(f"Top SHAP feature for {symbol} (Prophet): {explanation['features'][np.argmax(np.mean(np.abs(explanation['shap_values']), axis=0))]}")
    
    metrics = compute_forecast_metrics(y_true, y_pred)
    
    params = {
        'changepoint_prior_scale': prophet_model.changepoint_prior_scale,
        'seasonality_prior_scale': prophet_model.seasonality_prior_scale,
        'forecast_steps': forecast_steps
    }
    
    log_model_params_and_metrics('Prophet', symbol, params, metrics, str(prophet_model.model_dir))
    
    return {'metrics': metrics, 'params': params}


def evaluate_cnn_lstm(symbol: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, sequence_length: int = 30, forecast_horizon: int = 7, retrain_if_exists: bool = False, **kwargs):
    available_features = [col for col in train_df.select_dtypes(include=[np.number]).columns if col != 'time']
    feature_cols = available_features[:5]  
    
    forecaster = CNNLSTMForecaster(sequence_length, forecast_horizon, feature_cols=feature_cols)
    forecaster.prepare_data(train_df, val_df, test_df)

    forecaster.feature_cols = feature_cols 
    
    if len(forecaster.X_test) == 0:
        return {'metrics': {'mae': np.nan, 'r2': np.nan, 'rmse': np.nan}, 'params': {}}

    weights_path = Path(r"D:\python_projects\crypto-analytics-platform\src\modules\forecasting\models\saved\cnn-lstm\cnn_lstm_model.weights.h5")
    
    loaded = forecaster.load(str(weights_path), retrain_if_fails=not retrain_if_exists)
    if not loaded:
        print(f"Failed to load; skipping CNN-LSTM for {symbol}")
        return {'metrics': {'mae': np.nan, 'r2': np.nan, 'rmse': np.nan}, 'params': {}}
    
    y_pred = forecaster.predict()
    metrics = compute_forecast_metrics(forecaster.y_test, y_pred, multi_horizon=True)
    
    try:
        coin_pre = CoinPreprocessor()
        explanation = explain_model_predictions(model_type='CNN-LSTM', model=forecaster, preprocessor=coin_pre, symbol=symbol, test_df=test_df, n_samples=20)
        print(f"Top SHAP for {symbol}: {explanation['features'][np.argmax(np.mean(np.abs(explanation['shap_values']), axis=0))]}")
    except Exception as e:
        print(f"SHAP skipped: {e}")
    
    params = {'sequence_length': sequence_length, 'feature_cols': feature_cols}
    log_model_params_and_metrics('CNN-LSTM', symbol, params, metrics, str(weights_path.parent))
    
    return {'metrics': metrics, 'params': params}

def evaluate_tft(
    symbol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_encoder_length: int = 30,
    max_prediction_length: int = 7,
    retrain_if_exists: bool = False,
    **kwargs
):
    forecaster = TFTForecaster(
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length
    )
    forecaster.prepare_data(train_df, val_df, test_df)
    if len(forecaster.test_df) == 0:
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}
    forecaster.build_model()
    
    checkpoint_dir = Path(r"D:\python_projects\crypto-analytics-platform\src\modules\forecasting\models\saved\tft")
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "tft_best.ckpt"
    
    loaded = False
    if best_checkpoint_path.exists() and not retrain_if_exists:
        try:
            checkpoint = torch.load(str(best_checkpoint_path), map_location=torch.device('cpu'))
            forecaster.tft.load_state_dict(checkpoint['state_dict'], strict=False)  
            print(f"Loaded TFT state_dict from {best_checkpoint_path} on CPU")
            loaded = True
        except Exception as e:
            print(f"Manual load failed ({e}), falling back to train on CPU")
    
    if not loaded:
        forecaster.train(max_epochs=50, checkpoint_dir=str(checkpoint_dir))
        shutil.copy(forecaster.trainer.checkpoint_callback.best_model_path, best_checkpoint_path)
        forecaster.tft = TemporalFusionTransformer.load_from_checkpoint(str(best_checkpoint_path), map_location=torch.device('cpu'))
        print(f"Trained and saved new TFT to {best_checkpoint_path}")

    preds_raw = forecaster.tft.predict(forecaster.test_dataloader, return_y=True, mode="raw")
    median_idx = next(i for i, q in enumerate(forecaster.tft.loss.quantiles) if q == 0.5)
    y_pred = preds_raw.output[0][:, median_idx, :]
    y_true = preds_raw.y[0] if isinstance(preds_raw.y, (tuple, list)) else preds_raw.y

    # coin_pre = CoinPreprocessor()
    # explanation = explain_model_predictions(
    #     model_type='TFT',
    #     model=forecaster,
    #     preprocessor=coin_pre,  # FIXED: Coin instead of Panel
    #     symbol=symbol,
    #     test_df=test_df,
    #     n_samples=50
    # )
    # print(f"Top SHAP feature for {symbol} (TFT): {explanation['features'][np.argmax(np.mean(np.abs(explanation['shap_values']), axis=0))]}")
    
    metrics = compute_forecast_metrics(y_true, y_pred, multi_horizon=True)
    
    params = {
        'max_encoder_length': max_encoder_length,
        'max_prediction_length': max_prediction_length,
        'hidden_size': forecaster.hidden_size,
        'learning_rate': forecaster.learning_rate
    }
    
    log_model_params_and_metrics('TFT', symbol, params, metrics, str(checkpoint_dir))
    
    return {'metrics': metrics, 'params': params}

def run_benchmark(symbol: str = 'BTCUSDT', exchange: str = 'binance', interval: str = '1h', 
                  forecast_steps: int = 7, rolling_eval: bool = True, retrain_if_exists: bool = False):
    init_mlflow_experiment()
    
    coin_pre = CoinPreprocessor()
    df_feat = coin_pre.load_features_series(symbol, exchange=exchange, interval=interval)
    
    train_df, val_df, test_df = split_data_for_evaluation(df_feat)

    if 'symbol' not in train_df.columns:
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df['symbol'] = symbol
        val_df['symbol'] = symbol
        test_df['symbol'] = symbol
    if 'time' not in train_df.columns:
        train_df['time'] = train_df.index
        val_df['time'] = val_df.index
        test_df['time'] = test_df.index

    results = {}
    
    print(f"{ 'Rolling' if rolling_eval else 'Single'} OOS eval for {symbol}...")
    
    results['sarimax'] = evaluate_sarimax(symbol, train_df, val_df, test_df, forecast_steps, retrain_if_exists, rolling_eval=rolling_eval)
    print(f"SARIMAX MAE: {results['sarimax']['metrics']['mae']:.4f}")
    
    results['prophet'] = evaluate_prophet(symbol, train_df, val_df, test_df, forecast_steps, retrain_if_exists, rolling_eval=rolling_eval)
    print(f"Prophet MAE: {results['prophet']['metrics']['mae']:.4f}")
    
    # results['cnn_lstm'] = evaluate_cnn_lstm(symbol, train_df, val_df, test_df, forecast_horizon=forecast_steps, retrain_if_exists=retrain_if_exists)
    # print(f"CNN-LSTM MAE: {results['cnn_lstm']['metrics']['mae']:.4f}")
    
    # results['tft'] = evaluate_tft(symbol, train_df, val_df, test_df, max_prediction_length=forecast_steps, retrain_if_exists=retrain_if_exists)
    # print(f"TFT MAE: {results['tft']['metrics']['mae']:.4f}")
    
    summary = pd.DataFrame([
        {'Model': 'SARIMAX', 'MAE': results['sarimax']['metrics']['mae'], 'RMSE': results['sarimax']['metrics']['rmse'], 'R2': results['sarimax']['metrics']['r2'], 'MAPE': results['sarimax']['metrics']['mape'], 'Directional Acc': results['sarimax']['metrics']['directional_acc']},
        {'Model': 'Prophet', 'MAE': results['prophet']['metrics']['mae'], 'RMSE': results['prophet']['metrics']['rmse'], 'R2': results['prophet']['metrics']['r2'], 'MAPE': results['prophet']['metrics']['mape'], 'Directional Acc': results['prophet']['metrics']['directional_acc']},
        # {'Model': 'CNN-LSTM', 'MAE': results['cnn_lstm']['metrics']['mae'], 'RMSE': results['cnn_lstm']['metrics']['rmse'], 'R2': results['cnn_lstm']['metrics']['r2'], 'MAPE': results['cnn_lstm']['metrics']['mape'], 'Directional Acc': results['cnn_lstm']['metrics']['directional_acc']},
        # {'Model': 'TFT', 'MAE': results['tft']['metrics']['mae'], 'RMSE': results['tft']['metrics']['rmse'], 'R2': results['tft']['metrics']['r2'], 'MAPE': results['tft']['metrics']['mape'], 'Directional Acc': results['tft']['metrics']['directional_acc']}
    ])
    print("\nBenchmark Summary:\n", summary)
    
    return results