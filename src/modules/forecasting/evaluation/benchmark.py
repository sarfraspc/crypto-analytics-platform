"""Benchmarking utilities for comparing forecasting model performance."""

import logging
from typing import Any, List

import numpy as np
import pandas as pd

from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_panel import PanelPreprocessor
from modules.forecasting.evaluation.metrics import compute_forecast_metrics
from modules.forecasting.explainers.xai import explain_model_predictions
from modules.forecasting.models.cnn_lstm import CNNLSTMPanelForecaster
from modules.forecasting.models.prophet import ProphetModel
from modules.forecasting.models.sarimax import SarimaxModel
from modules.forecasting.models.tft import TFTPanelForecaster
from modules.forecasting.registry.mlflow_utils import init_mlflow_experiment, log_model_params_and_metrics

logger = logging.getLogger(__name__)


def split_data_for_evaluation(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1):
    """Split time series data into train, validation, and test sets."""
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


def evaluate_cnn_lstm_panel(
    symbol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_steps: int = 24,
    retrain_if_exists: bool = False,
    panel_symbols: List[str] = None,
    **kwargs
):
    if len(test_df) == 0:
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}

    if panel_symbols is None:
        panel_symbols = [symbol, 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    try:
        cnn_model = CNNLSTMPanelForecaster(forecast_horizon=forecast_steps)
        
        if not cnn_model.model_path.exists():
            logger.warning(f"No pre-trained CNN-LSTM panel model found at {cnn_model.model_path}")
            return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}
        
        cnn_model.load()
        
        forecast = cnn_model.forecast(symbol, steps=forecast_steps)
        
        if len(forecast) > 0 and len(test_df) >= forecast_steps:
            y_pred = forecast.values
            y_true = test_df['close'].head(forecast_steps).values
            metrics = compute_forecast_metrics(y_true, y_pred)
        else:
            metrics = {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}

        try:
            panel_pre = PanelPreprocessor()
            explanation = explain_model_predictions(
                model_type='CNN-LSTM',
                model=cnn_model,
                preprocessor=panel_pre,
                symbol=symbol,
                test_df=test_df,
                n_samples=20
            )
            print(f"Top SHAP feature for {symbol} (CNN-LSTM): {explanation['features'][np.argmax(np.mean(np.abs(explanation['shap_values']), axis=0))]}")
        except Exception as e:
            logger.warning(f"SHAP explanation failed for CNN-LSTM: {e}")
        
        params = {
            'sequence_length': cnn_model.sequence_length,
            'forecast_horizon': cnn_model.forecast_horizon,
            'panel_symbols': panel_symbols,
            'feature_cols': cnn_model.feature_cols[:5]  
        }
        
        log_model_params_and_metrics('CNN-LSTM-Panel', symbol, params, metrics, str(cnn_model.model_dir))
        
        return {'metrics': metrics, 'params': params}
        
    except Exception as e:
        logger.error(f"CNN-LSTM evaluation failed: {e}")
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}


def evaluate_tft_panel(
    symbol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_steps: int = 24,
    retrain_if_exists: bool = False,
    panel_symbols: List[str] = None,
    **kwargs
):
    if len(test_df) == 0:
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}

    if panel_symbols is None:
        panel_symbols = [symbol, 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    try:
        tft_model = TFTPanelForecaster(max_prediction_length=forecast_steps)
        
        if not tft_model.model_path.exists():
            logger.warning(f"No pre-trained TFT panel model found at {tft_model.model_path}")
            return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}
        
        tft_model.load()
        
        forecast = tft_model.forecast(symbol, steps=forecast_steps)

        if len(forecast) > 0 and len(test_df) >= forecast_steps:
            y_pred = forecast.values
            y_true = test_df['close'].head(forecast_steps).values
            metrics = compute_forecast_metrics(y_true, y_pred)
        else:
            metrics = {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}
 
        try:
            panel_pre = PanelPreprocessor()
            explanation = explain_model_predictions(
                model_type='TFT',
                model=tft_model,
                preprocessor=panel_pre,
                symbol=symbol,
                test_df=test_df,
                n_samples=20
            )
            print(f"Top SHAP feature for {symbol} (TFT): {explanation['features'][np.argmax(np.mean(np.abs(explanation['shap_values']), axis=0))]}")
        except Exception as e:
            logger.warning(f"SHAP explanation failed for TFT: {e}")
        
        params = {
            'max_encoder_length': tft_model.max_encoder_length,
            'max_prediction_length': tft_model.max_prediction_length,
            'panel_symbols': panel_symbols,
            'hidden_size': tft_model.hidden_size
        }
        
        log_model_params_and_metrics('TFT-Panel', symbol, params, metrics, str(tft_model.model_dir))
        
        return {'metrics': metrics, 'params': params}
        
    except Exception as e:
        logger.error(f"TFT evaluation failed: {e}")
        return {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_acc': np.nan}, 'params': {}}


def run_benchmark(
    symbol: str = 'BTC',
    exchange: str = 'kraken',
    interval: str = '1h',
    forecast_steps: int = 24,
    rolling_eval: bool = False,
    retrain_if_exists: bool = False,
    models: List[str] = None,
    panel_symbols: List[str] = None
):
    """Run comprehensive benchmark across all forecasting models."""
    if models is None:
        models = ['sarimax', 'prophet', 'cnn_lstm', 'tft']
    
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
    
    print(f"Running benchmark for {symbol}...")
    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if 'sarimax' in models:
        print("Evaluating SARIMAX...")
        results['sarimax'] = evaluate_sarimax(
            symbol, train_df, val_df, test_df, forecast_steps, 
            retrain_if_exists, rolling_eval=rolling_eval
        )
        print(f"SARIMAX MAE: {results['sarimax']['metrics']['mae']:.4f}")
    
    if 'prophet' in models:
        print("Evaluating Prophet...")
        results['prophet'] = evaluate_prophet(
            symbol, train_df, val_df, test_df, forecast_steps,
            retrain_if_exists, rolling_eval=rolling_eval
        )
        print(f"Prophet MAE: {results['prophet']['metrics']['mae']:.4f}")

    if 'cnn_lstm' in models:
        print("Evaluating CNN-LSTM Panel...")
        results['cnn_lstm'] = evaluate_cnn_lstm_panel(
            symbol, train_df, val_df, test_df, forecast_steps,
            retrain_if_exists, panel_symbols=panel_symbols
        )
        print(f"CNN-LSTM MAE: {results['cnn_lstm']['metrics']['mae']:.4f}")
    
    if 'tft' in models:
        print("Evaluating TFT Panel...")
        results['tft'] = evaluate_tft_panel(
            symbol, train_df, val_df, test_df, forecast_steps,
            retrain_if_exists, panel_symbols=panel_symbols
        )
        print(f"TFT MAE: {results['tft']['metrics']['mae']:.4f}")

    summary_data = []
    for model_name in models:
        if model_name in results and 'metrics' in results[model_name]:
            metrics = results[model_name]['metrics']
            summary_data.append({
                'Model': model_name.upper(),
                'MAE': metrics.get('mae', np.nan),
                'RMSE': metrics.get('rmse', np.nan),
                'R2': metrics.get('r2', np.nan),
                'MAPE': metrics.get('mape', np.nan),
                'Directional Acc': metrics.get('directional_acc', np.nan)
            })
    
    summary = pd.DataFrame(summary_data)
    print("\nBenchmark Summary:")
    print(summary.to_string(index=False))
    
    return results


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Run forecasting model benchmark')
    parser.add_argument('--symbol', default='BTC', help='Symbol to benchmark')
    parser.add_argument('--exchange', default='kraken', help='Exchange name')
    parser.add_argument('--interval', default='1h', help='Data interval')
    parser.add_argument('--forecast-steps', type=int, default=24, help='Forecast horizon')
    parser.add_argument('--models', nargs='+', default=['sarimax', 'prophet', 'cnn_lstm', 'tft'], 
                       help='Models to evaluate')
    parser.add_argument('--rolling-eval', action='store_true', help='Use rolling evaluation')
    parser.add_argument('--retrain-if-exists', action='store_true', help='Retrain models if they exist')
    parser.add_argument('--panel-symbols', nargs='+', help='Symbols for panel models training')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        symbol=args.symbol,
        exchange=args.exchange,
        interval=args.interval,
        forecast_steps=args.forecast_steps,
        rolling_eval=args.rolling_eval,
        retrain_if_exists=args.retrain_if_exists,
        models=args.models,
        panel_symbols=args.panel_symbols
    )