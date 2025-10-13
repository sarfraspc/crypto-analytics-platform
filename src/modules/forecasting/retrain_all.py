import logging
import argparse
from typing import List
from pathlib import Path

import pandas as pd

from core.database import get_timescale_engine
from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_panel import PanelPreprocessor
from modules.forecasting.models.sarimax import train_and_forecast as sarimax_train_and_forecast
from modules.forecasting.models.prophet import train_and_forecast as prophet_train_and_forecast
from modules.forecasting.models.cnn_lstm import train_and_forecast_cnn_lstm
from modules.forecasting.models.tft import train_and_forecast_tft

logger = logging.getLogger(__name__)

def get_all_symbols(exchange: str = "binance", min_data_points: int = 100):
    engine = get_timescale_engine()
    query = """
        SELECT DISTINCT symbol 
        FROM ohlcv 
        WHERE exchange = %(exchange)s 
        AND interval = '1h' 
        GROUP BY symbol 
        HAVING COUNT(*) > %(min_data_points)s
        ORDER BY symbol
    """
    df = pd.read_sql(query, engine, params={"exchange": exchange, "min_data_points": min_data_points})
    symbols = df["symbol"].tolist()
    logger.info(f"Found {len(symbols)} symbols with sufficient data")
    return symbols

def ensure_features(
    symbols: List[str],
    exchange: str = "binance",
    interval: str = "1h",
):
    coin_pre = CoinPreprocessor()
    
    successful_symbols = []
    
    for symbol in symbols:
        try:
            df_features = coin_pre.load_features_series(symbol, exchange, interval)
            if df_features is not None and not df_features.empty:
                successful_symbols.append(symbol)
                logger.info(f"Features verified for {symbol}: {len(df_features)} rows")
            else:
                logger.warning(f"No features found for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to verify features for {symbol}: {e}")
            continue

    logger.info(f"Features verified for {len(successful_symbols)}/{len(symbols)} symbols")
    
    logger.info("Panel data already exists in database - skipping panel creation")
    
    return successful_symbols

def retrain_individual_models(
    symbols: List[str],
    exchange: str = "binance",
    interval: str = "1h",
    forecast_steps: int = 24,
    models: List[str] = ["sarimax", "prophet"],
    retrain_if_exists: bool = False,
    batch_size: int = None  
):
    results = {}
    
    if batch_size and len(symbols) > batch_size:
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        logger.info(f"Processing {len(symbols)} symbols in {len(batches)} batches of {batch_size}")
    else:
        batches = [symbols]
    
    for model_name in models:
        results[model_name] = {}
        total_trained = 0
        total_failed = 0
        
        for batch_num, batch_symbols in enumerate(batches, 1):
            logger.info(f"Retraining {model_name} - Batch {batch_num}/{len(batches)} ({len(batch_symbols)} symbols)")
            
            for symbol in batch_symbols:
                try:
                    if model_name == "sarimax":
                        result = sarimax_train_and_forecast(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            forecast_steps=forecast_steps,
                            retrain_if_exists=retrain_if_exists,
                            ensure_features=False
                        )
                    elif model_name == "prophet":
                        result = prophet_train_and_forecast(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            forecast_steps=forecast_steps,
                            retrain_if_exists=retrain_if_exists,
                            ensure_features=False
                        )
                    else:
                        continue
                    
                    results[model_name][symbol] = result["forecast"]
                    total_trained += 1
                    if total_trained % 10 == 0: 
                        logger.info(f"{model_name}: {total_trained} symbols trained...")
                    
                except Exception as e:
                    logger.error(f"{model_name} failed for {symbol}: {e}")
                    results[model_name][symbol] = None
                    total_failed += 1
        
        logger.info(f"{model_name}: {total_trained} successful, {total_failed} failed")
    
    return results

def retrain_panel_models(
    symbols: List[str],
    exchange: str = "binance",
    interval: str = "1h",
    forecast_steps: int = 24,
    models: List[str] = ["cnn_lstm", "tft"],
    retrain_if_exists: bool = False
):
    results = {}
    
    if not symbols:
        logger.warning("No symbols provided for panel models")
        return results
    
    logger.info(f"Training panel models with {len(symbols)} symbols: {symbols[:5]}...")  
    
    for model_name in models:
        try:
            logger.info(f"Retraining {model_name} panel model...")
            
            if model_name == "cnn_lstm":
                result = train_and_forecast_cnn_lstm(
                    symbols=symbols,
                    exchange=exchange,
                    interval=interval,
                    forecast_steps=forecast_steps,
                    retrain_if_exists=retrain_if_exists,
                    ensure_features=False  
                )
            elif model_name == "tft":
                result = train_and_forecast_tft(
                    symbols=symbols,
                    exchange=exchange,
                    interval=interval,
                    forecast_steps=forecast_steps,
                    retrain_if_exists=retrain_if_exists,
                    ensure_features=False  
                )
            else:
                continue
            
            results[model_name] = result
            logger.info(f"{model_name} panel model trained")
            
        except Exception as e:
            logger.error(f"{model_name} panel model failed: {e}")
            results[model_name] = {"status": "failed", "error": str(e)}
    
    return results

def retrain_all_models(
    exchange: str = "binance", 
    interval: str = "1h",
    forecast_steps: int = 24,
    models: List[str] = ["sarimax", "prophet", "cnn_lstm", "tft"],
    retrain_if_exists: bool = False,
    min_data_points: int = 100,
    force_feature_update: bool = True,
    individual_batch_size: int = 50,  
    max_panel_symbols: int = 50  
):
    logger.info("Starting comprehensive model retraining for ALL symbols...")
    
    all_symbols = get_all_symbols(
        exchange=exchange, 
        min_data_points=min_data_points
    )
    
    if not all_symbols:
        logger.error("No symbols found for training")
        return {}
    
    logger.info(f"Processing {len(all_symbols)} total symbols")
    
    successful_symbols = all_symbols
    if force_feature_update:
        successful_symbols = ensure_features(all_symbols, exchange, interval)
        logger.info(f"Features verified for {len(successful_symbols)}/{len(all_symbols)} symbols")
    
    if not successful_symbols:
        logger.error("No symbols with successful feature verification")
        return {}
    
    results = {
        "individual_models": {},
        "panel_models": {},
        "summary": {
            "total_symbols": len(all_symbols),
            "successful_feature_symbols": len(successful_symbols),
            "individual_symbols": len(successful_symbols),
            "panel_symbols": min(len(successful_symbols), max_panel_symbols)
        }
    }
    
    individual_models = [m for m in models if m in ["sarimax", "prophet"]]
    if individual_models:
        logger.info(f"Training individual models for {len(successful_symbols)} symbols...")
        results["individual_models"] = retrain_individual_models(
            symbols=successful_symbols,
            exchange=exchange,
            interval=interval,
            forecast_steps=forecast_steps,
            models=individual_models,
            retrain_if_exists=retrain_if_exists,
            batch_size=individual_batch_size
        )
    
    panel_models = [m for m in models if m in ["cnn_lstm", "tft"]]
    if panel_models and successful_symbols:
        panel_symbols = successful_symbols[:max_panel_symbols]
        logger.info(f"Training panel models with {len(panel_symbols)} symbols...")
        results["panel_models"] = retrain_panel_models(
            symbols=panel_symbols,
            exchange=exchange,
            interval=interval,
            forecast_steps=forecast_steps,
            models=panel_models,
            retrain_if_exists=retrain_if_exists
        )
    
    successful_individual = 0
    failed_individual = 0
    
    for model_type, model_results in results["individual_models"].items():
        successful = len([v for v in model_results.values() if v is not None])
        failed = len([v for v in model_results.values() if v is None])
        successful_individual += successful
        failed_individual += failed
        logger.info(f"{model_type}: {successful} successful, {failed} failed")
    
    successful_panel = 0
    for model_type, model_result in results["panel_models"].items():
        if model_result and "forecast" in model_result:
            successful_panel += 1
            logger.info(f"{model_type}: SUCCESS")
        else:
            logger.info(f"{model_type}: FAILED")
    
    results["summary"].update({
        "successful_individual": successful_individual,
        "failed_individual": failed_individual,
        "successful_panel": successful_panel,
        "individual_success_rate": (successful_individual / (successful_individual + failed_individual)) * 100 if (successful_individual + failed_individual) > 0 else 0,
        "status": "completed"
    })
    
    logger.info("Model retraining completed!")
    logger.info(f"Individual models: {successful_individual} successful, {failed_individual} failed")
    logger.info(f"Panel models: {successful_panel} successful")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Retrain all forecasting models for ALL symbols')
    parser.add_argument('--exchange', default='binance', help='Exchange name')
    parser.add_argument('--interval', default='1h', help='Data interval')
    parser.add_argument('--forecast-steps', type=int, default=24, help='Forecast horizon')
    parser.add_argument('--models', nargs='+', default=['sarimax', 'prophet', 'cnn_lstm', 'tft'], 
                       help='Models to retrain')
    parser.add_argument('--min-data-points', type=int, default=100, 
                       help='Minimum data points required')
    parser.add_argument('--individual-batch-size', type=int, default=50,
                       help='Batch size for individual model training')
    parser.add_argument('--max-panel-symbols', type=int, default=50,
                       help='Maximum symbols for panel models')
    parser.add_argument('--retrain-if-exists', action='store_true', 
                       help='Retrain even if model exists')
    parser.add_argument('--skip-feature-update', action='store_true', 
                       help='Skip feature update')
    
    args = parser.parse_args()
    
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'retrain_all.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        results = retrain_all_models(
            exchange=args.exchange,
            interval=args.interval,
            forecast_steps=args.forecast_steps,
            models=args.models,
            retrain_if_exists=args.retrain_if_exists,
            min_data_points=args.min_data_points,
            force_feature_update=not args.skip_feature_update,
            individual_batch_size=args.individual_batch_size,
            max_panel_symbols=args.max_panel_symbols
        )
        
        print("\n" + "="*60)
        print("RETRAINING SUMMARY")
        print("="*60)
        summary = results.get("summary", {})
        print(f"Total Symbols Found: {summary.get('total_symbols', 0)}")
        print(f"Symbols with Features: {summary.get('successful_feature_symbols', 0)}")
        print(f"Individual Models: {summary.get('successful_individual', 0)} successful, {summary.get('failed_individual', 0)} failed")
        print(f"Individual Success Rate: {summary.get('individual_success_rate', 0):.1f}%")
        print(f"Panel Models: {summary.get('successful_panel', 0)} successful")
        print("="*60)
        
    except Exception as e:
        logger.exception("Retraining failed with exception")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        print("Running retraining for ALL symbols with default parameters...")
        results = retrain_all_models(
            exchange="binance",
            interval="1h", 
            forecast_steps=24,
            models=["sarimax", "prophet", "cnn_lstm", "tft"],
            retrain_if_exists=False,
            min_data_points=100,
            individual_batch_size=50,
            max_panel_symbols=15
        )
        
        if results:
            summary = results.get("summary", {})
            print(f"Retraining completed! {summary.get('successful_individual', 0)} individual models trained.")
        else:
            print("Retraining failed!")
    else:
        sys.exit(main())