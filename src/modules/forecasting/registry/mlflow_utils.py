import mlflow
import os
from typing import Dict, Any
from datetime import datetime
import numpy as np


def init_mlflow_experiment(experiment_name: str = "crypto_forecasting"):
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: {experiment_name} (ID: {experiment_id})")


def log_model_params_and_metrics(
    model_type: str,
    symbol: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts_path: str = None
):
    if symbol == 'multiple' and 'symbols' in params:
        symbol_str = f"panel_{len(params['symbols'])}_symbols"
    else:
        symbol_str = symbol

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{symbol_str}_{timestamp}"
    
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("symbol", symbol_str)
            
            for key, value in params.items():
                if isinstance(value, (list, dict)):
                    mlflow.log_param(f"{key}", str(value))
                else:
                    mlflow.log_param(f"{key}", value)

            for key, value in metrics.items():
                if isinstance(value, (list, np.ndarray)):
                    for i, v in enumerate(value):
                        if np.isfinite(v): 
                            mlflow.log_metric(f"{key}_{i+1}", float(v))
                else:
                    if np.isfinite(value): 
                        mlflow.log_metric(key, float(value))

            if artifacts_path and os.path.exists(artifacts_path):
                mlflow.log_artifacts(artifacts_path)
            
            print(f"Logged run '{run_name}' to MLflow")
    except Exception as e:
        print(f"Failed to log to MLflow: {e}")