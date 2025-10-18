import mlflow
import logging
from typing import Dict, Any, Optional
from core.config import settings  
logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name: str = "crypto-rag-experiments"):
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.info(f"Set MLflow tracking URI: {mlflow.get_tracking_uri()}")

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            experiment = mlflow.get_experiment_by_name(experiment_name)
        else:
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")

        mlflow.set_experiment(experiment_name)
        logger.info("MLflow configuration completed")
        
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise
    
def start_rag_run(run_name: str = "rag_experiment", params: Optional[Dict[str, Any]] = None):
    run = mlflow.start_run(run_name=run_name)
    if params:
        for k, v in params.items():
            mlflow.log_param(k, v)
    return run

def log_rag_metrics(metrics: Dict[str, float]):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def end_rag_run():
    mlflow.end_run()