# import os
# import yaml
# import mlflow
# from mlflow.tracking import MlflowClient
# from core.config import settings
# import traceback

# LOCAL_MLRUNS_DIR = os.path.abspath("mlruns")  

# mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
# client = MlflowClient()

# def migrate_local_runs():
#     for exp_id in os.listdir(LOCAL_MLRUNS_DIR):
#         exp_path = os.path.join(LOCAL_MLRUNS_DIR, exp_id)
#         meta_file = os.path.join(exp_path, "meta.yaml")
#         if not os.path.exists(meta_file):
#             continue
        
#         with open(meta_file, "r") as f:
#             exp_meta = yaml.safe_load(f)
#         exp_name = exp_meta.get("name", f"experiment_{exp_id}")

#         try:
#             remote_exp = client.get_experiment_by_name(exp_name)
#             if remote_exp:
#                 remote_exp_id = remote_exp.experiment_id
#                 print(f"Using existing experiment: {exp_name} (ID: {remote_exp_id})")
#             else:
#                 remote_exp_id = client.create_experiment(exp_name)
#                 print(f"Created new experiment: {exp_name} (ID: {remote_exp_id})")
#         except Exception as e:
#             print(f"Error with experiment {exp_name}: {e}")
#             print(traceback.format_exc())
#             continue

#         print(f"Migrating experiment: {exp_name}")

#         for run_id in os.listdir(exp_path):
#             run_path = os.path.join(exp_path, run_id)
#             meta_path = os.path.join(run_path, "meta.yaml")
#             if not os.path.exists(meta_path):
#                 continue
            
#             with open(meta_path, "r") as f:
#                 run_meta = yaml.safe_load(f)
#             run_name = run_meta.get("run_name", run_id)
#             print(f"→ Migrating run: {run_name}")

#             try:
#                 with mlflow.start_run(experiment_id=remote_exp_id, run_name=run_name) as active_run:
#                     active_run_id = active_run.info.run_id
#                     print(f"View run {run_name} at: {settings.MLFLOW_TRACKING_URI.replace('http', 'http://localhost')}/#/experiments/{remote_exp_id}/runs/{active_run_id}")

#                     params_dir = os.path.join(run_path, "params")
#                     if os.path.exists(params_dir):
#                         for p in os.listdir(params_dir):
#                             param_path = os.path.join(params_dir, p)
#                             if os.path.isfile(param_path):
#                                 with open(param_path) as f:
#                                     value = f.read().strip()
#                                     param_key = p.replace('.txt', '')
#                                     mlflow.log_param(param_key, value)
#                                     print(f"Logged param: {param_key} = {value}")

#                     metrics_dir = os.path.join(run_path, "metrics")
#                     if os.path.exists(metrics_dir):
#                         for m in os.listdir(metrics_dir):
#                             metric_path = os.path.join(metrics_dir, m)
#                             if os.path.isfile(metric_path):
#                                 with open(metric_path) as f:
#                                     lines = f.read().splitlines()
#                                     values = []
#                                     for line in lines:
#                                         if line.strip():
#                                             parts = line.split()
#                                             if len(parts) > 1 and parts[1].replace('.', '').isdigit():
#                                                 values.append(float(parts[1]))
#                                     if values:
#                                         last_value = values[-1]
#                                         metric_key = m.replace('.txt', '')
#                                         mlflow.log_metric(metric_key, last_value)
#                                         print(f"Logged metric: {metric_key} = {last_value}")

#                     artifacts_dir = os.path.join(run_path, "artifacts")
#                     if os.path.exists(artifacts_dir):
#                         mlflow.log_artifacts(artifacts_dir, recursive=True)
#                         print(f"Logged artifacts from {artifacts_dir}")

#             except Exception as e:
#                 print(f"Error migrating run {run_name}: {e}")
#                 print(traceback.format_exc())

#     print("Migration complete! Check your MLflow UI at:", settings.MLFLOW_TRACKING_URI.replace('http', 'http://localhost'))

# if __name__ == "__main__":
#     migrate_local_runs()