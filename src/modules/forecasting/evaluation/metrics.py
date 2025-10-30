import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
from typing import Dict, Union, List


def compute_forecast_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
    multi_horizon: bool = False
):
    y_true = np.asarray(y_true).flatten() if not multi_horizon else np.asarray(y_true)
    y_pred = np.asarray(y_pred).flatten() if not multi_horizon else np.asarray(y_pred)
    
    if multi_horizon:
        maes = [mean_absolute_error(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])]
        rmses = [np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h])) for h in range(y_true.shape[1])]
        r2s = [r2_score(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])] 
        
        mapes = []
        for h in range(y_true.shape[1]):
            true_h = y_true[:, h]
            pred_h = y_pred[:, h]
            mape_h = np.mean(np.abs((true_h - pred_h) / np.where(true_h == 0, 1, true_h))) * 100
            mapes.append(mape_h)

        dir_accs = []
        for h in range(y_true.shape[1]):
            true_h_diff = np.diff(y_true[:, h])
            pred_h_diff = np.diff(y_pred[:, h])
            if len(true_h_diff) > 0:
                dir_acc_h = np.mean(np.sign(true_h_diff) == np.sign(pred_h_diff)) * 100
            else:
                dir_acc_h = np.nan
            dir_accs.append(dir_acc_h)

        return {
            'mae': np.mean(maes),
            'rmse': np.mean(rmses),
            'r2': np.mean(r2s),  
            'mape': np.mean(mapes),
            'directional_acc': np.mean(dir_accs),
            'mae_per_horizon': maes,
            'rmse_per_horizon': rmses,
            'r2_per_horizon': r2s,  
            'mape_per_horizon': mapes,
            'directional_acc_per_horizon': dir_accs
        }
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)  
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        directional_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100 if len(np.diff(y_true)) > 0 else np.nan
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'directional_acc': directional_acc}  