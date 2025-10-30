import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.forecasting.data.preprocess_coin import CoinPreprocessor
from modules.forecasting.data.preprocess_panel import PanelPreprocessor
from modules.forecasting.registry.mlflow_utils import log_model_params_and_metrics  

logger = logging.getLogger(__name__)

MODEL_SHAP_STRATEGY = {
    "SARIMAX": "kernel",      
    "Prophet": "kernel",      
    "CNN-LSTM": "deep",      
    "TFT": "deep",           
}


class SHAPExplainer:
    def __init__(
        self,
        model: Any,
        model_type: str,
        preprocessor: Union[CoinPreprocessor, PanelPreprocessor],
        symbol: str,
        explain_dir: Optional[Union[str, Path]] = None,
        n_background: int = 100,  
        max_evals: int = 500,    
        feature_names: Optional[List[str]] = None,
        target_col: str = "close",
    ):
        self.model = model
        self.model_type = model_type.upper()
        self.preprocessor = preprocessor
        self.symbol = symbol.upper()
        self.target_col = target_col
        self.explain_dir = Path(explain_dir or f"src/modules/forecasting/models/saved/{model_type.lower()}/explanations")
        self.explain_dir.mkdir(parents=True, exist_ok=True)
        
        if feature_names is None:
            try:
                sample_df = preprocessor.load_features_series(symbol, start=None, end=None)
                all_numeric = sample_df.select_dtypes(include=[np.number]).columns.tolist()
                self.feature_names = [c for c in all_numeric if c != target_col] 
                logger.info(f"Inferred {len(self.feature_names)} features (excluded {target_col})")
            except Exception as e:
                logger.warning(f"Could not infer features: {e}. Using generic names.")
                self.feature_names = [f"feature_{i}" for i in range(10)] 
        else:
            self.feature_names = feature_names

        self.strategy = MODEL_SHAP_STRATEGY.get(self.model_type, "kernel")
        self.explainer = None
        self._setup_explainer(n_background, max_evals)

    def _setup_explainer(self, n_background: int, max_evals: int):
        background_df = self.preprocessor.load_features_series(self.symbol).tail(n_background)
        background_data = background_df[self.feature_names].values  
        
        if self.strategy == "kernel":
            if self.model_type == "SARIMAX":
                def predict_fn(X): 
                    X_df = pd.DataFrame(X, columns=self.feature_names)
                    weights = np.abs(np.random.normal(0, 0.05, len(self.feature_names)))  
                    base = self.model.model_fit.forecast(steps=1).mean() 
                    adjustments = np.sum(X_df.values * weights, axis=1)
                    return base + adjustments
            elif self.model_type == "PROPHET":  
                 def predict_fn(X):  
                    X_df = pd.DataFrame(X, columns=self.feature_names)
                    base_forecast = self.model.model.predict(self.model.model.make_future_dataframe(periods=1))['yhat'].iloc[-1]
                    vol_cols = [c for c in self.feature_names if 'volatility' in c]
                    if vol_cols:
                        vol_impact = X_df[vol_cols].mean(axis=1).values * 0.1 
                    else:
                        vol_impact = np.zeros(X.shape[0])
                    return base_forecast + vol_impact
            else:
                raise ValueError(f"No predict_fn for {self.model_type}")
            
            self.explainer = shap.KernelExplainer(predict_fn, background_data)
            self.explainer.nsamples = max_evals  

        elif self.strategy == "deep":
            if self.model_type == "CNN-LSTM":
                self.explainer = shap.DeepExplainer(self.model.model, np.random.random((n_background, self.model.sequence_length, len(self.model.feature_cols))))
            elif self.model_type == "TFT":
                n_background = 10  
                encoder_len = self.model.max_encoder_length
                decoder_len = self.model.max_prediction_length
                n_vars = len(self.model.time_varying_unknown_reals)
                
                dummy_x = {
                    'encoder_lengths': torch.tensor([encoder_len] * n_background, dtype=torch.long),
                    'decoder_lengths': torch.tensor([decoder_len] * n_background, dtype=torch.long),
                    'encoder_cat': torch.zeros(n_background, encoder_len, 0, dtype=torch.long),  
                    'decoder_cat': torch.zeros(n_background, decoder_len, 0, dtype=torch.long),
                    'encoder_cont': torch.rand(n_background, encoder_len, n_vars, dtype=torch.float32),
                    'decoder_cont': torch.rand(n_background, decoder_len, n_vars, dtype=torch.float32),
                    'groups': torch.zeros(n_background, dtype=torch.long),  
                }
                self.explainer = shap.GradientExplainer(self.model.tft, dummy_x)
            else:
                raise ValueError(f"No deep explainer for {self.model_type}")

        logger.info(f"SHAP Explainer initialized for {self.model_type} using {self.strategy} strategy. Features: {len(self.feature_names)}")

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        nsamples: Optional[int] = None,
    ):
        input_X_original = X  
        if isinstance(X, pd.DataFrame):
            if self.target_col in X.columns:
                X = X.drop(columns=[self.target_col])
            X = X[self.feature_names].values  
        else:
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"X shape[1]={X.shape[1]} != len(features)={len(self.feature_names)}. Align first!")
        
        logger.info(f"Explaining X shape: {X.shape} with {len(self.feature_names)} features")

        if self.strategy == "kernel":
            shap_values = self.explainer.shap_values(X, nsamples=nsamples or self.explainer.nsamples)
            if isinstance(shap_values, list):  
                shap_values = shap_values[0]  
        else:  
            shap_values = self.explainer.shap_values(X)

        if len(shap_values.shape) == 3:
            shap_values = np.squeeze(shap_values, axis=-1)

        if self.model_type in ["SARIMAX", "Prophet"]:
            predictions = np.full(X.shape[0], np.mean(shap_values, axis=1))
        elif self.model_type == "CNN-LSTM":
            predictions = self.model.model.predict(X).flatten()
        elif self.model_type == "TFT":
            raw_pred = self.model.tft.predict(pd.DataFrame(X), return_y=False, mode="raw")
            median_idx = next(i for i, q in enumerate(self.model.tft.loss.quantiles) if q == 0.5)
            predictions = raw_pred.output[0][:, median_idx, :].flatten()
        else:
            predictions = None

        explanation = {
            "shap_values": shap_values,
            "expected_value": self.explainer.expected_value,
            "predictions": predictions,
            "features": self.feature_names,
            "true_y": y.values if y is not None else None,
            "input_X": input_X_original 
        }

        logger.info(f"Computed SHAP values for {X.shape[0]} samples: mean |SHAP| = {np.mean(np.abs(shap_values)):.4f}")
        return explanation

    def plot(
        self,
        explanation: Dict[str, Any],
        sample_idx: int = 0,
        plot_type: str = "force", 
        save: bool = True,
        X_input: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        feature: Optional[str] = None,
    ):
        shap_values = explanation["shap_values"]
        feature_names = explanation["features"]
        
        x_plot_data = explanation.get("input_X", None) if X_input is None else X_input
        if x_plot_data is None:
            x_plot_data = np.zeros((len(shap_values), len(feature_names)))
        if isinstance(x_plot_data, pd.DataFrame):
            if self.target_col in x_plot_data.columns:
                x_plot_data = x_plot_data.drop(columns=[self.target_col])
            X_df = x_plot_data[self.feature_names]  
        else:
            X_df = pd.DataFrame(x_plot_data[:, :len(feature_names)], columns=feature_names)  

        if plot_type == "force":
            fig = shap.force_plot(
                explanation["expected_value"],
                shap_values[sample_idx:sample_idx+1],
                X_df.iloc[sample_idx:sample_idx+1] if hasattr(X_df, 'iloc') else pd.DataFrame(X_df[sample_idx:sample_idx+1], columns=feature_names),
                matplotlib=False,
                show=False,
            )
            if save:
                plot_path = self.explain_dir / f"{self.symbol}_{self.model_type}_force_plot_sample_{sample_idx}.html"
                shap.save_html(str(plot_path), fig)  
                logger.info(f"Saved force plot to {plot_path}")
            return None 

        elif plot_type == "summary":
            shap.summary_plot(
                shap_values,
                features=X_df,
                feature_names=feature_names,
                plot_type="dot", 
                show=False,
            )
            if save:
                plot_path = self.explain_dir / f"{self.symbol}_{self.model_type}_summary_plot.png"
                plt.gcf().savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved summary plot to {plot_path}")
            fig = plt.gcf()
            plt.close(fig)
            return fig

        elif plot_type == "waterfall":
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explanation["expected_value"],
                    data=X_df.iloc[sample_idx] if hasattr(X_df, 'iloc') else X_df[sample_idx],
                    feature_names=feature_names,
                ),
                show=False,
            )
            if save:
                plot_path = self.explain_dir / f"{self.symbol}_{self.model_type}_waterfall_sample_{sample_idx}.png"
                plt.gcf().savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved waterfall plot to {plot_path}")
            fig = plt.gcf()
            plt.close(fig)
            return fig

        elif plot_type == "dependence":
            if feature is None:
                feature = feature_names[0]  
            shap.dependence_plot(
                feature,
                shap_values,
                features=X_df,
                feature_names=feature_names,
                show=False,
            )
            if save:
                plot_path = self.explain_dir / f"{self.symbol}_{self.model_type}_dependence_{feature}.png"
                plt.gcf().savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved dependence plot to {plot_path}")
            fig = plt.gcf()
            plt.close(fig)
            return fig

        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        return None

    def log_to_mlflow(self, explanation: Dict[str, Any], run_id: Optional[str] = None):
        shap_path = self.explain_dir / f"{self.symbol}_{self.model_type}_shap_values.npy"
        np.save(shap_path, explanation["shap_values"])

        shap_metrics = {
            f"mean_abs_shap_{feat}": np.mean(np.abs(explanation["shap_values"][:, i]))
            for i, feat in enumerate(explanation["features"])
        }
        shap_metrics["total_shap_variance"] = np.var(explanation["shap_values"])

        log_model_params_and_metrics(
            f"{self.model_type}_SHAP",
            self.symbol,
            {"strategy": self.strategy, "n_features": len(self.feature_names)},
            shap_metrics,
            str(self.explain_dir),
        )

        logger.info(f"Logged SHAP explanations to MLflow for {self.symbol}")


def explain_model_predictions(
    model_type: str,
    model: Any,
    preprocessor: Union[CoinPreprocessor, PanelPreprocessor],
    symbol: str,
    test_df: pd.DataFrame,
    n_samples: int = 100,
    **kwargs
):
    explainer = SHAPExplainer(model, model_type, preprocessor, symbol)
    
    if model_type.upper() in ['CNN-LSTM', 'TFT']:
        if hasattr(model, 'X_test') and len(model.X_test) > 0:
            X_sample = model.X_test[:n_samples]  
            y_sample = model.y_test[:n_samples].flatten() if hasattr(model, 'y_test') else None
            if X_sample.shape[2] != len(explainer.feature_names):
                explainer.feature_names = [f"{feat}_t{i}" for i in range(X_sample.shape[1]) for feat in model.feature_cols[:X_sample.shape[2]]] 
        else:
            feature_cols = [c for c in test_df.select_dtypes(include=[np.number]).columns if c != explainer.target_col][:X_sample.shape[1]]
            X_sample = test_df[feature_cols].iloc[:n_samples].values
            y_sample = test_df[explainer.target_col].iloc[:n_samples]
    else:
        feature_cols = [c for c in test_df.select_dtypes(include=[np.number]).columns if c != explainer.target_col]
        X_sample = test_df[feature_cols].iloc[:n_samples]
        y_sample = test_df[explainer.target_col].iloc[:n_samples]
    
    explanation = explainer.explain(X_sample, y_sample)
    
    if len(explanation['shap_values'].shape) == 3:
        explanation['shap_values'] = np.mean(explanation['shap_values'], axis=2)  
    
    explainer.plot(explanation, plot_type="summary", save=True)
    explainer.plot(explanation, sample_idx=0, plot_type="force", save=True)
    
    explainer.log_to_mlflow(explanation)
    
    return explanation