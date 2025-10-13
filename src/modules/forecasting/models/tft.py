import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from pathlib import Path
import logging

from modules.forecasting.data.preprocess_panel import PanelPreprocessor

logger = logging.getLogger(__name__)

class TFTPanelForecaster:
    def __init__(
        self,
        max_encoder_length: int = 168,  
        max_prediction_length: int = 24,  
        target: str = "close",
        time_varying_known_reals: list = None,
        hidden_size: int = 64,
        learning_rate: float = 3e-3,
        dropout: float = 0.1,
        model_dir: str = "src/modules/forecasting/models/saved/tft",
        panel_parquet_path: str = "data/panel_cache/hourly_panel.parquet"
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.target = target
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        self.time_varying_known_reals = time_varying_known_reals or [
            "dayofweek", "month", "hour", "is_month_start"
        ]
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "tft_panel.ckpt"
        self.panel_parquet_path = Path(panel_parquet_path)
        self.panel_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.training = None
        self.tft = None
        self.trainer = None
        self.preprocessor = PanelPreprocessor()

    def load_or_create_panel_data(
        self, 
        symbols: list,
        exchange: str = "binance",
        interval: str = "1h",
        force_update: bool = False
    ):
        if self.panel_parquet_path.exists() and not force_update:
            try:
                logger.info(f"Loading panel data from {self.panel_parquet_path}")
                panel_df = pd.read_parquet(self.panel_parquet_path)
                available_symbols = panel_df['symbol'].unique()
                if all(sym in available_symbols for sym in symbols):
                    logger.info(f"Loaded panel data with symbols: {available_symbols}")
                    return panel_df
                else:
                    logger.info("Panel data missing some symbols, regenerating...")
            except Exception as e:
                logger.warning(f"Failed to load panel parquet: {e}, regenerating...")
   
        logger.info(f"Generating new panel data for {len(symbols)} symbols using update_panel()")
        panel_df, _ = self.preprocessor.update_panel(
            symbols, exchange=exchange, interval=interval, target_freq="H"
        )
        
        if panel_df.empty:
            raise ValueError("No panel data generated - check symbol availability")

        self.preprocessor.save_panel_parquet(panel_df, self.panel_parquet_path)
        logger.info(f"Saved panel data to {self.panel_parquet_path}")
        
        return panel_df

    def prepare_tft_splits(self, panel_df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
        panel_df = panel_df.reset_index().rename(columns={'index': 'time'})
        panel_df = panel_df.sort_values(['symbol', 'time'])

        panel_df = self._add_tft_features(panel_df)

        max_time_idx = panel_df["time_idx"].max()
        train_end_idx = int(train_ratio * max_time_idx)
        val_end_idx = int((train_ratio + val_ratio) * max_time_idx)
        
        train_df = panel_df[panel_df["time_idx"] <= train_end_idx]
        val_df = panel_df[(panel_df["time_idx"] > train_end_idx) & (panel_df["time_idx"] <= val_end_idx)]
        test_df = panel_df[panel_df["time_idx"] > val_end_idx]
        
        logger.info(f"TFT splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def _add_tft_features(self, df):
        """Add TFT-required features to panel data"""
        df['hour'] = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['is_month_start'] = df['time'].dt.is_month_start.astype(int)
        df['time_idx'] = ((df['time'] - df['time'].min()).dt.total_seconds() // 3600).astype(int)
        return df

    def create_datasets(self, train_df, val_df, test_df):
        if train_df.empty:
            raise ValueError("Training data is empty")
            
        exclude_cols = ['time', 'symbol', 'time_idx', self.target]
        numeric_cols = [
            col for col in train_df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
        
        time_varying_unknown_reals = numeric_cols + [self.target]
        
        logger.info(f"TFT using {len(time_varying_unknown_reals)} unknown real features: {time_varying_unknown_reals}")
        
        self.training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=self.target,
            group_ids=["symbol"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["symbol"],
            time_varying_known_reals=["time_idx"] + self.time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            self.training, val_df, predict=True, stop_randomization=True
        )
        test = TimeSeriesDataSet.from_dataset(
            self.training, test_df, predict=True, stop_randomization=True
        )

        batch_size = 32  
        self.train_dataloader = self.training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        self.val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        self.test_dataloader = test.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        
        logger.info("Created TFT datasets and dataloaders")
        return validation, test

    def build_model(self):
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=4,
            dropout=self.dropout,
            hidden_continuous_size=32,
            output_size=7, 
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        logger.info(f"Built TFT Panel model: output_size={self.tft.output_size}, max_prediction_length={self.max_prediction_length}")
        return self.tft

    def train(
        self,
        symbols: list,
        max_epochs: int = 20, 
        exchange: str = "binance",
        interval: str = "1h",
        retrain_if_exists: bool = False,
        force_panel_update: bool = False
    ):
        if self.model_path.exists() and not retrain_if_exists:
            self.load()
            logger.info("Using existing TFT model")
            return

        panel_df = self.load_or_create_panel_data(symbols, exchange, interval, force_panel_update)

        train_df, val_df, test_df = self.prepare_tft_splits(panel_df)

        validation, test = self.create_datasets(train_df, val_df, test_df)

        self.build_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"), 
            LearningRateMonitor(),
            ModelCheckpoint(
                dirpath=str(self.model_dir),
                filename="tft_panel_checkpoint_{epoch}_{val_loss:.4f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ]

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1, 
            callbacks=callbacks,
            logger=TensorBoardLogger(str(self.model_dir / "logs")),
            gradient_clip_val=0.1,
        )

        self.trainer.fit(
            self.tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )

        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            self.tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        self.save()

        logger.info(f"Trained TFT Panel model on {len(symbols)} symbols")

    def calculate_metrics(self, dataset, predictions):
        try:
            actuals = torch.cat([y for _, y in dataset])
            predictions = predictions.cpu()

            median_idx = 3 
            median_predictions = predictions[..., median_idx]
            
            mae = torch.mean(torch.abs(median_predictions - actuals)).item()
            mse = torch.mean((median_predictions - actuals) ** 2).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            }
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {'mae': 0, 'mse': 0, 'rmse': 0}

    def forecast(self, symbol: str, steps: int = None):
        if self.tft is None:
            self.load()
        
        if steps is None:
            steps = self.max_prediction_length

        coin_pre = self.preprocessor.coin_pre
        df = coin_pre.load_features_series(symbol, exchange='binance', interval='1h')

        prediction_data = self.prepare_prediction_data(df, symbol)

        raw_predictions = self.tft.predict(prediction_data, mode="raw")

        median_idx = 3
        forecast_values = raw_predictions.output[0][median_idx, -steps:].cpu().numpy()

        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1), 
            periods=steps, 
            freq='H'
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_dates, name='forecast')
        logger.info(f"Generated TFT forecast for {symbol}: {forecast_series.shape}")
        return forecast_series

    def prepare_prediction_data(self, df: pd.DataFrame, symbol: str):
        df = df.reset_index().rename(columns={'index': 'time'})
        df['symbol'] = symbol

        df = self._add_tft_features(df)
        
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self.training, df, predict=True, stop_randomization=True
        )
        
        return prediction_dataset.to_dataloader(train=False, batch_size=1)

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
        
        self.tft = TemporalFusionTransformer.load_from_checkpoint(str(self.model_path))
     
        if self.training is None:
            dummy_data = pd.DataFrame({
                'symbol': ['DUMMY'],
                'time_idx': [0],
                'close': [100.0],
                'time': [pd.Timestamp.now()],
                'hour': [12],
                'dayofweek': [1],
                'month': [1],
                'is_month_start': [0]
            })
            
            self.training = TimeSeriesDataSet(
                dummy_data,
                time_idx="time_idx",
                target="close",
                group_ids=["symbol"],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=["symbol"],
                time_varying_known_reals=["time_idx", "hour", "dayofweek", "month", "is_month_start"],
                time_varying_unknown_reals=["close"],
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
        
        logger.info("Loaded TFT Panel model")

    def save(self):
        if self.tft is None:
            raise RuntimeError("No model to save")
        
        if self.trainer:
            self.trainer.save_checkpoint(str(self.model_path))
        else:
            torch.save(self.tft.state_dict(), str(self.model_path))
            
        logger.info(f"Saved TFT Panel model to {self.model_path}")

def train_and_forecast_tft(
    symbols: list,
    df: pd.DataFrame = None,
    exchange: str = 'binance',
    interval: str = '1h',
    forecast_steps: int = 24,
    retrain_if_exists: bool = False,
    ensure_features: bool = True
):
    model = TFTPanelForecaster(max_prediction_length=forecast_steps)
    
    if model.model_path.exists() and not retrain_if_exists:
        model.load()
        logger.info("Loaded existing TFT model")
    else:
        logger.info("Training new TFT model")
        model.train(symbols, exchange=exchange, interval=interval, retrain_if_exists=retrain_if_exists)

    if symbols:
        forecast = model.forecast(symbols[0], steps=forecast_steps)

        coin_pre = PanelPreprocessor().coin_pre
        history_df = coin_pre.load_features_series(symbols[0], exchange=exchange, interval=interval)
        
        return {'forecast': forecast, 'history': history_df['close']}
    else:
        raise ValueError("No symbols provided for forecasting")