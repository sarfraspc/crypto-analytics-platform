import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import shutil
from pathlib import Path
from modules.forecasting.data.preprocess_panel import PanelPreprocessor

class TFTForecaster:
    def __init__(
        self,
        max_encoder_length=30,
        max_prediction_length=7,
        target="close",
        group_ids=["symbol"],
        time_idx="time_idx",
        static_categoricals=["symbol"],
        time_varying_known_reals=["time_idx", "dayofweek"],
        hidden_size=64,
        learning_rate=3e-3,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.target = target
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.static_categoricals = static_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.training_dataset = None
        self.tft = None
        self.trainer = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def prepare_data(self, train_df, val_df, test_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        for df in [self.train_df, self.val_df, self.test_df]:
            if self.time_idx not in df.columns:
                df[self.time_idx] = (df['time'] - df['time'].min()).dt.days.astype(int)
            if 'dayofweek' not in df.columns:
                df['dayofweek'] = df['time'].dt.dayofweek

        numeric_cols = self.train_df.select_dtypes(include='number').columns.tolist()
        self.time_varying_unknown_reals = [c for c in numeric_cols if c not in [self.time_idx]]

        print("Numeric (candidate) columns:", self.time_varying_unknown_reals)

        self.training_dataset = TimeSeriesDataSet(
            self.train_df,
            time_idx=self.time_idx,
            target=self.target,
            group_ids=self.group_ids,
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=self.static_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=self.group_ids),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )

        batch_size = 64
        num_workers = 0
        self.train_dataloader = self.training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        val_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, self.val_df, predict=True, stop_randomization=True)
        self.val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
        test_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, self.test_df, predict=True, stop_randomization=True)
        self.test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    def build_model(self):
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=self.max_prediction_length,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        print("TFT summary:")
        print(self.tft)

    def train(self, max_epochs=50, checkpoint_dir="checkpoints"):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            dirpath=checkpoint_dir
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
        lr_logger = LearningRateMonitor()
        
        log_dir = Path("src/logs/tft_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(str(log_dir), name="tft")  

        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            logger=logger
        )

        self.trainer.fit(
            self.tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

        best_path = checkpoint_callback.best_model_path
        print("Best checkpoint path:", best_path)
        self.tft = TemporalFusionTransformer.load_from_checkpoint(best_path, map_location='cpu') 

def train_and_forecast(symbol: str, df: pd.DataFrame = None, exchange: str = 'binance', interval: str = '1h', max_prediction_length: int = 7, retrain_if_exists: bool = False, ensure_features: bool = True):
    if df is None:
        coin_pre = PanelPreprocessor()
        if ensure_features:
            coin_pre.update_features(symbol, exchange=exchange, interval=interval)
        df = coin_pre.load_features_series(symbol, exchange=exchange, interval=interval)

    if 'symbol' not in df.columns:
        df['symbol'] = symbol
    if 'time' not in df.columns:
        df['time'] = df.index

    forecaster = TFTForecaster(max_prediction_length=max_prediction_length)
    
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    forecaster.prepare_data(train_df, val_df, test_df)
    forecaster.build_model()
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"tft_{symbol}_best.ckpt"
    
    if best_checkpoint_path.exists() and not retrain_if_exists:
        forecaster.tft = TemporalFusionTransformer.load_from_checkpoint(str(best_checkpoint_path))
        print(f"Loaded TFT from {best_checkpoint_path}")
    else:
        forecaster.train(max_epochs=50, checkpoint_dir=str(checkpoint_dir))
        shutil.copy(forecaster.trainer.checkpoint_callback.best_model_path, best_checkpoint_path)
        forecaster.tft = TemporalFusionTransformer.load_from_checkpoint(str(best_checkpoint_path))

    preds_raw = forecaster.tft.predict(forecaster.test_dataloader, return_y=True, mode="raw")
    median_idx = next(i for i, q in enumerate(forecaster.tft.loss.quantiles) if q == 0.5)
    y_pred = preds_raw.output[0][:, median_idx, :]
    
    return {'forecast': y_pred, 'history': preds_raw.y[0]}
