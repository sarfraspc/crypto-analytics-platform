import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
from modules.forecasting.data.preprocess_panel import PanelPreprocessor

class CNNLSTMForecaster:
    def __init__(self, sequence_length=30, forecast_horizon=7, feature_cols=None, target_col="close"):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols or ["close", "ema_8", "returns", "sma_7", "volume_zscore_30"]
        self.target_col = target_col
        self.model = None
        self.history = None
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.symbol_sequences_train = []
        self.symbol_sequences_val = []
        self.symbol_sequences_test = []

    def prepare_data(self, train_df, val_df, test_df):
        for split_group, X_list, y_list, sym_list in [
            (train_df, self.X_train, self.y_train, self.symbol_sequences_train),
            (val_df, self.X_val, self.y_val, self.symbol_sequences_val),
            (test_df, self.X_test, self.y_test, self.symbol_sequences_test)
        ]:
            for symbol, group in split_group.groupby("symbol"):
                data = group[self.feature_cols].values
                target = group[self.target_col].values

                for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                    X_list.append(data[i:i + self.sequence_length])
                    y_list.append(target[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
                    sym_list.append(symbol)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        print(f"Dataset created: Train X {self.X_train.shape}, Val {self.X_val.shape}, Test {self.X_test.shape}")
        print(f"y shapes: Train {self.y_train.shape}, Val {self.y_val.shape}, Test {self.y_test.shape}")

    def build_model(self):
        n_timesteps = self.sequence_length
        n_features = len(self.feature_cols)
        n_outputs = self.forecast_horizon

        self.model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            layers.LSTM(64, return_sequences=False, dropout=0.3),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_outputs)
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        self.model.summary()

    def train(self, epochs=100, batch_size=64, checkpoint_path=None):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        if checkpoint_path:
            callbacks.append(ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'))

        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )

    def load(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"No saved model at {path}")
        self.model = load_model(path)
        print(f"Loaded CNN-LSTM from {path}")

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(path)
        print(f"Saved CNN-LSTM to {path}")

def train_and_forecast(symbol: str, df: pd.DataFrame = None, exchange: str = 'binance', interval: str = '1h', forecast_horizon: int = 7, retrain_if_exists: bool = False, ensure_features: bool = True):
    if df is None:
        coin_pre = PanelPreprocessor()
        if ensure_features:
            coin_pre.update_features(symbol, exchange=exchange, interval=interval)
        df = coin_pre.load_features_series(symbol, exchange=exchange, interval=interval)

    if 'symbol' not in df.columns:
        df['symbol'] = symbol
    if 'time' not in df.columns:
        df['time'] = df.index
        
    forecaster = CNNLSTMForecaster(forecast_horizon=forecast_horizon)
    
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    forecaster.prepare_data(train_df, val_df, test_df)
    forecaster.build_model()
    
    checkpoint_path = Path(f"checkpoints/cnn_lstm_{symbol}.h5")
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    if checkpoint_path.exists() and not retrain_if_exists:
        forecaster.load(str(checkpoint_path))
    else:
        forecaster.train(epochs=50, checkpoint_path=str(checkpoint_path))
        forecaster.save(str(checkpoint_path))
    
    y_pred = forecaster.model.predict(forecaster.X_test)
    
    return {'forecast': y_pred, 'history': forecaster.y_test}