import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from pathlib import Path
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class CNNLSTMForecaster:
    def __init__(self, sequence_length=30, forecast_horizon=7, feature_cols=None, target_col="close"):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols or ["close", "ema_8", "returns", "sma_7", "volume_zscore_30"]
        self.target_col = target_col
        self.model = None
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []

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

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.summary()
        return self.model

    def load(self, path: str, retrain_if_fails: bool = True):
        if not Path(path).exists():
            raise FileNotFoundError(f"No file at {path}")

        if path.endswith('.h5') and 'weights' in path.lower():
            try:
                self.build_model() 
                self.model.load_weights(path)

                dummy_X = tf.random.normal((1, self.sequence_length, len(self.feature_cols)))
                _ = self.model.predict(dummy_X, verbose=0)
                
                logger.info(f"Loaded weights from {path}; validated shape")
                return True
            except Exception as e:
                logger.warning(f"Weights load/validation failed: {e}")

        try:
            self.model = load_model(path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info(f"Loaded full model from {path}")
            return True
        except Exception as e:
            logger.warning(f"Full load failed: {e}")

        if retrain_if_fails:
            logger.info("Falling back to fresh training")
            self.train(epochs=50)
            return True
        else:
            raise ValueError(f"All loads failed for {path}")

    def prepare_data(self, train_df, val_df, test_df):
        for split_group, X_list, y_list in [
            (train_df, self.X_train, self.y_train),
            (val_df, self.X_val, self.y_val),
            (test_df, self.X_test, self.y_test)
        ]:
            for symbol, group in split_group.groupby("symbol"):
                data = group[self.feature_cols].values
                target = group[self.target_col].values

                for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                    X_list.append(data[i:i + self.sequence_length])
                    y_list.append(target[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        print(f"Dataset created: Train X {self.X_train.shape}, Val {self.X_val.shape}, Test {self.X_test.shape}")

    def train(self, epochs=50, batch_size=64, checkpoint_path=None):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        if checkpoint_path:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'))

        self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )

    def save(self, path: str):
        self.model.save_weights(path)
        logger.info(f"Saved weights to {path}")

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.model.predict(X)