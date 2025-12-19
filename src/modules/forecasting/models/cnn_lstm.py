"""CNN-LSTM hybrid model for multi-asset panel forecasting."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

from modules.forecasting.data.preprocess_panel import PanelPreprocessor
from utils.gcs_loader import load_from_gcs, upload_to_gcs

logger = logging.getLogger(__name__)


class CNNLSTMPanelForecaster:
    """CNN-LSTM model for cross-asset panel price forecasting."""
    def __init__(
        self,
        sequence_length: int = 168,  
        forecast_horizon: int = 24,  
        feature_cols: list = None,
        target_col: str = "close",
        model_dir: str = "src/modules/forecasting/models/saved/cnn-lstm",
        panel_parquet_path: str = "data/panel_cache/hourly_panel.parquet"
    ):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols or [
            "open", "high", "low", "close", "volume", 
            "returns", "sma_7", "sma_21", "ema_8", "ema_20", 
            "volatility_7", "volatility_30", "volume_zscore"
        ]
        self.target_col = target_col
        self.symbols = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "cnn_lstm_panel.h5"
        self.panel_parquet_path = Path(panel_parquet_path)
        self.panel_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.preprocessor = PanelPreprocessor()

    def build_model(self, n_features: int):
        """Build CNN-LSTM architecture with specified feature count."""
        self.model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                         input_shape=(self.sequence_length, n_features)),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            layers.LSTM(64, return_sequences=False, dropout=0.3),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.forecast_horizon)
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logger.info(f"Built CNN-LSTM Panel model with {n_features} features")
        return self.model

    def load_or_create_panel_data(
        self,
        symbols: list,
        exchange: str = "kraken",
        interval: str = "1h",
        force_update: bool = False
    ):
        """Load panel data from ohlcv_features table (unscaled data)."""
        # Always load fresh from ohlcv_features to get unscaled data
        logger.info(f"Loading panel from ohlcv_features for {len(symbols)} symbols")
        
        df_dict = {}
        coin_pre = self.preprocessor.coin_pre
        
        for symbol in symbols:
            try:
                df = coin_pre.load_features_series(symbol, exchange, interval)
                if df is not None and not df.empty:
                    df_dict[symbol] = df
                    logger.debug(f"Loaded {symbol}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
                continue
        
        if not df_dict:
            raise ValueError("No data loaded for any symbols")
        
        self.symbols = list(df_dict.keys())
        logger.info(f"Loaded {len(self.symbols)} symbols from ohlcv_features")
        
        # Debug: check close values for first symbol
        first_sym = self.symbols[0]
        first_df = df_dict[first_sym]
        if 'close' in first_df.columns:
            close_vals = first_df['close'].dropna()
            logger.info(f"[DEBUG] {first_sym} close values - min: {close_vals.min():.2f}, max: {close_vals.max():.2f}, mean: {close_vals.mean():.2f}")
        
        # Align all dataframes to same time index
        global_start = min(df.index.min() for df in df_dict.values())
        global_end = max(df.index.max() for df in df_dict.values())
        full_idx = pd.date_range(global_start, global_end, freq='h', tz='UTC')
        
        # Create MultiIndex panel DataFrame
        panels = []
        for symbol, df in df_dict.items():
            df_aligned = df.reindex(full_idx).ffill().bfill()
            df_aligned.columns = pd.MultiIndex.from_product([[symbol], df_aligned.columns])
            panels.append(df_aligned)
        
        panel_df = pd.concat(panels, axis=1)
        logger.info(f"Created panel: {panel_df.shape}, symbols: {self.symbols}")
        
        return panel_df

    def split_panel_by_dates(self, panel_df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
        all_dates = sorted(panel_df.index.unique())
        if len(all_dates) == 0:
            raise ValueError("No dates available in panel data")
        
        train_end_idx = int(train_ratio * len(all_dates))
        val_end_idx = int((train_ratio + val_ratio) * len(all_dates))
        
        train_end = all_dates[train_end_idx]
        val_end = all_dates[val_end_idx]
        
        train_panels, val_panels, test_panels = {}, {}, {}
        
        if isinstance(panel_df.columns, pd.MultiIndex):
            symbols = list(panel_df.columns.get_level_values(0).unique())
        else:
            symbols = self.symbols or panel_df['symbol'].unique().tolist()
    
        for symbol in symbols:
            if isinstance(panel_df.columns, pd.MultiIndex):
                sym_cols = [(symbol, feat) for feat in self.feature_cols if (symbol, feat) in panel_df.columns]
                if not sym_cols:
                    logger.warning(f"No features for {symbol}, skipping")
                    continue
                symbol_data = panel_df[sym_cols].copy()
                symbol_data.columns = [feat for _, feat in sym_cols]
            else:
                symbol_data = panel_df[panel_df['symbol'] == symbol].copy().drop(columns=['symbol'])
            
            train_data = symbol_data[symbol_data.index <= train_end]
            val_data = symbol_data[(symbol_data.index > train_end) & (symbol_data.index <= val_end)]
            test_data = symbol_data[symbol_data.index > val_end]
            
            train_panels[symbol] = train_data
            val_panels[symbol] = val_data
            test_panels[symbol] = test_data
        
        logger.info(f"Split panel data: Train symbols: {len(train_panels)}, Val: {len(val_panels)}, Test: {len(test_panels)}")
        return train_panels, val_panels, test_panels

    def create_panel_sequences(self, panel_dict, raise_on_empty: bool = True):
        X_all, y_all = [], []
        
        for symbol, df in panel_dict.items():
            if len(df) < self.sequence_length + self.forecast_horizon:
                logger.debug(f"Skipping {symbol}: insufficient data ({len(df)} points)")
                continue
      
            available_features = [col for col in self.feature_cols if col in df.columns]
            if not available_features or self.target_col not in df.columns:
                logger.debug(f"Skipping {symbol}: missing features or target")
                continue
            
            # Get unique columns (avoid duplicating target if it's in features)
            all_cols = list(dict.fromkeys(available_features + [self.target_col]))
            
            # Replace inf with nan, then fill nan with 0
            df_clean = df[all_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_data = df_clean[available_features].values
            target_data = df_clean[self.target_col].values.flatten()  # Ensure 1D
            
            # Debug: log target statistics for first symbol
            if symbol == list(panel_dict.keys())[0]:
                logger.info(f"[DEBUG] {symbol} target stats - min: {target_data.min():.6f}, max: {target_data.max():.6f}, mean: {target_data.mean():.6f}, zeros: {(target_data == 0).sum()}/{len(target_data)}")
                logger.info(f"[DEBUG] {symbol} feature stats - shape: {feature_data.shape}, any_nan: {np.isnan(feature_data).any()}, any_inf: {np.isinf(feature_data).any()}")

            for i in range(self.sequence_length, len(df) - self.forecast_horizon + 1):
                X_all.append(feature_data[i - self.sequence_length:i])
                y_all.append(target_data[i:i + self.forecast_horizon])
        
        if len(X_all) == 0:
            if raise_on_empty:
                raise ValueError("No sequences created - check data length and feature availability")
            logger.warning("No sequences created for this split (insufficient data)")
            return np.array([]), np.array([])
            
        X_array = np.array(X_all, dtype=np.float32)
        y_array = np.array(y_all, dtype=np.float32)
        
        # Final safety check - replace any remaining nan/inf
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Created {len(X_array)} sequences with shape X: {X_array.shape}, y: {y_array.shape}")
        return X_array, y_array

    def prepare_training_data(
        self, 
        symbols: list,
        exchange: str = "kraken",
        interval: str = "1h",
        force_update: bool = False
    ):
        panel_df = self.load_or_create_panel_data(symbols, exchange, interval, force_update)
        
        if panel_df.empty:
            raise ValueError("No panel data available for training")
        
        logger.info(f"Panel data loaded: {panel_df.shape}, symbols: {self.symbols}")
        
        # Debug: log available columns and sample data
        if isinstance(panel_df.columns, pd.MultiIndex):
            sample_sym = self.symbols[0] if self.symbols else None
            if sample_sym:
                sym_cols = [c for c in panel_df.columns if c[0] == sample_sym]
                logger.info(f"[DEBUG] Sample columns for {sample_sym}: {[c[1] for c in sym_cols[:15]]}")
                if (sample_sym, 'close') in panel_df.columns:
                    close_vals = panel_df[(sample_sym, 'close')].dropna()
                    logger.info(f"[DEBUG] {sample_sym} close values - min: {close_vals.min():.6f}, max: {close_vals.max():.6f}, zeros: {(close_vals == 0).sum()}/{len(close_vals)}")
        
        train_panels, val_panels, test_panels = self.split_panel_by_dates(panel_df)

        X_train, y_train = self.create_panel_sequences(train_panels, raise_on_empty=True)
        X_val, y_val = self.create_panel_sequences(val_panels, raise_on_empty=False)
        X_test, y_test = self.create_panel_sequences(test_panels, raise_on_empty=False)
        
        logger.info(f"Training sequences - X: {X_train.shape if len(X_train) > 0 else 'Empty'}, y: {y_train.shape if len(y_train) > 0 else 'Empty'}")
        logger.info(f"Validation sequences - X: {X_val.shape if len(X_val) > 0 else 'Empty'}")
        logger.info(f"Test sequences - X: {X_test.shape if len(X_test) > 0 else 'Empty'}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(
        self,
        symbols: list,
        epochs: int = 50,
        batch_size: int = 32,
        exchange: str = "kraken",
        interval: str = "1h",
        retrain_if_exists: bool = False,
        force_panel_update: bool = False
    ):
        """Train CNN-LSTM model on panel data for multiple symbols."""
        if self.model_path.exists() and not retrain_if_exists:
            self.load()
            logger.info("Using existing CNN-LSTM model")
            return

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_training_data(
            symbols, exchange, interval, force_panel_update
        )

        if len(X_train) == 0:
            raise ValueError("No training sequences available")

        n_features = X_train.shape[2]
        self.build_model(n_features)

        # Handle case where validation data is empty
        has_validation = len(X_val) > 0
        
        if has_validation:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    str(self.model_path), save_best_only=True, monitor='val_loss'
                )
            ]
            validation_data = (X_val, y_val)
        else:
            logger.warning("No validation data available, training without validation")
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', factor=0.5, patience=5, min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    str(self.model_path), save_best_only=True, monitor='loss'
                )
            ]
            validation_data = None

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        # Save model locally and to GCS
        self.save()
        logger.info(f"Trained CNN-LSTM Panel model on {len(symbols)} symbols")

    def forecast(self, symbol: str, steps: int = None):
        """Generate price forecast for a symbol."""
        if self.model is None:
            self.load()

        if steps is None:
            steps = self.forecast_horizon

        coin_pre = self.preprocessor.coin_pre
        df = coin_pre.load_features_series(symbol, exchange='kraken', interval='1h')

        available_features = [col for col in self.feature_cols if col in df.columns]
        if not available_features:
            raise ValueError(f"No features available for {symbol}")
            
        feature_data = df[available_features].values
        
        if len(feature_data) < self.sequence_length:
            raise ValueError(f"Not enough data for {symbol}. Need {self.sequence_length} points, got {len(feature_data)}")

        last_sequence = feature_data[-self.sequence_length:].reshape(1, self.sequence_length, len(available_features))

        forecast_array = self.model.predict(last_sequence, verbose=0)
        
        forecast_values = forecast_array[0]  
    
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1), 
            periods=steps, 
            freq='H'
        )
        
        forecast_series = pd.Series(forecast_values[:steps], index=forecast_dates, name='forecast')
        logger.info(f"Generated forecast for {symbol}: {forecast_series.shape}")
        return forecast_series

    def load(self):
        """Load model from disk or GCS."""
        # Prefer local file if present
        if self.model_path.exists():
            self.model = load_model(self.model_path)
            logger.info(f"Loaded CNN-LSTM model from {self.model_path}")
            return True

        # Fallback: try to pull from GCS
        try:
            remote_key = "forecasting/cnn-lstm/cnn_lstm_panel.h5"
            local_path = load_from_gcs(remote_key, local_name=self.model_path.name)
            self.model = load_model(local_path)
            logger.info(f"Loaded CNN-LSTM model from GCS: {remote_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load CNN-LSTM model from GCS: {e}")
            return False

    def save(self):
        """Save model to disk and upload to GCS."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # Save locally
        self.model.save(self.model_path)
        logger.info(f"Saved CNN-LSTM model to {self.model_path}")
        
        # Upload to GCS
        try:
            remote_key = "forecasting/cnn-lstm/cnn_lstm_panel.h5"
            upload_to_gcs(self.model_path, remote_key)
            logger.info(f"Uploaded CNN-LSTM model to GCS: {remote_key}")
        except Exception as e:
            logger.warning(f"Failed to upload CNN-LSTM model to GCS: {e}")

def train_and_forecast_cnn_lstm(
    symbols: list,
    df: pd.DataFrame = None,
    exchange: str = 'kraken',
    interval: str = '1h',
    forecast_steps: int = 24,
    retrain_if_exists: bool = False,
    ensure_features: bool = True
):
    """Train CNN-LSTM model and generate forecast for first symbol."""
    model = CNNLSTMPanelForecaster(forecast_horizon=forecast_steps)
    
    if retrain_if_exists:
        # Force retrain
        logger.info("Training new CNN-LSTM model")
        model.train(symbols, exchange=exchange, interval=interval, retrain_if_exists=retrain_if_exists)
    else:
        # Try to load existing model (local or GCS)
        if not model.load():
            logger.info("No existing model found, training new CNN-LSTM model")
            model.train(symbols, exchange=exchange, interval=interval, retrain_if_exists=retrain_if_exists)
        else:
            logger.info("Using existing CNN-LSTM model")

    if symbols:
        forecast = model.forecast(symbols[0], steps=forecast_steps)

        coin_pre = PanelPreprocessor().coin_pre
        history_df = coin_pre.load_features_series(symbols[0], exchange=exchange, interval=interval)
        
        return {'forecast': forecast, 'history': history_df['close']}
    else:
        raise ValueError("No symbols provided for forecasting")