import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from pathlib import Path
import tensorflow as tf
import logging

from modules.forecasting.data.preprocess_panel import PanelPreprocessor

logger = logging.getLogger(__name__)

class CNNLSTMPanelForecaster:
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
            "returns", "sma_24", "ema_24", "volatility_24", "volume_zscore_30"
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
                    logger.info(f"Loaded panel data with symbols: {list(available_symbols)}")
                    return panel_df
                else:
                    logger.info("Panel data missing some symbols, loading from DB...")
            except Exception as e:
                logger.warning(f"Failed to load panel parquet: {e}, loading from DB...")

        logger.info(f"Loading panel directly from DB for {len(symbols)} symbols")
        engine = self.preprocessor.engine
        
        q = """
            SELECT time, symbol, feature_name, feature_value, exchange, interval
            FROM ohlcv_features_panel 
            WHERE symbol = ANY(%(symbols)s) AND exchange = %(exchange)s AND interval = %(interval)s
            ORDER BY time, symbol, feature_name;
        """
        df_long = pd.read_sql(q, engine, params={'symbols': symbols, 'exchange': exchange, 'interval': interval})
        
        if df_long.empty:
            raise ValueError("No panel data in DB - run export_panel_data first")
        
        df_pivot = df_long.pivot_table(
            index='time', columns=['symbol', 'feature_name'], 
            values='feature_value', aggfunc='first'
        )

        if isinstance(df_pivot.columns, pd.MultiIndex):
            loaded_symbols = list(df_pivot.columns.get_level_values(0).unique())
            self.symbols = [s for s in symbols if s in loaded_symbols]
            logger.info(f"Loaded symbols from DB: {self.symbols}")
        else:
            self.symbols = symbols
        
        global_start = df_pivot.index.min()
        global_end = df_pivot.index.max()
        full_idx = pd.date_range(global_start, global_end, freq='h', tz='UTC')
        df_pivot = df_pivot.reindex(full_idx).ffill().bfill()
        
        self.preprocessor.save_panel_parquet(df_pivot, self.panel_parquet_path)
        logger.info(f"Loaded and cached panel from DB: {df_pivot.shape}, symbols: {self.symbols}")
        
        return df_pivot

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

    def create_panel_sequences(self, panel_dict):
        X_all, y_all = [], []
        
        for symbol, df in panel_dict.items():
            if len(df) < self.sequence_length + self.forecast_horizon:
                logger.debug(f"Skipping {symbol}: insufficient data ({len(df)} points)")
                continue
      
            available_features = [col for col in self.feature_cols if col in df.columns]
            if not available_features or self.target_col not in df.columns:
                logger.debug(f"Skipping {symbol}: missing features or target")
                continue
                
            feature_data = df[available_features].values
            target_data = df[self.target_col].values

            for i in range(self.sequence_length, len(df) - self.forecast_horizon + 1):
                X_all.append(feature_data[i - self.sequence_length:i])
                y_all.append(target_data[i:i + self.forecast_horizon])
        
        if len(X_all) == 0:
            raise ValueError("No sequences created - check data length and feature availability")
            
        X_array = np.array(X_all)
        y_array = np.array(y_all)
        
        logger.info(f"Created {len(X_array)} sequences with shape X: {X_array.shape}, y: {y_array.shape}")
        return X_array, y_array

    def prepare_training_data(
        self, 
        symbols: list,
        exchange: str = "binance",
        interval: str = "1h",
        force_update: bool = False
    ):
        panel_df = self.load_or_create_panel_data(symbols, exchange, interval, force_update)
        
        if panel_df.empty:
            raise ValueError("No panel data available for training")
        
        logger.info(f"Panel data loaded: {panel_df.shape}, symbols: {self.symbols}")
 
        train_panels, val_panels, test_panels = self.split_panel_by_dates(panel_df)

        X_train, y_train = self.create_panel_sequences(train_panels)
        X_val, y_val = self.create_panel_sequences(val_panels)
        X_test, y_test = self.create_panel_sequences(test_panels)
        
        logger.info(f"Training sequences - X: {X_train.shape if len(X_train) > 0 else 'Empty'}, y: {y_train.shape if len(y_train) > 0 else 'Empty'}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(
        self,
        symbols: list,
        epochs: int = 50,  
        batch_size: int = 32,
        exchange: str = "binance",
        interval: str = "1h",
        retrain_if_exists: bool = False,
        force_panel_update: bool = False
    ):
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

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        logger.info(f"Trained CNN-LSTM Panel model on {len(symbols)} symbols")

    def forecast(self, symbol: str, steps: int = None):
        if self.model is None:
            self.load()
        
        if steps is None:
            steps = self.forecast_horizon

        coin_pre = self.preprocessor.coin_pre
        df = coin_pre.load_features_series(symbol, exchange='binance', interval='1h')

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
        if not self.model_path.exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
        
        self.model = load_model(self.model_path)
        logger.info("Loaded CNN-LSTM Panel model")

    def save(self):
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(self.model_path)
        logger.info("Saved CNN-LSTM Panel model")

def train_and_forecast_cnn_lstm(
    symbols: list,
    df: pd.DataFrame = None,
    exchange: str = 'binance',
    interval: str = '1h',
    forecast_steps: int = 24,
    retrain_if_exists: bool = False,
    ensure_features: bool = True
):
    model = CNNLSTMPanelForecaster(forecast_horizon=forecast_steps)
    
    if model.model_path.exists() and not retrain_if_exists:
        model.load()
        logger.info("Loaded existing CNN-LSTM model")
    else:
        logger.info("Training new CNN-LSTM model")
        model.train(symbols, exchange=exchange, interval=interval, retrain_if_exists=retrain_if_exists)

    if symbols:
        forecast = model.forecast(symbols[0], steps=forecast_steps)

        coin_pre = PanelPreprocessor().coin_pre
        history_df = coin_pre.load_features_series(symbols[0], exchange=exchange, interval=interval)
        
        return {'forecast': forecast, 'history': history_df['close']}
    else:
        raise ValueError("No symbols provided for forecasting")