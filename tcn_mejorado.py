#!/usr/bin/env python3
"""
🚀 SISTEMA MEJORADO DE TRADING - SOLUCIONANDO PROBLEMAS CRÍTICOS
Versión optimizada con mejores métricas y rendimiento real
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import talib
import warnings
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

# Importar motor centralizado de features
from centralized_features_engine2 import CentralizedFeaturesEngine

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfigMejorado:
    """Configuración mejorada para trading rentable"""

    # === CONFIGURACIÓN BÁSICA ===
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    lookback_periods: int = 24  # Reducido para mejor aprendizaje
    prediction_horizon: int = 6

    # === DATOS ===
    training_days: Optional[int] = 60  # Reducido para datos más recientes
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # === MODELO MEJORADO ===
    model_type: str = "tcn_optimizado"
    feature_set: str = "tcn_definitivo"
    scaler_type: str = "robust"

    # === ENTRENAMIENTO OPTIMIZADO ===
    test_size: float = 0.15  # Más datos para train
    validation_size: float = 0.10
    epochs: int = 150  # Más épocas pero con early stopping
    batch_size: int = 64  # Batch size más grande
    learning_rate: float = 0.001  # Learning rate más alto
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8

    # === UMBRALES OPTIMIZADOS ===
    buy_threshold: float = 0.55  # Umbrales más bajos y realistas
    sell_threshold: float = 0.55
    min_signal_strength: float = 0.6  # Menos restrictivo

    # === COSTOS REALISTAS ===
    commission_rate: float = 0.001
    spread_cost: float = 0.0003
    slippage_cost: float = 0.0002

    def __post_init__(self):
        self.validate_config()

    def validate_config(self):
        """Validación básica"""
        if self.lookback_periods < 5:
            raise ValueError("lookback_periods debe ser >= 5")
        if self.training_days and self.training_days < 7:
            raise ValueError("training_days debe ser >= 7")
        logger.info("✅ Configuración mejorada validada")

    @property
    def total_trading_cost(self) -> float:
        return self.commission_rate + self.spread_cost + self.slippage_cost

    @property
    def min_profitable_move(self) -> float:
        return self.total_trading_cost * 1.5  # Margen más realista


class TradingSystemMejorado:
    """Sistema de trading mejorado con mejor rendimiento"""

    def __init__(self, config: TradingConfigMejorado):
        self.config = config
        self.scaler = None
        self.model = None
        self.feature_columns = []

        # Motor de features
        self.features_engine = CentralizedFeaturesEngine()
        self.scaler = self._create_scaler()

        logger.info(f"🚀 Sistema Mejorado Iniciado: {config.symbol} - {config.timeframe}")
        logger.info(f"📊 Lookback: {config.lookback_periods} | Batch: {config.batch_size} | LR: {config.learning_rate}")

    def _create_scaler(self):
        """Crear escalador"""
        scalers = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
        return scalers.get(self.config.scaler_type, RobustScaler())

    async def fetch_market_data(self) -> pd.DataFrame:
        """Obtener datos de mercado optimizado"""

        # Calcular fechas
        if self.config.end_date:
            end_time = int(datetime.strptime(self.config.end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_time = int(datetime.now().timestamp() * 1000)

        if self.config.start_date:
            start_time = int(datetime.strptime(self.config.start_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            start_time = int((datetime.now() - timedelta(days=self.config.training_days)).timestamp() * 1000)

        logger.info(f"📥 Descargando {self.config.symbol} - Últimos {self.config.training_days} días")

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.config.symbol,
            'interval': self.config.timeframe,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        all_data = []
        async with aiohttp.ClientSession() as session:
            current_start = start_time

            while current_start < end_time:
                params['startTime'] = current_start

                try:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            logger.error(f"API Error: {response.status}")
                            break

                        data = await response.json()
                        if not data:
                            break

                        all_data.extend(data)
                        current_start = data[-1][6] + 1

                except Exception as e:
                    logger.error(f"Error obteniendo datos: {e}")
                    break

                await asyncio.sleep(0.1)

        if not all_data:
            raise ValueError("No se pudieron obtener datos de mercado")

        # Crear DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                  'taker_buy_quote', 'ignore']

        df = pd.DataFrame(all_data, columns=columns)

        # Limpiar y procesar
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        # Remover outliers menos agresivamente
        for col in numeric_cols:
            q1 = df[col].quantile(0.005)  # Menos agresivo
            q99 = df[col].quantile(0.995)
            df[col] = df[col].clip(lower=q1, upper=q99)

        df = df.dropna()

        logger.info(f"✅ Datos procesados: {len(df)} registros ({df.index.min()} a {df.index.max()})")

        return df

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features técnicos mejorados"""

        logger.info(f"🔧 Calculando features: {self.config.feature_set}")

        try:
            # Usar motor centralizado
            features = self.features_engine.calculate_features(df, self.config.feature_set)

            # Verificar calidad de features
            if features.empty or len(features.columns) < 10:
                logger.warning("Pocas features del motor centralizado, usando fallback")
                return self._calculate_enhanced_features(df)

            # Agregar features adicionales críticos
            features = self._add_critical_features(df, features)

            self.feature_columns = list(features.columns)
            logger.info(f"✅ Features calculadas: {len(self.feature_columns)}")

            return features

        except Exception as e:
            logger.error(f"Error en motor centralizado: {e}")
            return self._calculate_enhanced_features(df)

    def _add_critical_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Agregar features críticos para mejor rendimiento"""

        try:
            # Price momentum features
            features['price_momentum_5'] = df['close'].pct_change(5)
            features['price_momentum_10'] = df['close'].pct_change(10)
            features['price_momentum_20'] = df['close'].pct_change(20)

            # Volatility features
            features['volatility_5'] = df['close'].pct_change().rolling(5).std()
            features['volatility_10'] = df['close'].pct_change().rolling(10).std()
            features['volatility_20'] = df['close'].pct_change().rolling(20).std()

            # Volume features
            features['volume_momentum'] = df['volume'].pct_change(5)
            features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # Price level features
            features['price_position_bb'] = self._calculate_bb_position(df['close'])
            features['price_above_ma20'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)
            features['price_above_ma50'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)

            # Trend strength
            features['trend_strength'] = self._calculate_trend_strength(df['close'])

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            logger.error(f"Error agregando features críticos: {e}")
            return features

    def _calculate_bb_position(self, price_series, period=20):
        """Calcular posición en Bandas de Bollinger"""
        ma = price_series.rolling(period).mean()
        std = price_series.rolling(period).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)

        bb_position = (price_series - lower) / (upper - lower)
        return bb_position.clip(0, 1)

    def _calculate_trend_strength(self, price_series, period=20):
        """Calcular fuerza de tendencia"""
        ma = price_series.rolling(period).mean()
        trend = (price_series - ma) / ma
        return trend

    def _calculate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features mejorados como fallback"""

        features = pd.DataFrame(index=df.index)

        try:
            # Price features
            for period in [5, 10, 20]:
                features[f'returns_{period}'] = df['close'].pct_change(period)
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'price_std_{period}'] = df['close'].rolling(period).std()

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss.clip(lower=1e-8)
            features['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            # Volume
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']

            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            features['bb_upper'] = sma20 + (2 * std20)
            features['bb_lower'] = sma20 - (2 * std20)
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # Limpiar
            features = features.fillna(method='ffill').fillna(0)

            self.feature_columns = list(features.columns)
            logger.info(f"✅ Features fallback: {len(self.feature_columns)}")

            return features

        except Exception as e:
            logger.error(f"Error en features fallback: {e}")
            # Último recurso
            features['returns_1'] = df['close'].pct_change(1).fillna(0)
            features['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
            self.feature_columns = ['returns_1', 'sma_20']
            return features

    def create_realistic_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Crear etiquetas más realistas y menos conservadoras"""

        logger.info("🏷️ Creando etiquetas mejoradas...")

        # Calcular returns futuros para etiquetar
        future_returns = df['close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)

        labels = []

        for i in range(len(df)):
            if i < 30 or i >= len(df) - self.config.prediction_horizon:
                labels.append(1)  # HOLD
                continue

            # Return futuro real
            future_return = future_returns.iloc[i]

            if pd.isna(future_return):
                labels.append(1)
                continue

            # Condiciones del mercado actuales
            try:
                # RSI
                rsi_val = features.get('rsi_14', pd.Series(50, index=features.index)).iloc[i]

                # MACD
                macd_val = features.get('macd_histogram', pd.Series(0, index=features.index)).iloc[i]

                # Bollinger position
                bb_pos = features.get('bb_position', pd.Series(0.5, index=features.index)).iloc[i]

                # Volatility
                vol_cols = [col for col in features.columns if 'volatility' in col.lower() or 'std' in col.lower()]
                if vol_cols:
                    volatility = features[vol_cols[0]].iloc[i]
                else:
                    volatility = df['close'].pct_change().rolling(10).std().iloc[i]

                # Volume
                vol_ratio_cols = [col for col in features.columns if 'volume' in col.lower() and 'ratio' in col.lower()]
                if vol_ratio_cols:
                    vol_ratio = features[vol_ratio_cols[0]].iloc[i]
                else:
                    vol_ratio = 1.0

                # Momentum
                momentum_cols = [col for col in features.columns if 'momentum' in col.lower() or 'returns' in col.lower()]
                if momentum_cols:
                    momentum = features[momentum_cols[0]].iloc[i]
                else:
                    momentum = df['close'].pct_change(5).iloc[i]

                # Decisión basada en condiciones menos restrictivas
                min_profitable = self.config.min_profitable_move

                # Condiciones BUY (más flexibles)
                buy_conditions = [
                    future_return > min_profitable,  # Return futuro positivo
                    rsi_val < 70 if not pd.isna(rsi_val) else True,
                    macd_val > -0.001 if not pd.isna(macd_val) else True,
                    bb_pos < 0.9 if not pd.isna(bb_pos) else True,
                    volatility < 0.05 if not pd.isna(volatility) else True,
                    momentum > -0.01 if not pd.isna(momentum) else True
                ]

                # Condiciones SELL (más flexibles)
                sell_conditions = [
                    future_return < -min_profitable,  # Return futuro negativo
                    rsi_val > 30 if not pd.isna(rsi_val) else True,
                    macd_val < 0.001 if not pd.isna(macd_val) else True,
                    bb_pos > 0.1 if not pd.isna(bb_pos) else True,
                    volatility < 0.05 if not pd.isna(volatility) else True,
                    momentum < 0.01 if not pd.isna(momentum) else True
                ]

                # Scoring más flexible
                buy_score = sum(buy_conditions) / len(buy_conditions)
                sell_score = sum(sell_conditions) / len(sell_conditions)

                # Decisión con umbrales más bajos
                if buy_score >= self.config.min_signal_strength and buy_score > sell_score:
                    labels.append(2)  # BUY
                elif sell_score >= self.config.min_signal_strength and sell_score > buy_score:
                    labels.append(0)  # SELL
                else:
                    labels.append(1)  # HOLD

            except Exception as e:
                logger.debug(f"Error procesando etiqueta {i}: {e}")
                labels.append(1)

        # Verificar distribución
        df_labeled = df.copy()
        df_labeled['label'] = labels

        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        logger.info("📊 Distribución mejorada:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            logger.info(f"   {name}: {count} ({pct:.1f}%)")

        # Verificar que hay suficientes señales
        non_hold = label_counts.get(0, 0) + label_counts.get(2, 0)
        if non_hold / total < 0.1:  # Menos del 10% de señales
            logger.warning("⚠️ Muy pocas señales de trading, ajustando umbrales...")
            return self._create_more_aggressive_labels(df, features)

        return df_labeled

    def _create_more_aggressive_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Crear etiquetas más agresivas si hay muy pocas señales"""

        logger.info("🔥 Creando etiquetas más agresivas...")

        # Usar returns futuros directamente con umbrales más bajos
        future_returns = df['close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)

        # Umbrales más agresivos
        buy_threshold = self.config.min_profitable_move * 0.5  # Reducir a la mitad
        sell_threshold = -buy_threshold

        labels = []
        for i in range(len(df)):
            if i >= len(df) - self.config.prediction_horizon:
                labels.append(1)
                continue

            future_ret = future_returns.iloc[i]

            if pd.isna(future_ret):
                labels.append(1)
            elif future_ret > buy_threshold:
                labels.append(2)  # BUY
            elif future_ret < sell_threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD

        df_labeled = df.copy()
        df_labeled['label'] = labels

        # Verificar nueva distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        logger.info("📊 Distribución agresiva:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            logger.info(f"   {name}: {count} ({pct:.1f}%)")

        return df_labeled

    def prepare_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar secuencias optimizado"""

        logger.info(f"🔄 Preparando secuencias - Lookback: {self.config.lookback_periods}")

        # Seleccionar features numéricas y limpiar
        numeric_features = features.select_dtypes(include=[np.number])

        # Remover features con varianza cero
        var_filter = numeric_features.var() > 1e-8
        numeric_features = numeric_features.loc[:, var_filter]

        if numeric_features.empty:
            raise ValueError("No hay features numéricas válidas")

        # Actualizar feature columns
        self.feature_columns = list(numeric_features.columns)
        logger.info(f"   Features utilizadas: {len(self.feature_columns)}")

        # Manejar valores infinitos y NaN
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(method='ffill').fillna(0)

        # Escalar features
        try:
            features_scaled = self.scaler.fit_transform(numeric_features)
        except Exception as e:
            logger.error(f"Error escalando features: {e}")
            # Fallback: usar datos sin escalar pero normalizados
            features_scaled = (numeric_features - numeric_features.mean()) / numeric_features.std().clip(lower=1e-8)
            features_scaled = features_scaled.fillna(0).values

        # Crear secuencias
        X, y = [], []

        for i in range(self.config.lookback_periods, len(features_scaled)):
            sequence = features_scaled[i-self.config.lookback_periods:i]
            X.append(sequence)
            y.append(labels.iloc[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"   ✅ Secuencias creadas: X={X.shape}, y={y.shape}")

        # Verificar distribución de clases
        unique_labels, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        logger.info(f"   📊 Distribución: {class_distribution}")

        return X, y

    def create_optimized_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Crear modelo optimizado para mejor rendimiento"""

        logger.info(f"🧠 Creando modelo optimizado: {input_shape}")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),

            # Normalización inicial
            tf.keras.layers.LayerNormalization(),

            # Bloque 1: Captura de patrones locales
            tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            # Bloque 2: Patrones de mediano plazo
            tf.keras.layers.Conv1D(128, 3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),

            # Bloque 3: Patrones de largo plazo
            tf.keras.layers.Conv1D(256, 3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Bloque 4: Integración de patrones
            tf.keras.layers.Conv1D(128, 3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Attention mechanism simplificado
            tf.keras.layers.GlobalAveragePooling1D(),

            # Capas densas con regularización
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),

            # Salida con activación suave
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Compilar con optimizador mejorado
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=0.0001,
                beta_1=0.9,
                beta_2=0.999
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        total_params = model.count_params()
        logger.info(f"   ✅ Modelo creado: {total_params:,} parámetros")

        return model

    def temporal_train_test_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split temporal optimizado"""

        total_size = len(X)
        test_size = self.config.test_size
        validation_size = self.config.validation_size
        train_size = 1 - test_size - validation_size

        # Calcular índices
        train_idx = int(total_size * train_size)
        val_idx = int(total_size * (train_size + validation_size))

        # Split
        X_train = X[:train_idx]
        X_val = X[train_idx:val_idx]
        X_test = X[val_idx:]

        y_train = y[:train_idx]
        y_val = y[train_idx:val_idx]
        y_test = y[val_idx:]

        logger.info(f"📊 Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def calculate_enhanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray) -> Dict:
        """Calcular métricas mejoradas de trading"""

        logger.info("📊 Calculando métricas mejoradas...")

        # Métricas básicas
        accuracy = np.mean(y_true == y_pred)

        # Simulación de trading mejorada
        portfolio_value = 1.0
        trades = []
        position = 0  # 0=cash, 1=long, -1=short
        entry_price = 0

        for i in range(len(y_pred)):
            signal = y_pred[i]
            price = prices[i]

            # Condiciones de entrada más realistas
            if signal == 2 and position <= 0:  # BUY signal
                if position == -1:  # Close short first
                    pnl = (entry_price - price) / entry_price
                    portfolio_value *= (1 + pnl - self.config.total_trading_cost)
                    trades.append(('CLOSE_SHORT', price, portfolio_value))

                # Enter long
                entry_price = price
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = 1
                trades.append(('BUY', price, portfolio_value))

            elif signal == 0 and position >= 0:  # SELL signal
                if position == 1:  # Close long first
                    pnl = (price - entry_price) / entry_price
                    portfolio_value *= (1 + pnl - self.config.total_trading_cost)
                    trades.append(('CLOSE_LONG', price, portfolio_value))

                # Enter short
                entry_price = price
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = -1
                trades.append(('SELL', price, portfolio_value))

        # Cerrar posición final
        if position != 0:
            final_price = prices[-1]
            if position == 1:
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price
            portfolio_value *= (1 + pnl - self.config.total_trading_cost)
            trades.append(('CLOSE_FINAL', final_price, portfolio_value))

        # Calcular métricas
        total_return = (portfolio_value - 1.0) * 100
        num_trades = len(trades)

        # Win rate mejorado
        profitable_trades = 0
        losing_trades = 0

        for i in range(1, len(trades)):
            if trades[i][2] > trades[i-1][2]:
                profitable_trades += 1
            else:
                losing_trades += 1

        win_rate = (profitable_trades / max(1, profitable_trades + losing_trades)) * 100

        # Sharpe ratio mejorado
        if len(trades) > 2:
            returns = [trades[i][2] / trades[i-1][2] - 1 for i in range(1, len(trades))]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Anualizado
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Maximum Drawdown
        running_max = portfolio_value
        max_drawdown = 0
        for trade in trades:
            portfolio_val = trade[2]
            if portfolio_val > running_max:
                running_max = portfolio_val
            drawdown = (running_max - portfolio_val) / running_max
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        metrics = {
            'accuracy': accuracy,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'final_portfolio': portfolio_value,
            'profit_factor': profitable_trades / max(1, losing_trades)
        }

        logger.info("📈 Métricas finales:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")

        return metrics

    async def train(self) -> Dict:
        """Entrenamiento optimizado del sistema"""

        logger.info("🚀 Iniciando entrenamiento optimizado...")

        try:
            # 1. Datos
            df = await self.fetch_market_data()
            if len(df) < 100:
                raise ValueError(f"Insuficientes datos: {len(df)} registros")

            # 2. Features
            features = self.calculate_technical_features(df)

            # 3. Labels mejoradas
            df_labeled = self.create_realistic_labels(df, features)

            # 4. Secuencias
            X, y = self.prepare_sequences(features, df_labeled['label'])

            if len(X) < 50:
                raise ValueError(f"Insuficientes secuencias: {len(X)}")

            # 5. Split temporal
            X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_train_test_split(X, y)

            # 6. Class weights mejorados
            classes = np.unique(y_train)
            if len(classes) < 3:
                logger.warning("⚠️ No todas las clases presentes en train set")
                # Crear class weights manual
                class_weight_dict = {0: 1.0, 1: 0.5, 2: 1.0}  # Penalizar menos HOLD
            else:
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                class_weight_dict = {i: w for i, w in enumerate(class_weights)}

            logger.info(f"📊 Class weights: {class_weight_dict}")

            # 7. Crear modelo optimizado
            self.model = self.create_optimized_model((X.shape[1], X.shape[2]))

            # 8. Callbacks mejorados
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    monitor='val_accuracy',
                    min_delta=0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=self.config.reduce_lr_patience,
                    factor=0.5,
                    monitor='val_loss',
                    min_lr=1e-6
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model_temp.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                )
            ]

            logger.info(f"🎯 Entrenando: {self.config.epochs} épocas max, batch {self.config.batch_size}")

            # 9. Entrenar
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )

            # 10. Cargar mejor modelo
            if os.path.exists('best_model_temp.h5'):
                self.model = tf.keras.models.load_model('best_model_temp.h5')
                os.remove('best_model_temp.h5')

            # 11. Evaluación
            y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

            # Precios para métricas
            test_prices = df['close'].iloc[-len(y_test):].values

            metrics = self.calculate_enhanced_metrics(y_test, y_pred, test_prices)

            # 12. Reporte final mejorado
            logger.info("\n" + "="*60)
            logger.info("🎯 RESULTADOS FINALES MEJORADOS")
            logger.info("="*60)
            logger.info(f"💡 Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"💰 Retorno total: {metrics['total_return']:.2f}%")
            logger.info(f"🔄 Trades: {metrics['num_trades']}")
            logger.info(f"🎯 Win rate: {metrics['win_rate']:.1f}%")
            logger.info(f"📊 Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"📉 Max drawdown: {metrics['max_drawdown']:.2f}%")
            logger.info(f"💎 Profit factor: {metrics['profit_factor']:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Error durante entrenamiento: {e}")
            return {"error": str(e)}

    def save_model(self, path: str):
        """Guardar modelo y configuración"""
        os.makedirs(path, exist_ok=True)

        self.model.save(f"{path}/model.h5")

        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        with open(f"{path}/features.json", "w") as f:
            json.dump(self.feature_columns, f, indent=2)

        logger.info(f"✅ Modelo guardado en: {path}")


async def test_sistema_mejorado():
    """Función de prueba del sistema mejorado"""

    print("🚀 PROBANDO SISTEMA MEJORADO")
    print("=" * 50)

    # Configuración optimizada para prueba
    config = TradingConfigMejorado(
        symbol="BTCUSDT",
        timeframe="5m",
        lookback_periods=24,
        training_days=30,  # Menos días para prueba rápida
        epochs=50,         # Menos épocas para prueba
        batch_size=64,
        learning_rate=0.001
    )

    # Crear y entrenar sistema
    system = TradingSystemMejorado(config)

    try:
        results = await system.train()

        if "error" not in results:
            print("\n🎉 ¡SISTEMA MEJORADO FUNCIONA!")
            print("📊 Resultados:")
            for key, value in results.items():
                print(f"   {key}: {value:.3f}")

            # Guardar si es rentable
            if results['total_return'] > 0:
                system.save_model("models/sistema_mejorado")
                print("✅ Modelo rentable guardado")
        else:
            print(f"❌ Error: {results['error']}")

    except Exception as e:
        print(f"❌ Error en prueba: {e}")


if __name__ == "__main__":
    asyncio.run(test_sistema_mejorado())
