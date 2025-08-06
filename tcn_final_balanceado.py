#!/usr/bin/env python3
"""
🎯 SISTEMA FINAL BALANCEADO - LA SOLUCIÓN DEFINITIVA
Combina aprendizaje efectivo con trading selectivo y rentable
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from centralized_features_engine2 import CentralizedFeaturesEngine

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfigFinal:
    """Configuración final balanceada para trading rentable"""

    # === CONFIGURACIÓN BÁSICA ===
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    lookback_periods: int = 36  # Balanceado: más contexto histórico
    prediction_horizon: int = 8  # Horizonte más largo para mayor confianza

    # === DATOS ===
    training_days: Optional[int] = 45  # Balanceado: suficientes datos, no demasiado viejos
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # === MODELO ===
    model_type: str = "tcn_balanced"
    feature_set: str = "tcn_definitivo"
    scaler_type: str = "robust"

    # === ENTRENAMIENTO ===
    test_size: float = 0.15
    validation_size: float = 0.10
    epochs: int = 100
    batch_size: int = 32  # Reducido para mejor generalización
    learning_rate: float = 0.0008  # Balanceado
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 10

    # === UMBRALES SELECTIVOS ===
    min_profitable_move: float = 0.008  # 0.8% mínimo para justificar trade
    signal_confidence_threshold: float = 0.75  # Alta confianza requerida
    volatility_max: float = 0.03  # No tradear en alta volatilidad

    # === COSTOS ===
    commission_rate: float = 0.001
    spread_cost: float = 0.0003
    slippage_cost: float = 0.0002

    @property
    def total_trading_cost(self) -> float:
        return self.commission_rate + self.spread_cost + self.slippage_cost


class TradingSystemFinal:
    """Sistema de trading final balanceado"""

    def __init__(self, config: TradingConfigFinal):
        self.config = config
        self.scaler = RobustScaler()
        self.model = None
        self.feature_columns = []
        self.features_engine = CentralizedFeaturesEngine()

        logger.info(f"🎯 Sistema Final: {config.symbol} - {config.timeframe}")
        logger.info(f"📊 Lookback: {config.lookback_periods} | Min Move: {config.min_profitable_move:.3f}")

    async def fetch_market_data(self) -> pd.DataFrame:
        """Obtener datos de mercado"""

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

        # Limpiar datos
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        # Remover outliers suavemente
        for col in numeric_cols:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)

        df = df.dropna()

        logger.info(f"✅ Datos procesados: {len(df)} registros")

        return df

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features técnicos balanceados"""

        logger.info(f"🔧 Calculando features balanceados: {self.config.feature_set}")

        try:
            # Usar motor centralizado
            features = self.features_engine.calculate_features(df, self.config.feature_set)

            # Agregar features críticos balanceados
            features = self._add_balanced_features(df, features)

            # Limpiar features
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)

            # Filtrar features con varianza muy baja
            numeric_features = features.select_dtypes(include=[np.number])
            variance_filter = numeric_features.var() > 1e-6
            features = numeric_features.loc[:, variance_filter]

            self.feature_columns = list(features.columns)
            logger.info(f"✅ Features balanceadas: {len(self.feature_columns)}")

            return features

        except Exception as e:
            logger.error(f"Error en features: {e}")
            return self._calculate_fallback_features(df)

    def _add_balanced_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Agregar features balanceados críticos para trading selectivo"""

        try:
            # Momentum multi-timeframe
            features['momentum_short'] = df['close'].pct_change(3)
            features['momentum_medium'] = df['close'].pct_change(8)
            features['momentum_long'] = df['close'].pct_change(20)

            # Volatility normalizada
            returns = df['close'].pct_change()
            features['volatility_5'] = returns.rolling(5).std()
            features['volatility_10'] = returns.rolling(10).std()
            features['volatility_20'] = returns.rolling(20).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20'].clip(lower=1e-8)

            # Trend consistency
            ma_5 = df['close'].rolling(5).mean()
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()

            features['trend_alignment'] = ((ma_5 > ma_20) & (ma_20 > ma_50)).astype(int) - \
                                        ((ma_5 < ma_20) & (ma_20 < ma_50)).astype(int)

            # Support/Resistance
            features['price_percentile_20'] = df['close'].rolling(20).rank(pct=True)
            features['price_percentile_50'] = df['close'].rolling(50).rank(pct=True)

            # Volume strength
            features['volume_ma_20'] = df['volume'].rolling(20).mean()
            features['volume_strength'] = df['volume'] / features['volume_ma_20'].clip(lower=1)
            features['volume_surge'] = (features['volume_strength'] > 1.5).astype(int)

            # Market regime detection
            features['high_volatility_regime'] = (features['volatility_20'] > features['volatility_20'].rolling(50).quantile(0.7)).astype(int)

            return features

        except Exception as e:
            logger.error(f"Error agregando features balanceados: {e}")
            return features

    def _calculate_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features fallback balanceados"""

        features = pd.DataFrame(index=df.index)

        try:
            # Features esenciales
            features['returns_5'] = df['close'].pct_change(5)
            features['returns_10'] = df['close'].pct_change(10)
            features['sma_20'] = df['close'].rolling(20).mean()
            features['sma_50'] = df['close'].rolling(50).mean()

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

            # Volatility
            features['volatility'] = df['close'].pct_change().rolling(20).std()

            # Volume
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            features = features.fillna(method='ffill').fillna(0)
            self.feature_columns = list(features.columns)

            return features

        except Exception as e:
            logger.error(f"Error en fallback: {e}")
            features['returns_1'] = df['close'].pct_change(1).fillna(0)
            self.feature_columns = ['returns_1']
            return features

    def create_selective_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Crear etiquetas selectivas para trading de alta calidad"""

        logger.info("🎯 Creando etiquetas selectivas de alta calidad...")

        # Calcular returns futuros
        future_returns = df['close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)

        labels = []

        for i in range(len(df)):
            if i < 50 or i >= len(df) - self.config.prediction_horizon:
                labels.append(1)  # HOLD
                continue

            future_return = future_returns.iloc[i]

            if pd.isna(future_return):
                labels.append(1)
                continue

            try:
                # Condiciones del mercado para trading selectivo

                # Volatility check - no tradear en alta volatilidad
                vol_cols = [col for col in features.columns if 'volatility' in col.lower()]
                if vol_cols:
                    current_vol = features[vol_cols[0]].iloc[i]
                    if current_vol > self.config.volatility_max:
                        labels.append(1)  # HOLD en alta volatilidad
                        continue

                # RSI para evitar extremos
                rsi_cols = [col for col in features.columns if 'rsi' in col.lower()]
                if rsi_cols:
                    rsi_val = features[rsi_cols[0]].iloc[i]
                    if pd.isna(rsi_val) or rsi_val > 80 or rsi_val < 20:
                        labels.append(1)  # Evitar extremos RSI
                        continue

                # Trend alignment check
                trend_cols = [col for col in features.columns if 'trend' in col.lower()]
                momentum_cols = [col for col in features.columns if 'momentum' in col.lower()]

                trend_aligned = True
                if trend_cols:
                    trend_val = features[trend_cols[0]].iloc[i]
                    if abs(trend_val) < 0.1:  # Trend muy débil
                        trend_aligned = False

                # Volume confirmation
                volume_cols = [col for col in features.columns if 'volume' in col.lower()]
                volume_confirmed = False
                if volume_cols:
                    vol_strength = features[volume_cols[0]].iloc[i]
                    if vol_strength > 1.2:  # Volume above average
                        volume_confirmed = True

                # Decisión selectiva basada en calidad de señal
                min_move = self.config.min_profitable_move

                # BUY conditions - muy selectivas
                if (future_return > min_move and
                    trend_aligned and
                    volume_confirmed and
                    future_return > min_move * 1.5):  # Requerir mayor upside

                    # Verificar condiciones adicionales
                    momentum_positive = False
                    if momentum_cols:
                        momentum_val = features[momentum_cols[0]].iloc[i]
                        if momentum_val > 0.001:
                            momentum_positive = True

                    if momentum_positive or not momentum_cols:
                        labels.append(2)  # BUY
                        continue

                # SELL conditions - muy selectivas
                elif (future_return < -min_move and
                      volume_confirmed and
                      future_return < -min_move * 1.5):  # Requerir mayor downside

                    # Verificar condiciones adicionales
                    momentum_negative = False
                    if momentum_cols:
                        momentum_val = features[momentum_cols[0]].iloc[i]
                        if momentum_val < -0.001:
                            momentum_negative = True

                    if momentum_negative or not momentum_cols:
                        labels.append(0)  # SELL
                        continue

                # Default to HOLD si no hay señal clara
                labels.append(1)

            except Exception as e:
                logger.debug(f"Error procesando etiqueta {i}: {e}")
                labels.append(1)

        # Crear DataFrame con labels
        df_labeled = df.copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        logger.info("📊 Distribución selectiva:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            logger.info(f"   {name}: {count} ({pct:.1f}%)")

        # Verificar que tenemos señales pero no demasiadas
        non_hold = label_counts.get(0, 0) + label_counts.get(2, 0)
        signal_rate = non_hold / total

        if signal_rate < 0.05:  # Menos del 5%
            logger.warning("⚠️ Muy pocas señales, puede ser demasiado conservador")
        elif signal_rate > 0.4:  # Más del 40%
            logger.warning("⚠️ Demasiadas señales, puede ser demasiado agresivo")
        else:
            logger.info(f"✅ Tasa de señales balanceada: {signal_rate:.1%}")

        return df_labeled

    def prepare_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar secuencias balanceadas"""

        logger.info(f"🔄 Preparando secuencias balanceadas - Lookback: {self.config.lookback_periods}")

        # Seleccionar y limpiar features
        numeric_features = features.select_dtypes(include=[np.number])

        # Filtrar features con varianza suficiente
        variance_threshold = 1e-7
        var_filter = numeric_features.var() > variance_threshold
        numeric_features = numeric_features.loc[:, var_filter]

        if numeric_features.empty:
            raise ValueError("No hay features válidas después del filtrado")

        self.feature_columns = list(numeric_features.columns)
        logger.info(f"   Features utilizadas: {len(self.feature_columns)}")

        # Remover outliers extremos
        for col in numeric_features.columns:
            q1 = numeric_features[col].quantile(0.01)
            q99 = numeric_features[col].quantile(0.99)
            numeric_features[col] = numeric_features[col].clip(lower=q1, upper=q99)

        # Escalar features
        numeric_features = numeric_features.fillna(method='ffill').fillna(0)
        features_scaled = self.scaler.fit_transform(numeric_features)

        # Crear secuencias
        X, y = [], []

        for i in range(self.config.lookback_periods, len(features_scaled)):
            sequence = features_scaled[i-self.config.lookback_periods:i]
            X.append(sequence)
            y.append(labels.iloc[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"   ✅ Secuencias balanceadas: X={X.shape}, y={y.shape}")

        return X, y

    def create_balanced_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Crear modelo balanceado con arquitectura optimizada"""

        logger.info(f"🧠 Creando modelo balanceado: {input_shape}")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),

            # Normalización robusta
            tf.keras.layers.LayerNormalization(),

            # Extracción de patrones locales
            tf.keras.layers.Conv1D(32, 3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            # Patrones de medio plazo con dilatación
            tf.keras.layers.Conv1D(64, 3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),

            # Patrones de largo plazo
            tf.keras.layers.Conv1D(128, 3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Integración de información
            tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Pooling global para agregación
            tf.keras.layers.GlobalAveragePooling1D(),

            # Clasificación con regularización balanceada
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            # Salida suave
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Compilar con configuración balanceada
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(  # Usar legacy para MacOS
                learning_rate=self.config.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        total_params = model.count_params()
        logger.info(f"   ✅ Modelo balanceado: {total_params:,} parámetros")

        return model

    def temporal_train_test_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split temporal balanceado"""

        total_size = len(X)
        test_size = self.config.test_size
        validation_size = self.config.validation_size
        train_size = 1 - test_size - validation_size

        train_idx = int(total_size * train_size)
        val_idx = int(total_size * (train_size + validation_size))

        X_train = X[:train_idx]
        X_val = X[train_idx:val_idx]
        X_test = X[val_idx:]

        y_train = y[:train_idx]
        y_val = y[train_idx:val_idx]
        y_test = y[val_idx:]

        logger.info(f"📊 Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def calculate_realistic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray) -> Dict:
        """Calcular métricas realistas con costos de trading"""

        logger.info("📊 Calculando métricas realistas...")

        accuracy = np.mean(y_true == y_pred)

        # Simulación de trading realista
        portfolio_value = 1.0
        trades = []
        position = 0  # 0=cash, 1=long, -1=short
        entry_price = 0
        total_fees = 0

        for i in range(len(y_pred)):
            signal = y_pred[i]
            price = prices[i]

            # Solo ejecutar trades con alta confianza
            if signal == 2 and position != 1:  # BUY
                if position == -1:  # Close short
                    pnl = (entry_price - price) / entry_price
                    fees = self.config.total_trading_cost
                    portfolio_value *= (1 + pnl - fees)
                    total_fees += fees
                    trades.append(('CLOSE_SHORT', price, portfolio_value))

                # Enter long
                entry_price = price
                fees = self.config.total_trading_cost
                portfolio_value *= (1 - fees)
                total_fees += fees
                position = 1
                trades.append(('BUY', price, portfolio_value))

            elif signal == 0 and position != -1:  # SELL
                if position == 1:  # Close long
                    pnl = (price - entry_price) / entry_price
                    fees = self.config.total_trading_cost
                    portfolio_value *= (1 + pnl - fees)
                    total_fees += fees
                    trades.append(('CLOSE_LONG', price, portfolio_value))

                # Enter short
                entry_price = price
                fees = self.config.total_trading_cost
                portfolio_value *= (1 - fees)
                total_fees += fees
                position = -1
                trades.append(('SELL', price, portfolio_value))

        # Cerrar posición final
        if position != 0:
            final_price = prices[-1]
            if position == 1:
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price

            fees = self.config.total_trading_cost
            portfolio_value *= (1 + pnl - fees)
            total_fees += fees
            trades.append(('CLOSE_FINAL', final_price, portfolio_value))

        # Calcular métricas
        total_return = (portfolio_value - 1.0) * 100
        num_trades = len([t for t in trades if t[0] in ['BUY', 'SELL']])

        # Win rate
        winning_trades = 0
        losing_trades = 0

        trade_returns = []
        for i in range(1, len(trades)):
            trade_return = trades[i][2] / trades[i-1][2] - 1
            trade_returns.append(trade_return)
            if trade_return > 0:
                winning_trades += 1
            else:
                losing_trades += 1

        win_rate = (winning_trades / max(1, len(trade_returns))) * 100 if trade_returns else 0

        # Sharpe ratio
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        portfolio_values = [t[2] for t in trades]
        if portfolio_values:
            peak = portfolio_values[0]
            max_drawdown = 0

            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0

        metrics = {
            'accuracy': accuracy,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'final_portfolio': portfolio_value,
            'total_fees': total_fees * 100,
            'profit_factor': winning_trades / max(1, losing_trades)
        }

        logger.info("📈 Métricas realistas:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")

        return metrics

    async def train(self) -> Dict:
        """Entrenamiento balanceado del sistema final"""

        logger.info("🎯 Iniciando entrenamiento balanceado...")

        try:
            # 1. Obtener datos
            df = await self.fetch_market_data()
            if len(df) < 200:
                raise ValueError(f"Datos insuficientes: {len(df)} registros")

            # 2. Calcular features balanceadas
            features = self.calculate_technical_features(df)

            # 3. Crear etiquetas selectivas
            df_labeled = self.create_selective_labels(df, features)

            # 4. Preparar secuencias
            X, y = self.prepare_sequences(features, df_labeled['label'])

            if len(X) < 100:
                raise ValueError(f"Secuencias insuficientes: {len(X)}")

            # 5. Split temporal
            X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_train_test_split(X, y)

            # 6. Class weights balanceados
            unique_classes = np.unique(y_train)
            if len(unique_classes) >= 2:
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
                class_weight_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}

                # Ajustar pesos para favorecer señales de trading
                if 1 in class_weight_dict:  # HOLD
                    class_weight_dict[1] *= 0.7  # Reducir peso de HOLD
            else:
                class_weight_dict = {0: 1.0, 1: 0.5, 2: 1.0}

            logger.info(f"📊 Class weights balanceados: {class_weight_dict}")

            # 7. Crear modelo balanceado
            self.model = self.create_balanced_model((X.shape[1], X.shape[2]))

            # 8. Callbacks balanceados
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    monitor='val_accuracy',
                    min_delta=0.002
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=self.config.reduce_lr_patience,
                    factor=0.6,
                    monitor='val_loss',
                    min_lr=1e-6
                )
            ]

            logger.info(f"🎯 Entrenando modelo balanceado...")

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

            # 10. Evaluación
            y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

            # Obtener precios para métricas
            test_prices = df['close'].iloc[-len(y_test):].values

            metrics = self.calculate_realistic_metrics(y_test, y_pred, test_prices)

            # 11. Reporte final
            logger.info("\n" + "="*60)
            logger.info("🎯 SISTEMA FINAL BALANCEADO - RESULTADOS")
            logger.info("="*60)
            logger.info(f"💡 Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"💰 Retorno total: {metrics['total_return']:.2f}%")
            logger.info(f"🔄 Número de trades: {metrics['num_trades']}")
            logger.info(f"🎯 Win rate: {metrics['win_rate']:.1f}%")
            logger.info(f"📊 Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"📉 Max drawdown: {metrics['max_drawdown']:.2f}%")
            logger.info(f"💎 Profit factor: {metrics['profit_factor']:.2f}")
            logger.info(f"💸 Total fees: {metrics['total_fees']:.2f}%")

            return metrics

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento balanceado: {e}")
            return {"error": str(e)}

    def save_model(self, path: str):
        """Guardar modelo balanceado"""
        os.makedirs(path, exist_ok=True)

        self.model.save(f"{path}/model.h5")

        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        with open(f"{path}/features.json", "w") as f:
            json.dump(self.feature_columns, f, indent=2)

        logger.info(f"✅ Modelo balanceado guardado en: {path}")


async def demo_sistema_final():
    """Demostración del sistema final balanceado"""

    print("🎯 SISTEMA FINAL BALANCEADO")
    print("=" * 50)
    print("🔧 Características:")
    print("   ✅ Trading selectivo de alta calidad")
    print("   ✅ Evita overtrading")
    print("   ✅ Costos realistas incluidos")
    print("   ✅ Arquitectura balanceada")
    print("   ✅ Umbrales optimizados")
    print()

    # Configuración balanceada
    config = TradingConfigFinal(
        symbol="BTCUSDT",
        timeframe="5m",
        lookback_periods=36,
        training_days=45,
        epochs=80,
        batch_size=32,
        learning_rate=0.0008,
        min_profitable_move=0.008  # 0.8% mínimo
    )

    # Crear y entrenar
    system = TradingSystemFinal(config)

    try:
        results = await system.train()

        if "error" not in results:
            print("\n🎉 ¡SISTEMA FINAL COMPLETADO!")
            print("=" * 40)

            # Evaluar calidad del sistema
            is_profitable = results['total_return'] > 0
            is_reasonable_trades = 5 <= results['num_trades'] <= 50
            is_good_winrate = results['win_rate'] > 40
            is_good_sharpe = results['sharpe_ratio'] > 0.5

            quality_score = sum([is_profitable, is_reasonable_trades, is_good_winrate, is_good_sharpe])

            print(f"📊 Calidad del sistema: {quality_score}/4")
            print(f"   {'✅' if is_profitable else '❌'} Rentable: {results['total_return']:.2f}%")
            print(f"   {'✅' if is_reasonable_trades else '❌'} Trades razonables: {results['num_trades']}")
            print(f"   {'✅' if is_good_winrate else '❌'} Win rate: {results['win_rate']:.1f}%")
            print(f"   {'✅' if is_good_sharpe else '❌'} Sharpe ratio: {results['sharpe_ratio']:.3f}")

            if quality_score >= 3:
                system.save_model("models/sistema_final_balanceado")
                print("\n🏆 ¡SISTEMA DE ALTA CALIDAD GUARDADO!")
            elif quality_score >= 2:
                print("\n✅ Sistema funcional con mejoras posibles")
            else:
                print("\n⚠️ Sistema necesita más optimización")

        else:
            print(f"❌ Error: {results['error']}")

    except Exception as e:
        print(f"❌ Error en demo: {e}")


if __name__ == "__main__":
    asyncio.run(demo_sistema_final())
