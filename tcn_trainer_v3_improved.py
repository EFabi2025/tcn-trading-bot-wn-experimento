#!/usr/bin/env python3
"""
🎯 TCN TRAINER V3 MEJORADO - MAYOR ESTABILIDAD Y ACCURACY
Versión mejorada con arquitectura balanceada y mejor regularización
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import talib
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

from centralized_features_engine2 import CentralizedFeaturesEngine


class ImprovedTCNTrainer:
    """🎯 Entrenador TCN V3 MEJORADO para mayor estabilidad y accuracy"""

    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        self.lookback_window = 24
        self.prediction_horizon = 6
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ Configuración CONSERVADORA (para evitar pérdidas)
        self.use_adaptive_thresholds = True

        # 🛡️ THRESHOLDS MÁS CONSERVADORES (para evitar trades arriesgados)
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.003, 'weak_sell': -0.0015,  # MÁS CONSERVADOR
                'weak_buy': 0.004, 'strong_buy': 0.008
            },
            'ETHUSDT': {
                'strong_sell': -0.0012, 'weak_sell': -0.0012,
                'weak_buy': 0.004, 'strong_buy': 0.008
            },
            'BNBUSDT': {
                'strong_sell': -0.0012, 'weak_sell': -0.0012,
                'weak_buy': 0.004, 'strong_buy': 0.008
            },
            'XRPUSDT': {
                'strong_sell': -0.0012, 'weak_sell': -0.0012,
                'weak_buy': 0.004, 'strong_buy': 0.008
            }
        }

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """⚖️ Thresholds adaptativos BALANCEADOS"""
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]

        try:
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)

            # ATR de 14 períodos
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

            # Promedio de ATR reciente
            avg_atr = np.nanmean(atr_14[-50:]) if len(atr_14) > 50 else np.nanmean(atr_14)
            avg_price = np.mean(close_prices[-50:]) if len(close_prices) > 50 else np.mean(close_prices)

            # ATR como porcentaje del precio
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02

            # ⚖️ FACTOR BALANCEADO (ni muy agresivo ni muy conservador)
            base_threshold = atr_percent * 0.5  # Punto medio entre 0.3 y 0.8

            adaptive_thresholds = {
                'strong_sell': -base_threshold * 1.5,  # Balanceado
                'weak_sell': -base_threshold * 0.8,
                'weak_buy': base_threshold * 0.8,
                'strong_buy': base_threshold * 1.5
            }

            print(f"⚖️ {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%) - BALANCEADO")
            print(f"   📊 Thresholds BALANCEADOS: Buy {adaptive_thresholds['strong_buy']:.4f}, Sell {adaptive_thresholds['strong_sell']:.4f}")

            return adaptive_thresholds

        except Exception as e:
            print(f"⚠️ Error calculando thresholds: {e}")
            return self.fixed_thresholds[symbol]

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """⚖️ Etiquetado BALANCEADO - ni agresivo ni conservador"""

        print(f"⚖️ Creando etiquetas BALANCEADAS para {symbol}...")

        close_prices = df['close'].values
        thresholds = self.calculate_adaptive_thresholds(df, symbol)

        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]
            future_return = (future_price - current_price) / current_price

            # ⚖️ LÓGICA BALANCEADA
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris SELL: moderadamente estricta
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    if current_rsi > 65:  # Balanceado (era 70, antes 55)
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                except:
                    label = 0 if future_return < -0.0008 else 1  # Umbral moderado
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris BUY: moderadamente estricta
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    if current_rsi < 35:  # Balanceado (era 30, antes 45)
                        label = 2  # BUY
                    else:
                        label = 1  # HOLD
                except:
                    label = 2 if future_return > 0.0008 else 1  # Umbral moderado
            else:
                # Zona neutral: momentum moderado
                if i >= 4:  # Moderado (era 5, antes 3)
                    recent_momentum = (close_prices[i] - close_prices[i-4]) / close_prices[i-4]
                    if recent_momentum > 0.01:  # Moderado (era 0.015, antes 0.005)
                        label = 2  # BUY
                    elif recent_momentum < -0.01:
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD

            labels.append(label)

        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución balanceada
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución BALANCEADA:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        # Verificar balance óptimo
        hold_count = label_counts.get(1, 0) or 0
        hold_pct = (hold_count / total * 100) if total > 0 else 0

        if hold_pct > 60:
            print("⚠️ Puede ser muy conservador (HOLD > 60%)")
        elif hold_pct < 35:
            print("⚠️ Puede ser muy agresivo (HOLD < 35%)")
        else:
            print(f"✅ Distribución BALANCEADA: {hold_pct:.1f}% HOLD (óptimo: 35-60%)")

        return df_labeled

    async def get_real_market_data(self, symbol: str, days: int =30) -> pd.DataFrame:
        """📊 MÁS DATOS para mejor entrenamiento"""
        print(f"📊 Obteniendo {days} días de datos para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            all_data = []
            current_start = start_time

            while current_start < end_time:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        all_data.extend(data)
                        current_start = data[-1][6] + 1
                    else:
                        print(f"❌ Error API: {response.status}")
                        break

                await asyncio.sleep(0.1)

        # Convertir a DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(all_data, columns=columns)  # type: ignore

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros de {symbol}")
        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparación optimizada"""
        print("🔧 Preparando datos optimizados...")

        features_aligned = features.iloc[:-self.prediction_horizon]
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados: X shape: {X.shape}, y shape: {y.shape}")

        # Class weights menos agresivos
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        # Suavizar weights extremos
        class_weights = np.clip(class_weights, 0.5, 2.0)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_improved_tcn_model(self, input_shape: tuple):
        """🎯 Modelo TCN V3 MEJORADO con mejor regularización"""

        print("🎯 Creando modelo TCN V3 MEJORADO...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),

            # Normalización inicial
            tf.keras.layers.LayerNormalization(),

            # Bloques TCN optimizados - ARQUITECTURA PIRAMIDAL BALANCEADA
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.1),  # Mejor para secuencias

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.15),

            tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.2),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.25),

            tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=16, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.3),

            # Global pooling
            tf.keras.layers.GlobalAveragePooling1D(),

            # Capas densas más conservadoras con regularización L2
            tf.keras.layers.Dense(256, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Optimizador con learning rate más conservador
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=0.0005,  # Reducido de 0.001
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo V3 MEJORADO creado: {model.count_params():,} parámetros")
        return model

    def analyze_training_stability(self, history):
        """📊 Analiza la estabilidad del entrenamiento"""

        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        # Calcular volatilidad de últimas 10 epochs
        if len(val_acc) >= 10:
            recent_acc_std = np.std(val_acc[-10:])
            recent_loss_std = np.std(val_loss[-10:])

            print(f"📊 Estabilidad últimas 10 epochs:")
            print(f"   Val_accuracy std: {recent_acc_std:.4f}")
            print(f"   Val_loss std: {recent_loss_std:.4f}")

            # Criterios de estabilidad
            stable_acc = recent_acc_std < 0.02  # 2% variación
            stable_loss = recent_loss_std < 0.1

            if stable_acc and stable_loss:
                print("✅ Entrenamiento estable")
            else:
                print("⚠️ Entrenamiento inestable - considerar ajustes")

            return stable_acc and stable_loss

        return True

    async def train_improved_model(self, symbol: str) -> bool:
        """🎯 Entrenamiento V3 mejorado con mejores callbacks"""

        print(f"\n🎯 ENTRENANDO MODELO V3 MEJORADO PARA {symbol}")
        print("=" * 70)

        try:
            # 1. MÁS DATOS
            df = await self.get_real_market_data(symbol, days=60)

            # 2. Calcular features
            print(f"🔄 Calculando features...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            # 3. Etiquetas optimizadas
            df_labeled = self.create_balanced_labels(df, features, symbol)

            # 4. Preparar datos
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # 5. Split optimizado con validación
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Modelo mejorado
            model = self.create_improved_tcn_model((X.shape[1], X.shape[2]))

            # Directorio V3
            model_dir = f'models/adaptive_v3_improved_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)

            # Callbacks mejorados
            callbacks = [
                # Early stopping más agresivo
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,  # Reducido de 20
                    restore_best_weights=True,
                    verbose=1
                ),

                # ReduceLROnPlateau más sensible
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,  # Reducción más agresiva
                    patience=8,   # Más rápido
                    min_lr=1e-6,
                    verbose=1
                ),

                # Model checkpoint
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f'{model_dir}/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),

                # Opcional: TerminateOnNaN
                tf.keras.callbacks.TerminateOnNaN()
            ]

            print("🚀 Entrenamiento V3 MEJORADO...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,  # Reducido de 150
                batch_size=64,  # Aumentado para estabilidad
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                shuffle=True
            )

            # 7. Análisis de estabilidad
            is_stable = self.analyze_training_stability(history)

            # 8. Evaluación
            val_accuracy = max(history.history['val_accuracy'])
            print(f"🎯 Mejor val_accuracy alcanzado: {val_accuracy:.4f}")

            # Criterio de aceptación más estricto
            if val_accuracy < 0.65:  # Umbral mínimo
                print("⚠️ Accuracy insuficiente, considerar reentrenamiento")
                return False

            if not is_stable:
                print("⚠️ Entrenamiento inestable, pero accuracy aceptable")

            # 9. Guardar todo
            model.save(f'{model_dir}/model.h5')

            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)

            # Guardar historial para análisis
            with open(f'{model_dir}/training_history.pkl', 'wb') as f:
                pickle.dump(history.history, f)

            # Guardar métricas de estabilidad
            stability_metrics = {
                'final_val_accuracy': val_accuracy,
                'is_stable': is_stable,
                'training_epochs': len(history.history['val_accuracy']),
                'best_epoch': np.argmax(history.history['val_accuracy']) + 1
            }

            with open(f'{model_dir}/stability_metrics.pkl', 'wb') as f:
                pickle.dump(stability_metrics, f)

            print(f"✅ Modelo V3 MEJORADO guardado en {model_dir}/")
            print(f"✅ Accuracy final: {val_accuracy:.3f}")
            print(f"✅ Estabilidad: {'SÍ' if is_stable else 'NO'}")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

async def main():
    """🎯 Entrenar modelo V3 mejorado"""

    print("🎯 ENTRENADOR TCN V3 MEJORADO")
    print("=" * 70)

    trainer = ImprovedTCNTrainer()

    # Solo BTC primero para probar
    symbol = "BTCUSDT"
    print(f"\n🚀 Entrenando {symbol} con arquitectura V3 mejorada...")

    success = await trainer.train_improved_model(symbol)

    if success:
        print(f"\n✅ {symbol}: ENTRENAMIENTO V3 EXITOSO")
    else:
        print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO V3")

if __name__ == "__main__":
    asyncio.run(main())
