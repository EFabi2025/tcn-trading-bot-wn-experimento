#!/usr/bin/env python3
"""
🎯 TCN TRAINER V2 OPTIMIZADO - MAYOR ACCURACY
Versión mejorada con parámetros optimizados para mejor rendimiento
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


class OptimizedTCNTrainer:
    """🎯 Entrenador TCN V2 OPTIMIZADO para mayor accuracy"""

    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        self.lookback_window = 24
        self.prediction_horizon = 6
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ Configuración optimizada
        self.use_adaptive_thresholds = True
        
        # 🎯 THRESHOLDS MÁS AGRESIVOS (menos conservadores)
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.003, 'weak_sell': -0.0015,  # Menos restrictivo
                'weak_buy': 0.0015, 'strong_buy': 0.003
            },
            'ETHUSDT': {
                'strong_sell': -0.002, 'weak_sell': -0.001,
                'weak_buy': 0.001, 'strong_buy': 0.002
            },
            'BNBUSDT': {
                'strong_sell': -0.0012, 'weak_sell': -0.0006,
                'weak_buy': 0.0006, 'strong_buy': 0.0012
            },
            'XRPUSDT': {
                'strong_sell': -0.0015, 'weak_sell': -0.0008,
                'weak_buy': 0.0008, 'strong_buy': 0.0015
            }
        }

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """🎯 Thresholds adaptativos MENOS CONSERVADORES"""
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
            
            # 🚀 FACTOR MENOS CONSERVADOR (era 0.5, ahora 0.8)
            base_threshold = atr_percent * 0.8
            
            adaptive_thresholds = {
                'strong_sell': -base_threshold * 1.2,  # Menos restrictivo
                'weak_sell': -base_threshold * 0.6,
                'weak_buy': base_threshold * 0.6,
                'strong_buy': base_threshold * 1.2
            }
            
            print(f"🎯 {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%)")
            print(f"   📊 Thresholds OPTIMIZADOS: Buy {adaptive_thresholds['strong_buy']:.4f}, Sell {adaptive_thresholds['strong_sell']:.4f}")
            
            return adaptive_thresholds
            
        except Exception as e:
            print(f"⚠️ Error calculando thresholds: {e}")
            return self.fixed_thresholds[symbol]

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Etiquetado más balanceado y menos restrictivo"""

        print(f"🎯 Creando etiquetas OPTIMIZADAS para {symbol}...")

        close_prices = df['close'].values
        thresholds = self.calculate_adaptive_thresholds(df, symbol)
        
        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]
            future_return = (future_price - current_price) / current_price

            # 🚀 LÓGICA MÁS AGRESIVA Y BALANCEADA
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris SELL: más permisiva
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    if current_rsi > 55:  # Era 60, ahora 55
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                except:
                    label = 0 if future_return < -0.0005 else 1  # Umbral mínimo
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris BUY: más permisiva
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    if current_rsi < 45:  # Era 40, ahora 45
                        label = 2  # BUY
                    else:
                        label = 1  # HOLD
                except:
                    label = 2 if future_return > 0.0005 else 1  # Umbral mínimo
            else:
                # Zona neutral: momentum más sensible
                if i >= 3:  # Era 5, ahora 3
                    recent_momentum = (close_prices[i] - close_prices[i-3]) / close_prices[i-3]
                    if recent_momentum > 0.005:  # Era 0.01, ahora 0.005
                        label = 2  # BUY
                    elif recent_momentum < -0.005:
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD

            labels.append(label)

        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar y mejorar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución OPTIMIZADA:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    async def get_real_market_data(self, symbol: str, days: int = 15) -> pd.DataFrame:
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
        df = pd.DataFrame(all_data, columns=columns)

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

    def create_optimized_tcn_model(self, input_shape: tuple):
        """🎯 Modelo TCN OPTIMIZADO para mayor accuracy"""
        
        print("🎯 Creando modelo TCN OPTIMIZADO...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Normalización inicial
            tf.keras.layers.LayerNormalization(),
            
            # Bloques TCN optimizados
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),  # Menos dropout
            
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(filters=512, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=16, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Global pooling y capas densas optimizadas
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Optimizador y learning rate optimizados
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),  # LR mayor
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo OPTIMIZADO creado: {model.count_params():,} parámetros")
        return model

    async def train_optimized_model(self, symbol: str) -> bool:
        """🎯 Entrenamiento optimizado"""

        print(f"\n🎯 ENTRENANDO MODELO OPTIMIZADO PARA {symbol}")
        print("=" * 70)

        try:
            # 1. MÁS DATOS
            df = await self.get_real_market_data(symbol, days=15)  # Era 10, ahora 15

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

            # 5. Split optimizado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y  # Test más pequeño
            )

            # 6. Modelo optimizado
            model = self.create_optimized_tcn_model((X.shape[1], X.shape[2]))

            # Directorio optimizado
            model_dir = f'models/adaptive_v2_optimized_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)

            # Callbacks optimizados
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
                tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.7, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{model_dir}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            print("🚀 Entrenamiento OPTIMIZADO...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=150,  # Más epochs
                batch_size=64,  # Batch size mayor
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            # 7. Evaluación
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Accuracy OPTIMIZADA: {test_acc:.3f}")

            # 8. Guardar
            model.save(f'{model_dir}/model.h5')
            
            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                
            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)

            print(f"✅ Modelo OPTIMIZADO guardado en {model_dir}/")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

async def main():
    """🎯 Entrenar solo BTC optimizado"""
    
    print("🎯 ENTRENADOR TCN V2 OPTIMIZADO")
    print("=" * 70)
    
    trainer = OptimizedTCNTrainer()
    
    # Solo BTC primero para probar
    symbol = "BTCUSDT"
    print(f"\n🚀 Entrenando {symbol} con parámetros optimizados...")
    
    success = await trainer.train_optimized_model(symbol)
    
    if success:
        print(f"\n✅ {symbol}: ENTRENAMIENTO OPTIMIZADO EXITOSO")
    else:
        print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO")

if __name__ == "__main__":
    asyncio.run(main()) 