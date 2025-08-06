#!/usr/bin/env python3
"""
🎯 TCN ENSEMBLE TRAINER - SISTEMA DE ENSAMBLE AVANZADO
Combina modelos de 1m y 5m para generar predicciones más robustas y estables
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
import pickle
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

from centralized_features_engine2 import CentralizedFeaturesEngine


class TCNEnsembleTrainer:
    """🎯 Sistema de ensamble que combina modelos de diferentes timeframes"""

    def __init__(self):
        self.pairs = ["BNBUSDT","XRPUSDT"]
        self.features_engine = CentralizedFeaturesEngine()

        # Configuración de timeframes
        self.timeframes = {
            '5m': {'lookback_window': 48, 'prediction_horizon': 12},  # 48 min lookback, 12 min ahead
            ##'5m': {'lookback_window': 24, 'prediction_horizon': 6}    # 2h lookback, 30 min ahead
        }

        # THRESHOLDS COMPLETAMENTE CORREGIDOS
        self.thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.0014, 'weak_sell': -0.0007,
                'weak_buy': 0.0007, 'strong_buy': 0.0014
            },
            'ETHUSDT': {
                'strong_sell': -0.0026, 'weak_sell': -0.0012,
                'weak_buy': 0.0013, 'strong_buy': 0.0027
            },
            'BNBUSDT': {
                'strong_sell': -0.0014, 'weak_sell': -0.0007,
                'weak_buy': 0.0007, 'strong_buy': 0.0014
            },
            'XRPUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018
            }
        }

    async def get_market_data(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """📊 Obtener datos de mercado para el timeframe especificado"""

        print(f"📊 Obteniendo {days} días de datos {timeframe} para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
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

        print(f"✅ Obtenidos {len(df)} registros {timeframe} de {symbol}")
        return df

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame,
                             symbol: str, timeframe: str) -> pd.DataFrame:
        """🎯 Crear etiquetas balanceadas específicas por timeframe"""

        print(f"🎯 Creando etiquetas para {symbol} - {timeframe}...")

        close_prices = df['close'].values
        thresholds = self.thresholds[symbol]
        prediction_horizon = self.timeframes[timeframe]['prediction_horizon']

        labels = []

        for i in range(len(close_prices) - prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + prediction_horizon]

            # Calcular retorno futuro
            future_return = (future_price - current_price) / current_price

            # Lógica balanceada con análisis técnico
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris: usar indicadores técnicos
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi > 60 or current_macd < 0:
                    label = 0  # SELL
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris: usar indicadores técnicos
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi < 40 or current_macd > 0:
                    label = 2  # BUY
                else:
                    label = 1  # HOLD
            else:
                # Zona neutral: usar momentum
                if i >= 5:
                    recent_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                    if recent_momentum > 0.01:
                        label = 2  # BUY
                    elif recent_momentum < -0.01:
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD

            labels.append(label)

        # Crear DataFrame con labels
        df_labeled = df.iloc[:-prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print(f"📊 Distribución {timeframe}:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame,
                            timeframe: str) -> tuple:
        """🔧 Preparar datos para entrenamiento por timeframe"""

        print(f"🔧 Preparando datos para {timeframe}...")

        lookback_window = self.timeframes[timeframe]['lookback_window']
        prediction_horizon = self.timeframes[timeframe]['prediction_horizon']

        # Alinear features con labels
        features_aligned = features.iloc[:-prediction_horizon]

        # Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns
                         if features_aligned[col].dtype in ['float64', 'int64']]

        # Normalizar features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # Crear secuencias temporales
        X = []
        y = []

        for i in range(lookback_window, len(features_scaled)):
            sequence = features_scaled[i-lookback_window:i]
            X.append(sequence)
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos {timeframe} preparados:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features: {len(feature_columns)}")

        # Calcular class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_tcn_model(self, input_shape: tuple, timeframe: str):
        """🎯 Crear modelo TCN específico por timeframe"""

        print(f"🎯 Creando modelo TCN para {timeframe}...")

        # Arquitectura ajustada por timeframe
        if timeframe == '1m':
            # Modelo más profundo para capturar patrones de alta frecuencia
            filters = [64, 128, 256, 512, 256, 128]
            dilations = [1, 2, 4, 8, 16, 32]
        else:  # 5m
            # Modelo más amplio para patrones de baja frecuencia
            filters = [96, 192, 384, 192, 96]
            dilations = [1, 2, 4, 8, 16]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),
        ])

        # Bloques TCN
        for i, (f, d) in enumerate(zip(filters, dilations)):
            model.add(tf.keras.layers.Conv1D(
                filters=f, kernel_size=3, dilation_rate=d,
                padding='causal', activation='relu'
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.SpatialDropout1D(0.1 + i * 0.05))

        # Capas finales
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(256, activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(128, activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        # Compilar
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=0.0005 if timeframe == '1m' else 0.0003
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo {timeframe} creado: {model.count_params():,} parámetros")
        return model

    async def train_ensemble_models(self, symbol: str) -> bool:
        """🎯 Entrenar modelos de ensamble para un símbolo"""

        print(f"\n🎯 ENTRENANDO ENSEMBLE PARA {symbol}")
        print("=" * 70)

        results = {}

        for timeframe in ['5m']:
            print(f"\n🔄 Entrenando modelo {timeframe}...")

            try:
                # 1. Obtener datos
                df = await self.get_market_data(symbol, timeframe, days=90)

                # 2. Crear features
                features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
                if features.empty:
                    print(f"❌ Error calculando features {timeframe}")
                    results[timeframe] = False
                    continue

                # 3. Crear etiquetas
                df_labeled = self.create_balanced_labels(df, features, symbol, timeframe)

                # 4. Preparar datos
                X, y, scaler, feature_columns, class_weights = self.prepare_training_data(
                    df_labeled, features, timeframe
                )

                # 5. Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # 6. Crear modelo
                model = self.create_tcn_model((X.shape[1], X.shape[2]), timeframe)

                # 7. Directorio de modelos
                model_dir = f'models/definitivo_v3_5m_{symbol.lower()}'
                os.makedirs(model_dir, exist_ok=True)

                # 8. Callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        patience=15, restore_best_weights=True,
                        monitor='val_accuracy', verbose=1
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        patience=8, factor=0.7, monitor='val_loss', verbose=1
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        f'{model_dir}/best_model.h5',
                        save_best_only=True, monitor='val_accuracy', verbose=1
                    )
                ]

                # 9. Entrenar
                print(f"🚀 Entrenando modelo {timeframe}...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=64 if timeframe == '5m' else 32,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1
                )

                # 10. Evaluar
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                print(f"\n✅ Resultados {timeframe}:")
                print(f"   - Accuracy: {test_acc:.3f}")
                print(f"   - Loss: {test_loss:.3f}")

                # 11. Guardar componentes
                model.save(f'{model_dir}/model.h5')

                with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)

                with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                    pickle.dump(feature_columns, f)

                with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                    pickle.dump(class_weights, f)

                # Guardar métricas del ensemble
                ensemble_metrics = {
                    'timeframe': timeframe,
                    'accuracy': test_acc,
                    'loss': test_loss,
                    'symbol': symbol,
                    'lookback_window': self.timeframes[timeframe]['lookback_window'],
                    'prediction_horizon': self.timeframes[timeframe]['prediction_horizon'],
                    'features_count': len(feature_columns)
                }

                with open(f'{model_dir}/ensemble_metrics.pkl', 'wb') as f:
                    pickle.dump(ensemble_metrics, f)

                results[timeframe] = True
                print(f"✅ Modelo {timeframe} guardado en {model_dir}/")

            except Exception as e:
                print(f"❌ Error entrenando {timeframe}: {e}")
                results[timeframe] = False

        # Resumen final
        print(f"\n🎯 RESUMEN ENSEMBLE {symbol}:")
        print("=" * 50)
        for timeframe, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"   {timeframe}: {status}")

        successful = sum(results.values())
        print(f"\n🏆 Modelos exitosos: {successful}/2")

        return successful == 2


async def main():
    """🎯 Entrenar sistema de ensamble"""

    print("🎯 TCN ENSEMBLE TRAINER")
    print("=" * 80)
    print("🔄 Entrenando modelos 1m y 5m para crear ensamble robusto")
    print("🎯 Objetivo: Predicciones más estables y precisas")
    print("=" * 80)

    trainer = TCNEnsembleTrainer()

    # Entrenar un símbolo primero para prueba
    symbol = "BNBUSDT"
    print(f"\n🚀 Entrenando ensemble para {symbol}...")

    success = await trainer.train_ensemble_models(symbol)

    if success:
        print(f"\n✅ {symbol}: ENSEMBLE COMPLETADO EXITOSAMENTE")
        print(f"🎯 Modelos guardados en models/definitivo_v3_5m_{symbol.lower()}/")
    else:
        print(f"\n❌ {symbol}: ERROR EN ENSEMBLE")

    # Entrenar todos los símbolos
    train_all = input("\n🤔 ¿Entrenar ensemble para todos los símbolos? (y/n): ").lower().strip()

    if train_all == 'y':
        print("\n🚀 Entrenando ensemble completo...")
        results = {}

        for symbol in trainer.pairs:
            if symbol != "BNBUSDT":  # Ya entrenado
                print(f"\n🔄 Entrenando ensemble para {symbol}...")
                success = await trainer.train_ensemble_models(symbol)
                results[symbol] = success

        print(f"\n🎯 RESUMEN FINAL ENSEMBLE:")
        print("=" * 50)
        results["BNBUSDT"] = True  # Agregar BNB
        for symbol, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"   {symbol}: {status}")

        successful = sum(results.values())
        print(f"\n🏆 Ensembles completados: {successful}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
