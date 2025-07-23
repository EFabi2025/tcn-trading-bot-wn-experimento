#!/usr/bin/env python3
"""
🎯 TCN ADAPTIVE TRAINER - VERSIÓN DE BAJO IMPACTO
Entrenador con thresholds adaptativos sin cambiar el resto del sistema
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
from collections import Counter
warnings.filterwarnings('ignore')

# Importar motor de features actual (sin cambios)
from centralized_features_engine2 import CentralizedFeaturesEngine


class AdaptiveTCNTrainer:
    """🎯 Entrenador TCN con thresholds adaptativos - CAMBIOS MÍNIMOS"""

    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        self.lookback_window = 24
        self.prediction_horizon = 6
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ NUEVO: Thresholds adaptativos habilitados por configuración
        self.use_adaptive_thresholds = True  # Cambiar a False para volver al modo original

        # 🎯 THRESHOLDS FIJOS (Mantener para compatibilidad)
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.004, 'weak_sell': -0.002,
                'weak_buy': 0.002, 'strong_buy': 0.004
            },
            'ETHUSDT': {
                'strong_sell': -0.0026, 'weak_sell': -0.0012,
                'weak_buy': 0.0013, 'strong_buy': 0.0027
            },
            'BNBUSDT': {
                'strong_sell': -0.0015, 'weak_sell': -0.0007,
                'weak_buy': 0.0007, 'strong_buy': 0.0015
            },
            'XRPUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
            'SOLUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
            'DOGEUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
            'ADAUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
            'DOTUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
        }

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        🎯 Calcular thresholds adaptativos basados en volatilidad ATR

        CAMBIO MÍNIMO: Solo esta función es nueva
        """
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]

        try:
            # Calcular ATR para volatilidad adaptativa
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)

            # ATR de 14 períodos
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

            # Promedio de ATR reciente (últimas 50 velas)
            avg_atr = np.nanmean(atr_14[-50:]) if len(atr_14) > 50 else np.nanmean(atr_14)
            avg_price = np.mean(close_prices[-50:]) if len(close_prices) > 50 else np.mean(close_prices)

            # ATR como porcentaje del precio
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02

            # Thresholds adaptativos basados en ATR
            base_threshold = atr_percent * 0.5  # Factor conservador

            adaptive_thresholds = {
                'strong_sell': -base_threshold * 1.5,
                'weak_sell': -base_threshold * 0.75,
                'weak_buy': base_threshold * 0.75,
                'strong_buy': base_threshold * 1.5
            }

            print(f"🎯 {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%)")
            print(f"   📊 Thresholds: Buy {adaptive_thresholds['strong_buy']:.4f}, Sell {adaptive_thresholds['strong_sell']:.4f}")

            return adaptive_thresholds

        except Exception as e:
            print(f"⚠️ Error calculando thresholds adaptativos para {symbol}: {e}")
            print(f"   🔄 Usando thresholds fijos como fallback")
            return self.fixed_thresholds[symbol]

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas con thresholds adaptativos - MODIFICACIÓN MÍNIMA"""

        print(f"🎯 Creando etiquetas {'adaptativas' if self.use_adaptive_thresholds else 'fijas'} para {symbol}...")

        close_prices = df['close'].values

        # ✅ CAMBIO PRINCIPAL: Usar thresholds adaptativos
        thresholds = self.calculate_adaptive_thresholds(df, symbol)

        labels = []

        # 🔄 RESTO DE LA LÓGICA: Sin cambios (mantener compatibilidad)
        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Calcular retorno futuro
            future_return = (future_price - current_price) / current_price

            # 🎯 LÓGICA BALANCEADA (IGUAL QUE ANTES)
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris: usar indicadores técnicos para decidir
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi > 60 or current_macd < 0:
                    label = 0  # SELL (confirmación técnica)
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris: usar indicadores técnicos para decidir
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi < 40 or current_macd > 0:
                    label = 2  # BUY (confirmación técnica)
                else:
                    label = 1  # HOLD
            else:
                # Zona neutral: usar momentum para decidir
                if i >= 5:
                    recent_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                    if recent_momentum > 0.01:
                        label = 2  # BUY (momentum positivo)
                    elif recent_momentum < -0.01:
                        label = 0  # SELL (momentum negativo)
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD

            labels.append(label)

        # Agregar labels al DataFrame
        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 and count is not None else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    # ✅ RESTO DE MÉTODOS: Copiados exactamente del trainer original
    async def get_real_market_data(self, symbol: str, days: int =10) -> pd.DataFrame:
        """📊 Obtener datos reales de mercado - SIN CAMBIOS"""
        print(f"📊 Obteniendo {days} días de datos reales para {symbol}...")

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
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convertir tipos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros de {symbol}")
        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento - SIN CAMBIOS"""
        print("🔧 Preparando datos para entrenamiento...")

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

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_definitive_tcn_model(self, input_shape: tuple) -> tf.keras.Model:
        """🎯 Crear modelo TCN - SIN CAMBIOS"""
        print("🎯 Creando modelo TCN adaptativo...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo adaptativo creado: {model.count_params():,} parámetros")
        return model

    async def train_adaptive_model(self, symbol: str) -> bool:
        """🎯 Entrenar modelo con thresholds adaptativos"""

        print(f"\n🎯 ENTRENANDO MODELO ADAPTATIVO PARA {symbol}")
        print("=" * 70)

        try:
            # 1. Obtener datos
            df = await self.get_real_market_data(symbol, days=10)

            # 2. Calcular features
            print(f"🔄 Calculando features...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            # 3. Crear etiquetas con thresholds adaptativos
            df_labeled = self.create_balanced_labels(df, features, symbol)

            # 4. Preparar datos
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # 5. Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear y entrenar modelo
            model = self.create_definitive_tcn_model((X.shape[1], X.shape[2]))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/adaptive_{symbol.lower()}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            print("🚀 Entrenando modelo adaptativo...")
            os.makedirs(f'models/adaptive_{symbol.lower()}', exist_ok=True)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            # 7. Evaluar
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Accuracy: {test_acc:.3f}")

            # 8. Guardar componentes
            model.save(f'models/adaptive_{symbol.lower()}/model.h5')

            with open(f'models/adaptive_{symbol.lower()}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            with open(f'models/adaptive_{symbol.lower()}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            print(f"✅ Modelo adaptativo guardado en models/adaptive_{symbol.lower()}/")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

# ✅ FUNCIÓN PARA COMPARAR MODELOS
async def compare_models():
    """🧪 Comparar modelos fijos vs adaptativos"""

    print("🧪 COMPARANDO MODELOS: FIJOS vs ADAPTATIVOS")
    print("=" * 60)

    trainer = AdaptiveTCNTrainer()

    for symbol in ['BTCUSDT']:  # Test con un símbolo primero
        print(f"\n📊 Analizando {symbol}...")

        # Obtener datos
        df = await trainer.get_real_market_data(symbol, days=7)
        features = trainer.features_engine.calculate_features(df, feature_set='tcn_definitivo')

        # Thresholds fijos
        trainer.use_adaptive_thresholds = False
        df_fixed = trainer.create_balanced_labels(df, features, symbol)
        fixed_dist = df_fixed['label'].value_counts().sort_index()

        # Thresholds adaptativos
        trainer.use_adaptive_thresholds = True
        df_adaptive = trainer.create_balanced_labels(df, features, symbol)
        adaptive_dist = df_adaptive['label'].value_counts().sort_index()

        # Comparar distribuciones
        print(f"\n📊 COMPARACIÓN DE DISTRIBUCIONES:")
        class_names = ['SELL', 'HOLD', 'BUY']
        print(f"{'Clase':<8} {'Fijos':<12} {'Adaptativos':<12} {'Diferencia'}")
        print("-" * 50)

        for i, name in enumerate(class_names):
            fixed_count = fixed_dist.get(i, 0)
            adaptive_count = adaptive_dist.get(i, 0)
            diff = adaptive_count - fixed_count

            print(f"{name:<8} {fixed_count:<12} {adaptive_count:<12} {diff:+}")

async def main():
    """🎯 Entrenar modelos adaptativos"""

    print("🎯 ENTRENADOR TCN ADAPTATIVO - CAMBIOS MÍNIMOS")
    print("=" * 70)
    print("🎯 Objetivo: Mejorar modelos sin tocar el sistema de trading")
    print("=" * 70)

    trainer = AdaptiveTCNTrainer()

    # Opción 1: Comparar primero
    print("\n🧪 PASO 1: Comparando estrategias...")
    await compare_models()

    # Opción 2: Entrenar modelos mejorados
    print(f"\n🚀 PASO 2: Entrenando modelos adaptativos...")

    results = {}
    for symbol in trainer.pairs:
        success = await trainer.train_adaptive_model(symbol)
        results[symbol] = success

    print(f"\n🎯 RESUMEN:")
    for symbol, success in results.items():
        status = "✅ ÉXITO" if success else "❌ FALLO"
        print(f"   {symbol}: {status}")

    successful = sum(results.values())
    print(f"\n🎯 Modelos adaptativos entrenados: {successful}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main())
