#!/usr/bin/env python3
"""
🎯 ENTRENADOR XRP SIMPLIFICADO PERO EFECTIVO
============================================

Versión simplificada del entrenador TCN definitivo original
Optimizado para velocidad pero manteniendo efectividad
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

class SimpleXRPTrainer:
    """🎯 Entrenador XRP simplificado pero efectivo"""

    def __init__(self):
        self.pairs = ["XRPUSDT"]
        self.lookback_window = 48  # 4 horas
        self.prediction_horizon = 6  # 30 minutos

        # 🎯 THRESHOLDS OPTIMIZADOS PARA XRP
        self.thresholds = {
            'XRPUSDT': {
                'strong_sell': -0.0025,  # -0.25%
                'weak_sell': -0.0012,    # -0.12%
                'weak_buy': 0.0012,      # +0.12%
                'strong_buy': 0.0025     # +0.25%
            }
        }

    async def get_real_market_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """📊 Obtener datos reales optimizado"""

        print(f"📊 Obteniendo {days} días de datos para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
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
                        break

                await asyncio.sleep(0.1)

        # Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros de {symbol}")
        return df

    def create_66_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear exactamente 66 features (versión rápida)"""

        print("🔧 Creando 66 features técnicos...")

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = pd.DataFrame(index=df.index)

        try:
            # === MOMENTUM INDICATORS (15 features) ===
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['rsi_7'] = talib.RSI(close, timeperiod=7)

            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist

            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd

            features['williams_r'] = talib.WILLR(high, low, close)
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_20'] = talib.ROC(close, timeperiod=20)
            features['momentum_10'] = talib.MOM(close, timeperiod=10)
            features['momentum_20'] = talib.MOM(close, timeperiod=20)
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)

            # === TREND INDICATORS (12 features) ===
            features['sma_10'] = talib.SMA(close, timeperiod=10)
            features['sma_20'] = talib.SMA(close, timeperiod=20)
            features['sma_50'] = talib.SMA(close, timeperiod=50)
            features['ema_10'] = talib.EMA(close, timeperiod=10)
            features['ema_20'] = talib.EMA(close, timeperiod=20)
            features['ema_50'] = talib.EMA(close, timeperiod=50)

            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            features['psar'] = talib.SAR(high, low)

            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down

            # === VOLATILITY INDICATORS (10 features) ===
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)
            features['true_range'] = talib.TRANGE(high, low, close)
            features['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
            features['natr_20'] = talib.NATR(high, low, close, timeperiod=20)

            # === VOLUME INDICATORS (8 features) ===
            features['ad'] = talib.AD(high, low, close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            features['obv'] = talib.OBV(close, volume)

            features['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            features['volume_ratio'] = volume / features['volume_sma_20']

            features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            features['mfi_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

            # === PRICE PATTERNS (8 features) ===
            features['hl_ratio'] = (high - low) / close
            features['oc_ratio'] = (close - df['open'].values) / close
            features['price_position'] = (close - low) / (high - low)

            close_series = pd.Series(close, index=features.index)
            features['price_change_1'] = close_series.pct_change(1)
            features['price_change_5'] = close_series.pct_change(5)
            features['price_change_10'] = close_series.pct_change(10)

            returns = np.log(close_series / close_series.shift(1))
            features['volatility_10'] = returns.rolling(10).std()
            features['volatility_20'] = returns.rolling(20).std()

            # === MARKET STRUCTURE (8 features) ===
            features['higher_high'] = (pd.Series(high, index=features.index) > pd.Series(high, index=features.index).shift(1)).astype(int)
            features['lower_low'] = (pd.Series(low, index=features.index) < pd.Series(low, index=features.index).shift(1)).astype(int)

            features['uptrend_strength'] = (close_series > close_series.shift(1)).rolling(10).sum() / 10
            features['downtrend_strength'] = (close_series < close_series.shift(1)).rolling(10).sum() / 10

            features['resistance_touch'] = (close_series >= close_series.rolling(20).max() * 0.99).astype(int)
            features['support_touch'] = (close_series <= close_series.rolling(20).min() * 1.01).astype(int)

            features['efficiency_ratio'] = (np.abs(close_series - close_series.shift(10)) /
                                          (np.abs(close_series.diff()).rolling(10).sum())).fillna(0)

            features['fractal_dimension'] = 0.5  # Simplificado

            # === MOMENTUM DERIVATIVES (5 features) ===
            features['rsi_momentum'] = features['rsi_14'].diff().fillna(0)
            features['macd_momentum'] = pd.Series(macd_hist, index=features.index).diff().fillna(0)
            features['ad_momentum'] = features['ad'].diff().fillna(0)
            features['volume_momentum'] = pd.Series(volume, index=features.index).pct_change().fillna(0)
            features['price_acceleration'] = features['price_change_1'].diff().fillna(0)

            # Limpiar datos
            features = features.fillna(method='ffill').fillna(0)
            features = features.replace([np.inf, -np.inf], 0)

            # Clip valores extremos
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    q99 = features[col].quantile(0.99)
                    q01 = features[col].quantile(0.01)
                    features[col] = features[col].clip(q01, q99)

            # Asegurar exactamente 66 features
            if len(features.columns) != 66:
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]

            print(f"✅ {len(features.columns)} features creados")
            return features

        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return pd.DataFrame()

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas balanceadas"""

        print(f"🎯 Creando etiquetas balanceadas para {symbol}...")

        close_prices = df['close'].values
        thresholds = self.thresholds[symbol]

        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            future_return = (future_price - current_price) / current_price

            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
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

        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento"""

        print("🔧 Preparando datos...")

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

        print(f"✅ Datos preparados: X {X.shape}, y {y.shape}")

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_simple_tcn_model(self, input_shape: tuple) -> tf.keras.Model:
        """🎯 Crear modelo TCN simplificado pero efectivo"""

        print("🎯 Creando modelo TCN...")

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

            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo creado: {model.count_params():,} parámetros")
        return model

    async def train_simple_model(self, symbol: str) -> bool:
        """🎯 Entrenar modelo simplificado"""

        print(f"\n🎯 ENTRENANDO MODELO SIMPLIFICADO PARA {symbol}")
        print("=" * 60)

        try:
            # 1. Obtener datos
            df = await self.get_real_market_data(symbol, days=60)

            # 2. Crear features
            features = self.create_66_features(df)

            # 3. Crear labels
            df_labeled = self.create_balanced_labels(df, features, symbol)

            # 4. Preparar datos
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # 5. Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear modelo
            model = self.create_simple_tcn_model((X.shape[1], X.shape[2]))

            # 7. Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]

            # 8. Entrenar
            print("🚀 Entrenando...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,  # Reducido para velocidad
                batch_size=64,  # Más grande para velocidad
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            # 9. Evaluar
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ RESULTADOS:")
            print(f"   - Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

            # 10. Guardar
            os.makedirs('models', exist_ok=True)
            model.save('models/definitivo_xrpusdt.h5')

            with open('models/xrp_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            print(f"✅ Modelo guardado: models/definitivo_xrpusdt.h5")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

async def main():
    """🎯 Entrenar modelo XRP simplificado"""

    print("🎯 ENTRENADOR XRP SIMPLIFICADO")
    print("=" * 50)
    print("🎯 Objetivo: Modelo funcional rápido")
    print("📊 Datos: 60 días, 66 features, 48 timesteps")
    print("=" * 50)

    trainer = SimpleXRPTrainer()
    success = await trainer.train_simple_model("XRPUSDT")

    if success:
        print(f"\n🎉 ¡MODELO XRP COMPLETADO!")
        print(f"📁 models/definitivo_xrpusdt.h5")
    else:
        print(f"\n❌ FALLO EN ENTRENAMIENTO")

if __name__ == "__main__":
    asyncio.run(main()) 